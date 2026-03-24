"""discover.py — Data discovery: entity key detection, invariant scanning, residual analysis.

Populates DerivedColumn and EntityKey nodes in the memory graph.
Called via: uv run autoresearch discover

THIS FILE IS PART OF THE FIXED HARNESS. THE AGENT NEVER TOUCHES IT.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cluster import AgglomerativeClustering

from autoresearch_tabular.config import load_config
from autoresearch_tabular.memory_graph import MemoryGraph, load_graph

__all__ = ["run_discovery"]

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_DERIVED_COLS = 50
MAX_ENTITY_KEYS = 20
MAX_FALLBACK_COLS = 30      # Cap input columns before pairing in fallback
CV_THRESHOLD = 0.1
MIN_GROUP_SIZE = 10
SAMPLE_THRESHOLD = 100_000
SAMPLE_FRAC = 0.10
RESIDUAL_TREES = 50

# Semantic keyword groups for program.md parsing
SEMANTIC_KEYWORDS: dict[str, list[str]] = {
    "temporal": ["seconds", "timestamp", "timedelta", "days", "datetime", "time", "date", "elapsed"],
    "monetary": ["usd", "amount", "price", "cost", "dollar", "value", "fee", "payment"],
    "count": ["count", "frequency", "number of", "how many"],
    "distance": ["distance", "dist", "miles", "km", "meters"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _node_id_derived(expr: str) -> str:
    """Stable node ID for a derived expression."""
    return f"derived_{hashlib.sha256(expr.encode()).hexdigest()[:8]}"


def _node_id_entity(columns: tuple[str, ...]) -> str:
    """Stable node ID for an entity key."""
    return f"entity_{hashlib.sha256(str(sorted(columns)).encode()).hexdigest()[:8]}"


# ---------------------------------------------------------------------------
# 2a. Extended column profiling
# ---------------------------------------------------------------------------

def profile_columns(X_train: pd.DataFrame, mg: MemoryGraph) -> None:
    """Update existing Column nodes with richer univariate stats."""
    G = mg.graph

    # Missingness pattern clustering
    miss_matrix = X_train.isna().astype(float)
    cols_with_missing = [c for c in X_train.columns if miss_matrix[c].sum() > 0]

    missingness_groups: dict[str, int] = {}
    if len(cols_with_missing) >= 2:
        miss_corr = miss_matrix[cols_with_missing].corr().fillna(0)
        dist = (1 - miss_corr.values) / 2  # Convert correlation to distance
        np.fill_diagonal(dist, 0)
        dist = np.clip(dist, 0, None)  # Ensure non-negative

        n_clusters = min(10, len(cols_with_missing))
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage="complete",
        )
        labels = clustering.fit_predict(dist)
        for col, label in zip(cols_with_missing, labels):
            missingness_groups[col] = int(label)

    for col in X_train.columns:
        col_id = f"col_{col}"
        if not G.has_node(col_id):
            continue

        updates: dict[str, Any] = {}

        if pd.api.types.is_numeric_dtype(X_train[col]):
            valid = X_train[col].dropna()
            if len(valid) > 0:
                quantiles = valid.quantile([0.05, 0.25, 0.75, 0.95]).to_dict()
                updates["p5"] = float(quantiles.get(0.05, 0))
                updates["p25"] = float(quantiles.get(0.25, 0))
                updates["p75"] = float(quantiles.get(0.75, 0))
                updates["p95"] = float(quantiles.get(0.95, 0))
                updates["skewness"] = float(valid.skew())

        # Top-5 most frequent values
        top5 = X_train[col].value_counts().head(5)
        updates["top5_values"] = list(top5.index.astype(str))
        updates["top5_counts"] = [int(c) for c in top5.values]

        # Missingness group
        if col in missingness_groups:
            updates["missingness_group"] = missingness_groups[col]

        G.nodes[col_id].update(updates)

    mg.save()
    print(f"   Profiled {len(X_train.columns)} columns.")


# ---------------------------------------------------------------------------
# 2b. Semantic tag parsing
# ---------------------------------------------------------------------------

def _parse_semantic_tags(program_md_path: Path, columns: list[str]) -> dict[str, list[str]]:
    """Parse program.md for semantic column groupings based on keywords.

    Returns:
        Dict mapping semantic group name to list of column names.
    """
    groups: dict[str, list[str]] = {k: [] for k in SEMANTIC_KEYWORDS}

    if not program_md_path.exists():
        return groups

    text = program_md_path.read_text()
    col_set = set(columns)

    # Scan each line for column names + keyword matches
    for line in text.splitlines():
        # Only look at lines that might be table rows or descriptions
        line_lower = line.lower()

        # Find which columns are mentioned in this line
        mentioned = [c for c in col_set if f"`{c}`" in line or f"| {c} " in line or f"|{c}|" in line]
        if not mentioned:
            continue

        # Check which semantic groups this line matches
        for group, keywords in SEMANTIC_KEYWORDS.items():
            if any(kw in line_lower for kw in keywords):
                for col in mentioned:
                    if col not in groups[group]:
                        groups[group].append(col)

    return groups


# ---------------------------------------------------------------------------
# 2c. Derived column enumeration
# ---------------------------------------------------------------------------

def _enumerate_derived_columns(
    X_train: pd.DataFrame,
    semantic_groups: dict[str, list[str]],
    target: str,
) -> list[dict[str, Any]]:
    """Enumerate candidate derived columns from semantic column pairings.

    Returns list of {expr, col_a, col_b, operation, variance, series}.
    """
    candidates: list[dict[str, Any]] = []
    seen_exprs: set[str] = set()
    numeric_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c]) and c != target]

    def _add_candidate(col_a: str, col_b: str, operation: str, expr: str, series: pd.Series) -> None:
        if expr in seen_exprs:
            return
        seen_exprs.add(expr)
        var = float(series.var())
        if np.isfinite(var) and var > 0:
            candidates.append({
                "expr": expr,
                "col_a": col_a,
                "col_b": col_b,
                "operation": operation,
                "variance": var,
                "series": series,
            })

    # Pair columns within the same semantic group
    paired_something = False
    for group, cols in semantic_groups.items():
        group_numeric = [c for c in cols if c in numeric_cols]
        if len(group_numeric) < 2:
            continue
        paired_something = True

        for i, col_a in enumerate(group_numeric):
            for col_b in group_numeric[i + 1:]:
                a, b = X_train[col_a], X_train[col_b]

                # A - B
                diff = a - b
                _add_candidate(col_a, col_b, "difference", f"{col_a} - {col_b}", diff)

                # B - A
                diff_rev = b - a
                _add_candidate(col_b, col_a, "difference", f"{col_b} - {col_a}", diff_rev)

                # A / B (where B != 0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = a / b.replace(0, np.nan)
                _add_candidate(col_a, col_b, "ratio", f"{col_a} / {col_b}", ratio)

                # Temporal-specific: floor(A/86400) - B, A/86400 - B
                if group == "temporal":
                    scaled = a / 86400 - b
                    _add_candidate(col_a, col_b, "temporal_diff", f"{col_a}/86400 - {col_b}", scaled)

                    floored = np.floor(a / 86400) - b
                    _add_candidate(col_a, col_b, "temporal_floor_diff", f"floor({col_a}/86400) - {col_b}", floored)

                    # Reverse direction
                    scaled_rev = b / 86400 - a
                    _add_candidate(col_b, col_a, "temporal_diff", f"{col_b}/86400 - {col_a}", scaled_rev)

                    floored_rev = np.floor(b / 86400) - a
                    _add_candidate(col_b, col_a, "temporal_floor_diff", f"floor({col_b}/86400) - {col_a}", floored_rev)

    # Fallback: if fewer than 3 columns were classified, pair top numeric cols by variance
    total_classified = sum(len(v) for v in semantic_groups.values())
    if total_classified < 3 or not paired_something:
        variances = {c: float(X_train[c].var()) for c in numeric_cols if np.isfinite(X_train[c].var())}
        top_cols = sorted(variances, key=variances.get, reverse=True)[:MAX_FALLBACK_COLS]

        for i, col_a in enumerate(top_cols):
            for col_b in top_cols[i + 1:]:
                a, b = X_train[col_a], X_train[col_b]

                diff = a - b
                _add_candidate(col_a, col_b, "difference", f"{col_a} - {col_b}", diff)

                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = a / b.replace(0, np.nan)
                _add_candidate(col_a, col_b, "ratio", f"{col_a} / {col_b}", ratio)

    # Sort by variance descending, cap at MAX_DERIVED_COLS
    candidates.sort(key=lambda c: c["variance"], reverse=True)
    return candidates[:MAX_DERIVED_COLS]


# ---------------------------------------------------------------------------
# 2d. Entity key identification
# ---------------------------------------------------------------------------

def _identify_entity_keys(
    X_train: pd.DataFrame,
    target: str,
    categorical_columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Identify candidate entity group keys from categorical/low-cardinality columns.

    Returns list of {columns: tuple, cardinality: int}.
    """
    cat_from_config = set(categorical_columns or [])

    # Find categorical-like columns:
    # 1. Explicitly listed in config.yaml categorical_columns
    # 2. Object or category dtype
    # 3. Integer dtype with cardinality < 100k (e.g., card1, addr1 stored as int/float)
    # Exclude high-cardinality continuous numerics (V columns, C columns, D columns etc.)
    cat_candidates: list[tuple[str, int]] = []
    for col in X_train.columns:
        if col == target:
            continue
        nuniq = X_train[col].nunique()
        is_cat = (
            col in cat_from_config
            or X_train[col].dtype == "object"
            or X_train[col].dtype.name == "category"
            or (pd.api.types.is_integer_dtype(X_train[col]) and nuniq < 100_000)
            or (pd.api.types.is_float_dtype(X_train[col]) and nuniq < 1_000)
        )
        if is_cat and 50 <= nuniq <= 100_000:
            cat_candidates.append((col, nuniq))

    entity_keys: list[dict[str, Any]] = []

    # Singles
    for col, card in cat_candidates:
        entity_keys.append({"columns": (col,), "cardinality": card})

    # Pairs — only from columns with individual cardinality [10, 50_000]
    pair_eligible = [(c, n) for c, n in cat_candidates if 10 <= n <= 50_000]
    for i, (col_a, _) in enumerate(pair_eligible):
        for col_b, _ in pair_eligible[i + 1:]:
            combined = X_train.groupby([col_a, col_b]).ngroups
            if 500 <= combined <= 500_000:
                entity_keys.append({
                    "columns": (col_a, col_b),
                    "cardinality": combined,
                })

    # Sort by cardinality descending — higher cardinality entity keys are typically
    # more useful (they represent finer-grained identities like users/cards).
    entity_keys.sort(key=lambda e: e["cardinality"], reverse=True)
    return entity_keys[:MAX_ENTITY_KEYS]


# ---------------------------------------------------------------------------
# 2e. Within-group variance scan
# ---------------------------------------------------------------------------

def _variance_scan(
    X_train: pd.DataFrame,
    derived_exprs: list[dict[str, Any]],
    entity_keys: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Scan for (expression, entity_key) pairs where the expression is near-constant within groups.

    Returns list of {expr, node_id, entity_columns, median_cv, n_groups, is_raw}.
    """
    if not entity_keys:
        return []

    # Sample for large datasets
    df = X_train
    if len(df) > SAMPLE_THRESHOLD:
        df = df.sample(frac=SAMPLE_FRAC, random_state=42)
        print(f"   Sampled {len(df):,} rows for variance scan.")

    # Build expression Series dict: derived + raw numeric (top 50 by variance)
    expr_series: dict[str, tuple[pd.Series, str, bool]] = {}  # expr -> (series, node_id, is_raw)

    # Derived columns
    for d in derived_exprs:
        series = d["series"]
        if len(df) < len(X_train):
            # Recompute on sample
            series = series.reindex(df.index)
        expr_series[d["expr"]] = (series, _node_id_derived(d["expr"]), False)

    # Raw numeric columns (top 50 by variance)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    variances = {c: float(df[c].var()) for c in numeric_cols if np.isfinite(df[c].var())}
    top_raw = sorted(variances, key=variances.get, reverse=True)[:50]
    for col in top_raw:
        expr_series[col] = (df[col], f"col_{col}", True)

    invariants: list[dict[str, Any]] = []

    for ek in entity_keys:
        ek_cols = list(ek["columns"])

        # Precompute group sizes for this entity key
        group_sizes = df.groupby(ek_cols).size()
        valid_groups = group_sizes[group_sizes >= MIN_GROUP_SIZE].index

        if len(valid_groups) == 0:
            continue

        for expr_name, (series, node_id, is_raw) in expr_series.items():
            try:
                # Build a frame for groupby
                tmp = pd.DataFrame({"_val": series, **{c: df[c] for c in ek_cols}})
                tmp = tmp.dropna(subset=["_val"])

                grouped = tmp.groupby(ek_cols)["_val"]
                stats = grouped.agg(["std", "mean", "count"])
                stats = stats[stats["count"] >= MIN_GROUP_SIZE]

                if len(stats) == 0:
                    continue

                # CV = std / |mean|, handle zero-mean groups
                with np.errstate(divide="ignore", invalid="ignore"):
                    cvs = stats["std"] / stats["mean"].abs()

                # Drop inf/nan CVs
                cvs = cvs.replace([np.inf, -np.inf], np.nan).dropna()
                if len(cvs) == 0:
                    continue

                median_cv = float(cvs.median())

                if median_cv < CV_THRESHOLD:
                    invariants.append({
                        "expr": expr_name,
                        "node_id": node_id,
                        "entity_columns": ek["columns"],
                        "median_cv": median_cv,
                        "n_groups": int(len(stats)),
                        "is_raw": is_raw,
                    })
            except Exception:
                continue

    # Sort by median CV ascending (best invariants first)
    invariants.sort(key=lambda x: x["median_cv"])
    return invariants


# ---------------------------------------------------------------------------
# 2f. Residual-guided ICC
# ---------------------------------------------------------------------------

def _compute_icc(residuals: np.ndarray, groups: np.ndarray) -> float:
    """Compute intraclass correlation coefficient (ICC).

    ICC = (MSB - MSW) / (MSB + (k-1)*MSW)
    where MSB = between-group mean square, MSW = within-group mean square, k = mean group size.
    """
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return 0.0

    grand_mean = residuals.mean()
    n_total = len(residuals)
    n_groups = len(unique_groups)

    # Between-group and within-group sums of squares
    ssb = 0.0
    ssw = 0.0
    total_n = 0
    for g in unique_groups:
        mask = groups == g
        group_vals = residuals[mask]
        n_g = len(group_vals)
        if n_g == 0:
            continue
        group_mean = group_vals.mean()
        ssb += n_g * (group_mean - grand_mean) ** 2
        ssw += ((group_vals - group_mean) ** 2).sum()
        total_n += n_g

    df_between = n_groups - 1
    df_within = total_n - n_groups

    if df_between <= 0 or df_within <= 0:
        return 0.0

    msb = ssb / df_between
    msw = ssw / df_within
    k = total_n / n_groups  # mean group size

    denom = msb + (k - 1) * msw
    if denom <= 0:
        return 0.0

    return float((msb - msw) / denom)


def _compute_residual_icc(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    entity_keys: list[dict[str, Any]],
    config: Any,
) -> dict[tuple[str, ...], float]:
    """Train a quick XGBoost on raw features, compute residual ICC per entity key.

    Returns dict mapping entity_columns tuple to ICC value.
    """
    if not entity_keys:
        return {}

    # Label-encode string targets for XGBoost
    if not pd.api.types.is_numeric_dtype(y_train):
        from sklearn.preprocessing import LabelEncoder
        _le = LabelEncoder()
        y_train = pd.Series(_le.fit_transform(y_train), index=y_train.index)

    # Raw numeric features only, NaN → median
    numeric_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    X_numeric = X_train[numeric_cols].copy()
    medians = X_numeric.median()
    X_numeric = X_numeric.fillna(medians)

    # Replace any remaining inf
    X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Fit XGBoost based on task type
    is_classification = config.is_classification
    n_classes = y_train.nunique()

    if is_classification and n_classes > 2:
        # Multi-class: predict probability of true class, residual = 1 - p(true)
        model = xgb.XGBClassifier(
            n_estimators=RESIDUAL_TREES, learning_rate=0.05,
            random_state=42, n_jobs=1, verbosity=0,
            objective="multi:softprob",
        )
        model.fit(X_numeric, y_train)
        probs = model.predict_proba(X_numeric)
        # 1 - p(true_class) for each row
        residuals = np.array([
            1.0 - probs[i, int(y_train.iloc[i])]
            for i in range(len(y_train))
        ])
    elif is_classification:
        # Binary: residual = y - p(positive)
        model = xgb.XGBClassifier(
            n_estimators=RESIDUAL_TREES, learning_rate=0.05,
            random_state=42, n_jobs=1, verbosity=0,
        )
        model.fit(X_numeric, y_train)
        probs = model.predict_proba(X_numeric)[:, 1]
        residuals = y_train.values - probs
    else:
        # Regression: residual = y - y_pred
        model = xgb.XGBRegressor(
            n_estimators=RESIDUAL_TREES, learning_rate=0.05,
            random_state=42, n_jobs=1, verbosity=0,
        )
        model.fit(X_numeric, y_train)
        preds = model.predict(X_numeric)
        residuals = y_train.values - preds

    # Compute ICC per entity key
    icc_results: dict[tuple[str, ...], float] = {}
    for ek in entity_keys:
        ek_cols = list(ek["columns"])
        # Build group labels
        if len(ek_cols) == 1:
            group_labels = X_train[ek_cols[0]].fillna(-999).astype(str).values
        else:
            parts = [X_train[c].fillna(-999).astype(str) for c in ek_cols]
            group_labels = parts[0].str.cat(parts[1:], sep="_").values
        icc = _compute_icc(residuals, group_labels)
        icc_results[ek["columns"]] = icc

    return icc_results


# ---------------------------------------------------------------------------
# 2g. Graph population
# ---------------------------------------------------------------------------

def _populate_graph(
    mg: MemoryGraph,
    derived_exprs: list[dict[str, Any]],
    entity_keys: list[dict[str, Any]],
    invariants: list[dict[str, Any]],
    icc_results: dict[tuple[str, ...], float],
) -> None:
    """Create DerivedColumn and EntityKey nodes, add all discovery edges."""
    G = mg.graph

    # Create DerivedColumn nodes
    for d in derived_exprs:
        node_id = _node_id_derived(d["expr"])
        G.add_node(
            node_id,
            node_type="DerivedColumn",
            name=d["expr"],
            expr=d["expr"],
            col_a=d["col_a"],
            col_b=d["col_b"],
            operation=d["operation"],
            variance=float(d["variance"]),
        )
        # DERIVED_FROM_COLS edges
        col_a_id = f"col_{d['col_a']}"
        col_b_id = f"col_{d['col_b']}"
        if G.has_node(col_a_id):
            G.add_edge(node_id, col_a_id, rel="DERIVED_FROM_COLS")
        if G.has_node(col_b_id):
            G.add_edge(node_id, col_b_id, rel="DERIVED_FROM_COLS")

    # Create EntityKey nodes
    for ek in entity_keys:
        node_id = _node_id_entity(ek["columns"])
        icc = icc_results.get(ek["columns"])
        G.add_node(
            node_id,
            node_type="EntityKey",
            name=", ".join(ek["columns"]),
            columns=list(ek["columns"]),
            cardinality=int(ek["cardinality"]),
            residual_icc=float(icc) if icc is not None else None,
        )
        # CANDIDATE_ENTITY_KEY edges
        for col in ek["columns"]:
            col_id = f"col_{col}"
            if G.has_node(col_id):
                G.add_edge(node_id, col_id, rel="CANDIDATE_ENTITY_KEY")

    # INVARIANT_WITHIN edges
    for inv in invariants:
        src_id = inv["node_id"]
        tgt_id = _node_id_entity(inv["entity_columns"])
        if G.has_node(src_id) and G.has_node(tgt_id):
            G.add_edge(
                src_id, tgt_id,
                rel="INVARIANT_WITHIN",
                median_cv=float(inv["median_cv"]),
                n_groups=int(inv["n_groups"]),
            )

    mg.save()


# ---------------------------------------------------------------------------
# 2h. Main entry point
# ---------------------------------------------------------------------------

def run_discovery() -> dict[str, Any]:
    """Run the full discovery pipeline. Idempotent — safe to re-run."""
    t0 = time.time()

    print("Running data discovery ...")

    # Initialize data pipeline (same pattern as cmd_prepare)
    import autoresearch_tabular.prepare as prepare
    prepare._initialize()

    config = load_config()
    folds = prepare.get_folds()
    X_train, _, y_train, _ = folds[0]

    print(f"   X_train: {X_train.shape[0]:,} rows × {X_train.shape[1]} columns")

    # Load graph and clear previous discovery nodes
    mg = load_graph()
    n_cleared = mg.clear_discovery_nodes()
    if n_cleared:
        print(f"   Cleared {n_cleared} previous discovery nodes.")

    # Also clear query log for new session
    query_log = PROJECT_ROOT / "db" / "query_log.json"
    if query_log.exists():
        query_log.unlink()

    # 2a. Extended column profiling
    print("\n[1/5] Profiling columns ...")
    profile_columns(X_train, mg)

    # 2b. Parse semantic tags
    print("[2/5] Parsing semantic tags from program.md ...")
    program_md = PROJECT_ROOT / "program.md"
    semantic_groups = _parse_semantic_tags(program_md, list(X_train.columns))
    for group, cols in semantic_groups.items():
        if cols:
            print(f"   {group}: {', '.join(cols)}")

    # 2c. Enumerate derived columns
    print("[3/5] Enumerating derived columns ...")
    derived_exprs = _enumerate_derived_columns(X_train, semantic_groups, config.target)
    print(f"   {len(derived_exprs)} candidate derived columns.")

    # 2d. Identify entity keys
    print("[4/5] Identifying entity keys ...")
    entity_keys = _identify_entity_keys(X_train, config.target, config.categorical_columns)
    print(f"   {len(entity_keys)} candidate entity keys.")
    for ek in entity_keys:
        print(f"     ({', '.join(ek['columns'])})  cardinality={ek['cardinality']:,}")

    # 2e. Within-group variance scan
    print("[5/5] Scanning for invariant expressions ...")
    invariants = _variance_scan(X_train, derived_exprs, entity_keys)
    print(f"   {len(invariants)} invariant (expression, entity key) pairs found.")

    # 2f. Residual ICC
    print("\n   Computing residual ICC ...")
    icc_results = _compute_residual_icc(X_train, y_train, entity_keys, config)
    for ek_cols, icc in sorted(icc_results.items(), key=lambda x: -x[1]):
        print(f"     ({', '.join(ek_cols)})  ICC={icc:.4f}")

    # 2g. Populate graph
    # Strip series from derived_exprs before passing to graph (not serializable)
    for d in derived_exprs:
        d.pop("series", None)
    _populate_graph(mg, derived_exprs, entity_keys, invariants, icc_results)

    elapsed = time.time() - t0
    print(f"\nDiscovery complete in {elapsed:.1f}s.")
    print(f"   DerivedColumn nodes: {len(derived_exprs)}")
    print(f"   EntityKey nodes:     {len(entity_keys)}")
    print(f"   Invariant pairs:     {len(invariants)}")

    if invariants:
        print("\n   Top invariants:")
        for inv in invariants[:10]:
            print(f"     {inv['expr']:<50}  within ({', '.join(inv['entity_columns'])})  CV={inv['median_cv']:.4f}")

    return {
        "n_derived": len(derived_exprs),
        "n_entity_keys": len(entity_keys),
        "n_invariants": len(invariants),
        "elapsed_seconds": elapsed,
    }
