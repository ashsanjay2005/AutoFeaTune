"""memory_graph.py — NetworkX-based experiment knowledge graph.

Node types:
    Column      — raw source columns from the dataset
    Feature     — engineered features created by the agent
    Experiment  — a single evaluation run with its scores
    Hypothesis  — the agent's reasoning about what worked/failed

Edge types (builtin — see relationship_registry for dynamic types):
    DERIVED_FROM   — Feature → Column or Feature (lineage)
    USED_IN        — Feature → Experiment (direct link)
    IMPROVED_OVER  — Experiment → previous Experiment
    SUPPORTS       — Hypothesis → Experiment
    CONTRADICTS    — Hypothesis → Experiment
    SUPERSEDES     — Hypothesis → Hypothesis

The relationship registry (stored in graph.graph["relationship_registry"])
allows new edge types to be registered at runtime. Traversal code should
use category-based lookups instead of hardcoded rel type names.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

__all__ = ["MemoryGraph", "load_graph", "DEFAULT_GRAPH_PATH"]

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_GRAPH_PATH = PROJECT_ROOT / "db" / "memory_graph.json"

# Transform families as a module-level constant (replaces seeded Transformation nodes).
TRANSFORM_FAMILIES: dict[str, dict[str, str]] = {
    "log": {"category": "mathematical", "description": "Logarithmic transform (log1p) — reduces right skew"},
    "sqrt": {"category": "mathematical", "description": "Square root transform — moderate skew reduction"},
    "power": {"category": "mathematical", "description": "Power transform (square, cube) — amplifies differences"},
    "reciprocal": {"category": "mathematical", "description": "Reciprocal (1/x) — inverts scale"},
    "abs": {"category": "mathematical", "description": "Absolute value — removes sign"},
    "clip": {"category": "mathematical", "description": "Clip/winsorize — caps outliers at percentiles"},
    "normalize": {"category": "mathematical", "description": "Min-max or z-score normalization"},
    "day_of_week": {"category": "temporal", "description": "Extract day of week from datetime"},
    "month": {"category": "temporal", "description": "Extract month from datetime"},
    "year": {"category": "temporal", "description": "Extract year from datetime"},
    "time_since": {"category": "temporal", "description": "Elapsed time since a reference date"},
    "is_weekend": {"category": "temporal", "description": "Binary flag for weekend days"},
    "season": {"category": "temporal", "description": "Map month to season"},
    "target_encoding": {"category": "categorical", "description": "Encode category with mean target (fit on train only)"},
    "frequency_encoding": {"category": "categorical", "description": "Encode category with its frequency"},
    "onehot": {"category": "categorical", "description": "One-hot encoding — creates binary columns per category"},
    "ordinal": {"category": "categorical", "description": "Map ordered categories to integers"},
    "label": {"category": "categorical", "description": "Simple integer label encoding"},
    "count_encoding": {"category": "categorical", "description": "Encode category with occurrence count"},
    "product": {"category": "interaction", "description": "Multiply two features together"},
    "ratio": {"category": "interaction", "description": "Divide one feature by another"},
    "difference": {"category": "interaction", "description": "Subtract one feature from another"},
    "sum": {"category": "interaction", "description": "Add two features together"},
    "quantile_bin": {"category": "binning", "description": "Bin into equal-frequency quantile buckets"},
    "equal_width_bin": {"category": "binning", "description": "Bin into equal-width buckets"},
    "custom_bin": {"category": "binning", "description": "Domain-specific custom bin edges"},
}

# ---------------------------------------------------------------------------
# Relationship registry — seed for builtin edge types
# ---------------------------------------------------------------------------

_RELATIONSHIP_SEED: list[dict[str, Any]] = [
    {
        "rel_type": "DERIVED_FROM",
        "source_type": "Feature",
        "target_type": ["Column", "Feature"],
        "category": "lineage",
        "builtin": True,
        "description": "Feature lineage: derived from Column or Feature",
    },
    {
        "rel_type": "USED_IN",
        "source_type": "Feature",
        "target_type": ["Experiment"],
        "category": "membership",
        "builtin": True,
        "description": "Feature used in Experiment",
    },
    {
        "rel_type": "IMPROVED_OVER",
        "source_type": "Experiment",
        "target_type": ["Experiment"],
        "category": "improvement",
        "builtin": True,
        "description": "Experiment improved over previous best",
    },
    {
        "rel_type": "SUPPORTS",
        "source_type": "Hypothesis",
        "target_type": ["Experiment"],
        "category": "hypothesis",
        "builtin": True,
        "description": "Hypothesis supported by experiment outcome",
    },
    {
        "rel_type": "CONTRADICTS",
        "source_type": "Hypothesis",
        "target_type": ["Experiment"],
        "category": "hypothesis",
        "builtin": True,
        "description": "Hypothesis contradicted by experiment outcome",
    },
    {
        "rel_type": "SUPERSEDES",
        "source_type": "Hypothesis",
        "target_type": ["Hypothesis"],
        "category": "hypothesis",
        "builtin": True,
        "description": "Hypothesis replaces an older one",
    },
    # --- Discovery edge types ---
    {
        "rel_type": "INVARIANT_WITHIN",
        "source_type": "DerivedColumn",
        "target_type": ["EntityKey"],
        "category": "discovery",
        "builtin": True,
        "description": "Expression is near-constant within entity groups (CV < 0.1)",
    },
    {
        "rel_type": "CANDIDATE_ENTITY_KEY",
        "source_type": "EntityKey",
        "target_type": ["Column"],
        "category": "discovery",
        "builtin": True,
        "description": "Entity key composed from source column(s)",
    },
    {
        "rel_type": "DERIVED_FROM_COLS",
        "source_type": "DerivedColumn",
        "target_type": ["Column"],
        "category": "discovery",
        "builtin": True,
        "description": "Derived expression computed from source columns",
    },
]


class MemoryGraph:
    """NetworkX DiGraph with typed nodes and JSON persistence.

    All nodes carry a ``node_type`` attribute that identifies the type.
    All edges carry a ``rel`` attribute for the relationship type.

    Node ID conventions:
        col_<name>     — Column
        feat_<name>    — Feature
        exp_<id>       — Experiment
        hyp_<id>       — Hypothesis
    """

    def __init__(self, path: Path = DEFAULT_GRAPH_PATH) -> None:
        self.path = path
        self.graph: nx.DiGraph = nx.DiGraph()
        if path.exists():
            self._load()
            self._backfill_relationship_registry()
        else:
            self._seed_relationships()

    # -------------------------------------------------------------------------
    # Dataset identity / reset
    # -------------------------------------------------------------------------

    def ensure_dataset_signature(
        self,
        signature: str,
        meta: dict[str, Any] | None = None,
        backup_on_change: bool = True,
    ) -> bool:
        """Ensure the on-disk graph matches the current dataset signature.

        If the dataset changes, the existing graph is reset.

        Returns:
            True if the graph was reset due to signature mismatch, else False.
        """
        prior = self.graph.graph.get("dataset_signature")
        if prior == signature:
            return False

        if prior is None:
            self.graph.graph["dataset_signature"] = signature
            if meta is not None:
                self.graph.graph["dataset_meta"] = meta
            self.save()
            return False

        if backup_on_change and self.path.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            prior_short = str(prior)[:8]
            backup_path = self.path.with_name(f"memory_graph.{prior_short}.{ts}.json")
            try:
                shutil.move(str(self.path), str(backup_path))
            except Exception:
                pass

        self.graph = nx.DiGraph()
        self.graph.graph["dataset_signature"] = signature
        if meta is not None:
            self.graph.graph["dataset_meta"] = meta
        self.save()
        return True

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self) -> None:
        """Persist the graph to JSON."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.graph)
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load(self) -> None:
        with open(self.path) as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data, directed=True, multigraph=False)

    # -------------------------------------------------------------------------
    # Seeding
    # -------------------------------------------------------------------------

    def _seed_relationships(self) -> None:
        """Seed the relationship registry from _RELATIONSHIP_SEED."""
        registry: dict[str, dict[str, Any]] = {}
        for entry in _RELATIONSHIP_SEED:
            registry[entry["rel_type"]] = dict(entry)
        self.graph.graph["relationship_registry"] = registry

    def _backfill_relationship_registry(self) -> None:
        """Ensure loaded graphs have a relationship_registry (migration)."""
        if "relationship_registry" in self.graph.graph:
            registry = self.graph.graph["relationship_registry"]
            changed = False
            for entry in _RELATIONSHIP_SEED:
                if entry["rel_type"] not in registry:
                    registry[entry["rel_type"]] = dict(entry)
                    changed = True
            # Migrate old MEMBER_OF/TESTED_IN entries to USED_IN
            for old_type in ("MEMBER_OF", "TESTED_IN"):
                if old_type in registry and "USED_IN" in registry:
                    del registry[old_type]
                    changed = True
            if changed:
                self.save()
            return
        self._seed_relationships()
        self.save()

    # -------------------------------------------------------------------------
    # Relationship registry
    # -------------------------------------------------------------------------

    def get_relationship_types(
        self, category: str | None = None
    ) -> list[dict[str, Any]]:
        """Return registered relationship types, optionally filtered by category."""
        registry = self.graph.graph.get("relationship_registry", {})
        entries = list(registry.values())
        if category is not None:
            entries = [e for e in entries if e.get("category") == category]
        return sorted(entries, key=lambda e: e["rel_type"])

    def get_rel_types_for_category(self, category: str) -> set[str]:
        """Return the set of rel_type strings for a given category."""
        return {
            e["rel_type"]
            for e in self.get_relationship_types(category=category)
        }

    def get_edges_by_type(self, rel_type: str) -> list[tuple[str, str, dict]]:
        """Return all edges with the given rel attribute."""
        return [
            (u, v, d)
            for u, v, d in self.graph.edges(data=True)
            if d.get("rel") == rel_type
        ]

    def register_relationship_type(
        self,
        rel_type: str,
        description: str,
        source_type: str,
        target_type: list[str],
        category: str,
    ) -> str:
        """Register a new relationship type in the registry."""
        registry = self.graph.graph.get("relationship_registry", {})
        if rel_type not in registry:
            registry[rel_type] = {
                "rel_type": rel_type,
                "source_type": source_type,
                "target_type": target_type,
                "category": category,
                "builtin": False,
                "description": description,
            }
            self.graph.graph["relationship_registry"] = registry
            self.save()
        return rel_type

    def add_edge_typed(
        self,
        source: str,
        target: str,
        rel_type: str,
        save: bool = True,
        **attrs: Any,
    ) -> bool:
        """Add an edge with a rel_type, auto-registering if unknown."""
        registry = self.graph.graph.get("relationship_registry", {})
        if rel_type not in registry:
            src_type = self.graph.nodes.get(source, {}).get("node_type", "Unknown")
            tgt_type = self.graph.nodes.get(target, {}).get("node_type", "Unknown")
            self.register_relationship_type(
                rel_type=rel_type,
                description="Auto-registered from edge creation",
                source_type=src_type,
                target_type=[tgt_type],
                category="auto",
            )

        if not self.graph.has_node(source) or not self.graph.has_node(target):
            return False

        self.graph.add_edge(source, target, rel=rel_type, **attrs)
        if save:
            self.save()
        return True

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _nodes_of_type(self, node_type: str) -> list[dict[str, Any]]:
        """Return attribute dicts for all nodes with the given node_type."""
        return [
            d
            for _, d in self.graph.nodes(data=True)
            if d.get("node_type") == node_type
        ]

    def _next_id(self, prefix: str) -> int:
        """Return max(existing IDs with prefix) + 1, or 1 if none exist."""
        ids = [
            int(n.split("_", 1)[1])
            for n in self.graph.nodes
            if n.startswith(prefix + "_")
        ]
        return max(ids) + 1 if ids else 1

    def _tried_columns(self) -> set[str]:
        """Return names of Column nodes reachable via DERIVED_FROM from any Feature."""
        tried: set[str] = set()
        for _, feat_data in self.graph.nodes(data=True):
            if feat_data.get("node_type") != "Feature":
                continue
            feat_id = f"feat_{feat_data['name']}"
            for neighbor in self.graph.successors(feat_id):
                nd = self.graph.nodes.get(neighbor, {})
                if nd.get("node_type") == "Column":
                    tried.add(nd.get("name", ""))
        return tried

    # -------------------------------------------------------------------------
    # Source column population
    # -------------------------------------------------------------------------

    def populate_source_columns(self, df: pd.DataFrame) -> None:
        """Add Column nodes from a DataFrame (idempotent)."""
        existing = self._nodes_of_type("Column")
        if existing:
            return

        for col in df.columns:
            dtype_str = str(df[col].dtype)
            cardinality = int(df[col].nunique())
            n = len(df[col])
            missingness = float(df[col].isna().sum() / n) if n > 0 else 0.0

            if pd.api.types.is_numeric_dtype(df[col]):
                col_mean = float(df[col].mean()) if not df[col].isna().all() else 0.0
                col_std = float(df[col].std()) if not df[col].isna().all() else 0.0
            else:
                col_mean = 0.0
                col_std = 0.0

            self.graph.add_node(
                f"col_{col}",
                node_type="Column",
                name=col,
                dtype=dtype_str,
                cardinality=cardinality,
                missingness_rate=missingness,
                mean=col_mean,
                std=col_std,
            )

        self.save()

    # -------------------------------------------------------------------------
    # Experiment recording — trusted write path (only train.py calls this)
    # -------------------------------------------------------------------------

    def get_next_experiment_id(self) -> int:
        """Return the next available experiment ID."""
        return self._next_id("exp")

    def record_experiment(
        self,
        cv_score: float,
        cv_std: float,
        delta: float,
        n_features: int,
        composite_score: float,
        kept: bool,
        description: str = "",
        features_used: list[str] | None = None,
        feature_shap: dict[str, float] | None = None,
        feature_shap_std: dict[str, float] | None = None,
    ) -> int:
        """Record an experiment result. THIS IS THE TRUSTED WRITE PATH."""
        exp_id = self.get_next_experiment_id()
        node_id = f"exp_{exp_id}"
        timestamp = datetime.now().isoformat()

        self.graph.add_node(
            node_id,
            node_type="Experiment",
            exp_id=exp_id,
            cv_score=float(cv_score),
            cv_std=float(cv_std),
            delta=float(delta),
            n_features=int(n_features),
            composite_score=float(composite_score),
            kept=bool(kept),
            description=str(description),
            timestamp=timestamp,
            features_used=list(features_used or []),
            feature_shap=dict(feature_shap) if feature_shap else {},
            feature_shap_std=dict(feature_shap_std) if feature_shap_std else {},
        )

        if kept and exp_id > 1:
            prev = self._get_best_kept_before(exp_id)
            if prev:
                self.graph.add_edge(
                    node_id,
                    f"exp_{prev['exp_id']}",
                    rel="IMPROVED_OVER",
                    delta=float(delta),
                )

        self.save()
        return exp_id

    # -------------------------------------------------------------------------
    # Feature-Experiment registration (direct edges, no FeatureSet intermediary)
    # -------------------------------------------------------------------------

    def register_feature_set(self, experiment_id: int, features_used: list[str]) -> None:
        """Link Feature nodes directly to an Experiment via USED_IN edges.

        Creates Feature nodes if they don't exist, then adds
        feat_<name> → exp_<id> edges with rel="USED_IN".
        """
        exp_node = f"exp_{experiment_id}"
        if not self.graph.has_node(exp_node):
            return

        for feat in features_used:
            feat_node = f"feat_{feat}"
            if not self.graph.has_node(feat_node):
                self.graph.add_node(
                    feat_node,
                    node_type="Feature",
                    name=feat,
                    status="active",
                    created_at_experiment=int(experiment_id),
                )
            if not self.graph.has_edge(feat_node, exp_node):
                self.graph.add_edge(feat_node, exp_node, rel="USED_IN")

        self.save()

    def _get_best_kept_before(self, current_id: int) -> dict[str, Any] | None:
        """Return the most recent kept experiment with exp_id < current_id."""
        candidates = [
            d
            for d in self._nodes_of_type("Experiment")
            if d.get("kept", False)
            and d.get("exp_id") is not None
            and d["exp_id"] < current_id
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda x: x["exp_id"])

    # -------------------------------------------------------------------------
    # Feature registration
    # -------------------------------------------------------------------------

    def register_feature(
        self,
        name: str,
        sources: list[str],
        experiment_id: int,
        save: bool = True,
    ) -> None:
        """Register a feature node and its DERIVED_FROM lineage edges."""
        node_id = f"feat_{name}"
        if not self.graph.has_node(node_id):
            self.graph.add_node(
                node_id,
                node_type="Feature",
                name=name,
                status="active",
                created_at_experiment=experiment_id,
            )

        for src in sources:
            src_col_id = f"col_{src}"
            if self.graph.has_node(src_col_id):
                if not self.graph.has_edge(node_id, src_col_id):
                    self.graph.add_edge(node_id, src_col_id, rel="DERIVED_FROM")
                continue

            if src == name:
                continue

            src_feat_id = f"feat_{src}"
            if self.graph.has_node(src_feat_id):
                if not self.graph.has_edge(node_id, src_feat_id):
                    self.graph.add_edge(node_id, src_feat_id, rel="DERIVED_FROM")

        if save:
            self.save()

    def update_active_feature_statuses(self, current_feature_columns: list[str]) -> None:
        """Sync Feature node statuses to match what is actually in features.py now."""
        current_set = set(current_feature_columns)
        changed = False
        for node_id, data in self.graph.nodes(data=True):
            if data.get("node_type") != "Feature":
                continue
            name = data.get("name", "")
            new_status = "active" if name in current_set else "inactive"
            if data.get("status") != new_status:
                self.graph.nodes[node_id]["status"] = new_status
                changed = True
        if changed:
            self.save()

    # -------------------------------------------------------------------------
    # Discovery management
    # -------------------------------------------------------------------------

    def clear_discovery_nodes(self) -> int:
        """Remove all DerivedColumn and EntityKey nodes and their edges.

        Enables idempotent re-runs of ``discover``.

        Returns:
            Count of nodes removed.
        """
        to_remove = [
            n for n, d in self.graph.nodes(data=True)
            if d.get("node_type") in ("DerivedColumn", "EntityKey")
        ]
        self.graph.remove_nodes_from(to_remove)
        if to_remove:
            self.save()
        return len(to_remove)

    def get_discovery_summary(self) -> dict[str, Any]:
        """Return summary statistics for discovery nodes."""
        derived = self._nodes_of_type("DerivedColumn")
        entities = self._nodes_of_type("EntityKey")
        invariant_edges = self.get_edges_by_type("INVARIANT_WITHIN")
        return {
            "n_derived_columns": len(derived),
            "n_entity_keys": len(entities),
            "n_invariant_pairs": len(invariant_edges),
            "derived_columns": derived,
            "entity_keys": entities,
        }

    # -------------------------------------------------------------------------
    # Hypothesis management
    # -------------------------------------------------------------------------

    def add_hypothesis(
        self,
        text: str,
        predicted_direction: str = "+",
        predicted_delta: float | None = None,
    ) -> int:
        """Record a pre-experiment hypothesis with a testable prediction."""
        hyp_id = self._next_id("hyp")
        node_id = f"hyp_{hyp_id}"

        self.graph.add_node(
            node_id,
            node_type="Hypothesis",
            hyp_id=hyp_id,
            text=str(text),
            predicted_direction=predicted_direction,
            predicted_delta=predicted_delta,
            validated=False,
            created_at=datetime.now().isoformat(),
        )

        self.save()
        return hyp_id

    def resolve_hypothesis(
        self,
        hyp_id: int,
        experiment_id: int,
        kept: bool,
        actual_delta: float,
    ) -> None:
        """Resolve a pending hypothesis against a completed experiment."""
        node_id = f"hyp_{hyp_id}"
        if not self.graph.has_node(node_id):
            return

        predicted = self.graph.nodes[node_id].get("predicted_direction", "+")
        if predicted == "?":
            correct = True
        else:
            correct = (predicted == "+" and kept) or (predicted == "-" and not kept)

        rel = "SUPPORTS" if correct else "CONTRADICTS"

        self.graph.nodes[node_id]["validated"] = True
        self.graph.nodes[node_id]["actual_kept"] = kept
        self.graph.nodes[node_id]["actual_delta"] = actual_delta
        self.graph.nodes[node_id]["created_at_experiment"] = experiment_id

        exp_node = f"exp_{experiment_id}"
        if self.graph.has_node(exp_node):
            if not self.graph.has_edge(node_id, exp_node):
                self.graph.add_edge(node_id, exp_node, rel=rel)

        self.save()

    def supersede_hypothesis(self, old_hyp_id: int, new_hyp_id: int) -> None:
        """Mark old_hyp_id as superseded by new_hyp_id."""
        new_node = f"hyp_{new_hyp_id}"
        old_node = f"hyp_{old_hyp_id}"
        if self.graph.has_node(new_node) and self.graph.has_node(old_node):
            if not self.graph.has_edge(new_node, old_node):
                self.graph.add_edge(new_node, old_node, rel="SUPERSEDES")
        self.save()

    # -------------------------------------------------------------------------
    # Query functions
    # -------------------------------------------------------------------------

    def get_experiment_history(self, n: int = 20) -> list[dict[str, Any]]:
        """Return the N most recent experiments, most recent first."""
        exps = self._nodes_of_type("Experiment")
        exps.sort(key=lambda x: x.get("exp_id", 0), reverse=True)
        return exps[:n]

    def get_best_experiment(self, is_higher_better: bool = True) -> dict[str, Any] | None:
        """Return the best kept experiment."""
        kept = [d for d in self._nodes_of_type("Experiment") if d.get("kept", False)]
        if not kept:
            return None
        if is_higher_better:
            return max(kept, key=lambda x: x.get("composite_score", 0.0))
        else:
            return min(kept, key=lambda x: x.get("composite_score", float("inf")))

    def get_active_features(self) -> list[dict[str, Any]]:
        """Return all Feature nodes with status='active'."""
        return [d for d in self._nodes_of_type("Feature") if d.get("status") == "active"]

    def get_active_hypotheses(self) -> list[dict[str, Any]]:
        """Return all Hypothesis nodes that have not been superseded."""
        superseded = {
            v
            for u, v, d in self.graph.edges(data=True)
            if d.get("rel") == "SUPERSEDES"
        }

        result = []
        for d in self._nodes_of_type("Hypothesis"):
            node_id = f"hyp_{d['hyp_id']}"
            if node_id in superseded:
                continue
            edge_type = None
            for _, _, edata in self.graph.out_edges(node_id, data=True):
                if edata.get("rel") in ("SUPPORTS", "CONTRADICTS"):
                    edge_type = edata["rel"]
                    break
            entry = dict(d)
            entry["edge_type"] = edge_type
            result.append(entry)
        return result

    def get_feature_lineage(self, feature_name: str) -> list[dict[str, Any]]:
        """Return all nodes in the lineage of a feature (BFS from feature node)."""
        feat_id = f"feat_{feature_name}"
        if not self.graph.has_node(feat_id):
            return []

        lineage: list[dict[str, Any]] = []
        visited: set[str] = set()
        queue = [feat_id]
        while queue:
            current = queue.pop()
            for neighbor in self.graph.successors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    node_data = self.graph.nodes[neighbor]
                    lineage.append(
                        {
                            "node_id": neighbor,
                            "node_type": node_data.get("node_type", "Unknown"),
                            "name": node_data.get("name", neighbor),
                        }
                    )
                    queue.append(neighbor)
        return lineage

    def get_source_columns(self) -> list[dict[str, Any]]:
        """Return all Column nodes."""
        return self._nodes_of_type("Column")

    def get_failed_patterns(self) -> list[str]:
        """Return descriptions of the last 10 experiments that were not kept."""
        reverted = [
            d.get("description", "")
            for d in self._nodes_of_type("Experiment")
            if not d.get("kept", True)
        ]
        return reverted[-10:]

    def get_feature_set_diff(self) -> dict[str, Any]:
        """Return a summary of tried feature families and column coverage."""
        all_features = self._nodes_of_type("Feature")
        active_features = [f for f in all_features if f.get("status") == "active"]
        cols_tried = self._tried_columns()
        all_cols = {d["name"] for d in self._nodes_of_type("Column")}
        untried_cols = sorted(all_cols - cols_tried)

        return {
            "n_features_tried": len(all_features),
            "n_features_active": len(active_features),
            "columns_tried": sorted(cols_tried),
            "columns_untried": untried_cols,
        }


def load_graph(path: Path = DEFAULT_GRAPH_PATH) -> MemoryGraph:
    """Load or create a MemoryGraph from the given path."""
    return MemoryGraph(path)
