Read this at the start of every session. Then loop until the stop condition is hit or manualy stopped.

**NEVER STOP.** Do not ask the human if you should continue — they may be asleep. Loop until `STATUS: STOP` is hit, write the final markdown report describing the features chosen, and finish. If stuck, re-read `program.md`, combine near-misses, or try a structurally different transform family entirely.

## Setup

Run these commands **in order, waiting for each to complete before starting the next**. Do NOT skip any step. Do NOT start the loop until all three have finished.

```bash
uv run autoresearch init
uv run autoresearch prepare
```

**Data Discovery (MANDATORY — must complete before the loop begins):**
```bash
uv run autoresearch discover     # WARNING: can take 10–20 min on large datasets. Do NOT let it get killed by a default timeout.
```
This scans for hidden entity structures (invariant expressions, entity keys). The loop depends on discovery results — starting without them means missing the highest-leverage features. **Wait for this command to finish. Do not run it in the background. Do not proceed to the loop until it exits.**

After discovery completes, review what was found:
```bash
uv run autoresearch inspect --discovery
```

**Structural Pre-computation (Optional but Recommended for Large Datasets):**
Before engineering features on datasets with many columns (>50), write a one-off Python script to map column families (e.g., matching prefixes or identical null-patterns).
Record these relationships in the memory graph using the formal registry (like in `train.py`):
```python
from autoresearch_tabular.memory_graph import load_graph
mg = load_graph()
mg.register_relationship_type(
    rel_type="GROUPED_WITH", description="Columns sharing a common prefix",
    source_type="Column", target_type=["Column"], category="structural"
)
mg.add_edge_typed("col_V1", "col_V2", "GROUPED_WITH")
```
This structural layer helps you explore systematically instead of randomly sampling.

`db/memory_graph.json` accumulates permanently across sessions — every experiment appended, never reset.

---

## What you edit

**Only** `src/autoresearch_tabular/features.py` and one-off exploration scripts to map column families. Everything else is read-only.

```python
def engineer_features(
    X_train, X_val, X_test, y_train=None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
```

- Fit ALL transforms on `X_train`/`y_train` only. `y_val` and `y_test` do not exist.
- Target-aware features on **training rows**: OOF encoding (KFold inner split — each row's stat computed from the *other* inner folds, never its own target).
- Target-aware features on **val/test rows**: map using full training statistics.
- Return `(X_train_eng, X_val_eng, X_test_eng)`. Only `numpy`, `pandas`, `sklearn`.
- Description: `col=<...>; op=<...>; fit=train_only; reason=<...>`
- Never repeat a description listed in FAILED PATTERNS (`--failed`).
- Scoring: `composite = cv_score − 0.001 × n_features` (higher-is-better) or `cv_score + 0.001 × n_features` (lower-is-better). A feature must improve cv_score by ~0.1% to justify its existence.

**Run a baseline pass (MANDATORY):**
Before entering the loop, run:
```bash
uv run python -m autoresearch_tabular.train --description "baseline: passthrough"
```
Ensure `src/autoresearch_tabular/features.py` is unmodified. This establishes the baseline `cv_score` so `+0.000` delta starts from the raw accuracy. Keep this experiment (`git add src/autoresearch_tabular/features.py && git commit -m "baseline: passthrough"`).

---

## LOOP FOREVER

### 0 — Status check (every iteration, first)

```bash
uv run autoresearch status
```

- `STOP:` → Write a final markdown report (e.g., `final_features_report.md`) detailing the best features discovered, the successful transformations, and the general engineering strategy. Then print `results.tsv` and stop entirely.
- `EXPLORE:` → skip to step 4.
- `CONTINUE:` → step 1.

---

### 1 — Read context

Read `program.md` and `src/autoresearch_tabular/features.py`. Then:

```bash
uv run autoresearch inspect --failed        # gate: exact reverted descriptions — never propose these again
uv run autoresearch inspect --shap          # primary signal: zero-SHAP = dead weight; high shap_std = overfitting
uv run autoresearch inspect --load-bearing  # hard constraint: in every kept experiment — do not remove without --shap evidence
uv run autoresearch inspect --ablation      # features in kept vs reverted — confirms winners and losers
uv run autoresearch inspect --rates         # transform families by keep rate — prioritise the top ones
uv run autoresearch inspect --hypotheses    # calibration: SUPPORTS = prediction correct, CONTRADICTS = prediction wrong — adjust priors
uv run autoresearch inspect --discovery    # entity keys and invariant expressions found by discover
```

New session or 10+ experiments since last read: also run `--context` (lossless compressed history — re-orients without re-reading individual experiments).

For deeper dives: `--exp <id>`, `--col <name>`, `--central` (feature centrality ranking), `--longest-path` (longest improvement chain), `--correlations` (feature pairs with |r| > 0.8 — drop candidates), `--edges` (registered relationship types), or `inspect` (no flags) for the full report.

**Statistical queries** (max 20/session, leakage-protected):
```bash
uv run autoresearch query within_group_variance --expr "TransactionDT/86400 - D1" --groupby card1 addr1
uv run autoresearch query cardinality --cols card1 addr1
uv run autoresearch query correlation --col_a TransactionAmt --col_b D1
uv run autoresearch query conditional_distribution --col TransactionAmt --groupby ProductCD --n_groups 5
```

---

### 2 — Hypothesise and edit

**Discovery-guided hypotheses:** If `--discovery` shows INVARIANT_WITHIN relationships, these indicate hidden entity structures. An expression that is near-constant within an entity key group can be used to create powerful features: use the entity key as a groupby for per-entity aggregation features (mean, std, count, nunique of other columns). This is the highest-leverage transform family when present. Use `uv run autoresearch query within_group_variance --expr "..." --groupby col1 col2` to verify.

Before touching `features.py`, write a testable prediction to the graph:

```bash
uv run python -c "
from autoresearch_tabular.memory_graph import load_graph
mg = load_graph()
hyp_id = mg.add_hypothesis(
    '<one sentence: why this change should help>',
    predicted_direction='+',   # '+' = expect improvement, '-' = expect regression
    predicted_delta=0.002,     # rough expected composite score change (optional)
)
# Forward planning: Link this hypothesis to the primary column(s) it targets
mg.add_edge_typed(f'hyp_{hyp_id}', 'col_<target_name>', 'TARGETS')
print('HYP_ID:', hyp_id)
"
```

Note the printed `HYP_ID` — you will need it in step 3. Then make one targeted change to `src/autoresearch_tabular/features.py`. **Do not commit yet** — HEAD must stay on the last kept experiment so the revert in step 3 works correctly.

---

### 3 — Train and ratchet

```bash
uv run python -m autoresearch_tabular.train --description "<description>" > run.log 2>&1
grep -E "^METRIC:|^Result:|^Delta:" run.log
```

Resolve the hypothesis from step 2 using the `HYP_ID` and the `experiment_id`, `kept`, and `delta` values printed above:

```bash
uv run python -c "
from autoresearch_tabular.memory_graph import load_graph
mg = load_graph()
mg.resolve_hypothesis(<hyp_id>, experiment_id=<exp_id>, kept=<True/False>, actual_delta=<delta>)
"
```

**`kept=True`** — commit as the new rolling best (amend keeps a single "best" commit):
```bash
git add src/autoresearch_tabular/features.py
git commit --amend -m "best: <description>"
```
> First experiment ever in a fresh repo (nothing to amend): use `git commit -m "best: <description>"` instead.

**`kept=False`** — restore from HEAD (always the last kept experiment):
```bash
git checkout HEAD -- src/autoresearch_tabular/features.py
```

Crashed: fix trivial bugs and retry once. Fundamentally broken: revert and move on.

**Optional** — if this result overturns a prior belief from `--hypotheses`, supersede the stale entry:
```bash
uv run python -c "
from autoresearch_tabular.memory_graph import load_graph
mg = load_graph()
mg.supersede_hypothesis(<old_hyp_id>, <new_hyp_id>)
"
```

Go to step 0.

---

### 4 — Forced exploration

Triggered by stagnation, 10+ consecutive failures, or every 15 experiments.

```bash
uv run autoresearch inspect --coverage     # columns with zero features ever derived — start here
uv run autoresearch inspect --untried      # (column, transform_family) pairs never attempted
uv run autoresearch inspect --saturated    # columns whose signal is exhausted — avoid these
uv run autoresearch inspect --rates        # best-performing transform families
uv run autoresearch inspect --ablation     # always-kept features to preserve
```

Pick a column from `--coverage` (untouched) or `--untried` (untried transform family). Alternatively, use an untried structural family mapped during pre-computation. Use a family with a strong `--rates` score. Preserve anything `--ablation` marks "always kept". Do not repeat anything in `--failed`. Go to step 2.

---

