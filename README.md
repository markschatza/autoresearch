# AutoResearch Adult Income

This repo is a minimal AutoResearch-style setup for the Adult Income tabular benchmark. The fixed harness lives in `prepare.py`, the only mutable experiment file is `train.py`, and `program.md` defines the research loop.

## Files

- `prepare.py`: fixed evaluation harness, dataset caching, split creation, metric helpers, and OpenML reference lookup.
- `train.py`: baseline model pipeline and structured result writing. This is the only file future agent loops should modify.
- `program.md`: rules for the constrained experiment loop.
- `artifacts/`: cached dataset, split metadata, fixed local baseline, and OpenML reference metrics.
- `results/`: structured experiment outputs from `train.py`.

## Commands

```bash
uv sync
uv run prepare.py
uv run train.py
```

## Benchmark Contract

- Dataset: Adult Income from OpenML.
- Target: income above `$50K`.
- Split: fixed train/validation/test split frozen in `prepare.py`.
- Primary metric: validation ROC-AUC.
- Secondary metrics: validation accuracy and validation log loss.

## Workflow

1. Run `uv run prepare.py` once to cache the dataset, freeze the split, compute a trivial local baseline, and fetch OpenML reference scores when available.
2. Run `uv run train.py` to train the current baseline and write `results/latest.json` plus a timestamped result file.
3. Future AutoResearch loops should inspect prior JSON results, modify only `train.py`, rerun the fixed evaluation, and keep only meaningful ROC-AUC improvements.

The code is intentionally simple, local-first, and reproducible. There are no notebooks, services, or hidden caches outside the explicit `artifacts/` and `results/` directories.
