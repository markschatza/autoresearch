# AutoResearch Adult Income

This repository follows the AutoResearch pattern with one fixed evaluation harness and one mutable training file.

## Setup

Before starting a fresh experiment run, work with the user to:

1. Agree on a run tag. Propose a short tag based on today's date, such as `apr9`, and make sure `autoresearch/<tag>` does not already exist.
2. Create the branch from the current mainline: `git checkout -b autoresearch/<tag>`.
3. Read the in-scope files for full context:
   - `README.md` for repository context.
   - `prepare.py` for fixed constants, data preparation, tokenizer handling, data loading, and evaluation logic. Do not modify it.
   - `train.py` for model architecture, optimizer settings, and the training loop. This is the experimental file.
4. Verify the prepared data exists under `artifacts/`. If the required cached dataset or split artifacts are missing, run `uv run prepare.py` before training.
5. Confirm the results directory is ready. `train.py` will write `results/latest.json` and timestamped JSON outputs after the first run.
6. Confirm setup looks correct, then begin the experiment loop.

## Scope

Read these files first:

1. `README.md`
2. `prepare.py`
3. `train.py`

`prepare.py` is fixed and read-only.
`train.py` is the only file the agent may edit.

## Hard Rules

The agent may:

- Edit only `train.py`.
- Run `uv run prepare.py` and `uv run train.py`.
- Read prior result files under `results/`.
- Update `program.md` only when the user explicitly asks to revise the research instructions.

The agent may not:

- Edit `prepare.py`.
- Edit `README.md`, `pyproject.toml`, or any other file unless the user explicitly asks.
- Change the data source, data cleaning rules, target definition, metric, or split.
- Install new packages or add dependencies beyond what is already declared in `pyproject.toml`.
- Introduce hidden state outside explicit files under `artifacts/` and `results/`.

## Fixed Evaluation

The benchmark is the Adult Income tabular classification task.

- Primary metric: validation ROC-AUC.
- Secondary metrics: validation accuracy and validation log loss.
- The train/validation/test split is frozen in `prepare.py`.
- Any improvement decision must use the fixed validation ROC-AUC produced by the current `uv run train.py`.

`prepare.py` is the ground-truth harness. If a change disagrees with it, the change is wrong.

## Experiment Loop

Always follow this loop:

1. Inspect the most recent result in `results/latest.json` and the best prior ROC-AUC from `results/*.json`.
2. Propose one concrete hypothesis for improving validation ROC-AUC.
3. Edit only `train.py`.
4. Run the fixed harness first if needed: `uv run prepare.py`.
5. Run the experiment: `uv run train.py`.
6. Compare the new validation ROC-AUC against the prior best.
7. Keep the change only if the improvement is meaningful relative to the added complexity.
8. Revert regressions or no-op changes so `train.py` stays near the best known version.

One experiment means one main idea. Avoid bundling many unrelated changes together.

## Simplicity Bias

Prefer changes that are:

- Deterministic.
- Easy to explain.
- Local to preprocessing, model choice, regularization, or optimization settings inside `train.py`.

If two versions are close, keep the simpler one.

## Result Handling

- `train.py` writes structured JSON results under `results/`.
- Use those JSON files for comparisons instead of informal log reading.
- Treat validation ROC-AUC as the deciding metric.
- Use test metrics for context only after validation improves.

## Starting Point

The initial baseline is intentionally modest. Improve it incrementally without breaking reproducibility or mutating the fixed harness.
