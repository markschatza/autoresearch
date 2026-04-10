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


## Simplicity Bias

Prefer changes that are:

- Deterministic.
- Easy to explain.
- Local to preprocessing, model choice, regularization, or optimization settings inside `train.py`.

If two versions are close, keep the simpler one.

Simplicity criterion: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 ROC_AUC improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 ROC_AUC improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

## Output Format

Once `uv run train.py` finishes, it prints a summary in this shape:

---
Validation metrics
  roc_auc:  0.928374
  accuracy: 0.872145
  log_loss: 0.281903
Test metrics
  roc_auc:  0.925110
  accuracy: 0.869441
  log_loss: 0.286552
Prior best: 20260410T031522Z (delta_val_roc_auc=+0.001284)
Result JSON: results/20260410T032041Z.json

The exact numbers will vary by experiment. The key metric is validation ROC-AUC.

Prefer reading the structured result instead of scraping terminal logs. You can extract the key metric from the latest JSON result with:

```powershell
Get-Content results/latest.json | python -c "import json,sys; print(json.load(sys.stdin)['metrics']['validation']['roc_auc'])"
```

If you are reading terminal output directly, the relevant line is the validation `roc_auc` under `Validation metrics`.

## Result Handling

- `train.py` writes structured JSON results under `results/`.
- Use those JSON files for comparisons and as the source of truth for metrics.
- Treat validation ROC-AUC as the deciding metric.
- Use test metrics for context only after validation improves.
- Canonical files are `results/latest.json` and `results/<run_id>.json`.

Each result includes:

- `run_id`
- `primary_metric`
- `model`
- `runtime.elapsed_seconds`
- `metrics.validation.roc_auc`
- `metrics.validation.accuracy`
- `metrics.validation.log_loss`
- `metrics.test.*`
- `comparison_to_prior_best.delta_val_roc_auc` when available

## Logging Results

When an experiment is done, also log it to `results.tsv` as a tab-separated file. Use tabs, not commas.

The TSV has a header row and 5 columns:

- `commit`
- `val_roc_auc`
- `runtime_seconds`
- `status`
- `description`

Definitions:

- `commit`: git commit hash, short 7 chars, for example from `git rev-parse --short HEAD`
- `val_roc_auc`: validation ROC-AUC achieved, for example `0.928374`; use `0.000000` for crashes
- `runtime_seconds`: total runtime in seconds from `runtime.elapsed_seconds`, formatted reasonably, for example `12.4`; use `0.0` for crashes
- `status`: `keep`, `discard`, or `crash`
- `description`: short text description of what the experiment tried

Example:

```text
commit	val_roc_auc	runtime_seconds	status	description
a1b2c3d	0.928374	12.4	keep	baseline hist gradient boosting
b2c3d4e	0.930112	12.8	keep	increase max_iter to 500
c3d4e5f	0.927901	12.1	discard	raise learning rate to 0.1
d4e5f6g	0.000000	0.0	crash	invalid categorical encoding idea
```

Prefer extracting the metric from `results/latest.json` rather than from terminal output. A typical workflow is:

```powershell
$commit = git rev-parse --short HEAD
$result = Get-Content results/latest.json | python -c "import json,sys; print(json.dumps(json.load(sys.stdin)))"
```

The TSV is a lightweight experiment ledger. The JSON result files remain the canonical detailed record.

## Experiment Loop

The experiment runs on a dedicated branch (e.g. autoresearch/mar5 or autoresearch/mar5-gpu0).

LOOP FOREVER:

1. Inspect the most recent result in `results/latest.json` and the best prior ROC-AUC from `results/*.json`.
2. Propose one concrete hypothesis for improving validation ROC-AUC.
3. Edit only `train.py`.
4. Run the fixed harness first if needed: `uv run prepare.py`.
5. Run the experiment: `uv run train.py`.
6. Compare the new validation ROC-AUC against the prior best.
7. Keep the change only if the improvement is meaningful relative to the added complexity.
8. Revert regressions or no-op changes so `train.py` stays near the best known version.

One experiment means one main idea. Avoid bundling many unrelated changes together.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

Crashes: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, log it as `crash` in `results.tsv` with `val_roc_auc=0.000000` and `runtime_seconds=0.0`, then move on.

NEVER STOP: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
