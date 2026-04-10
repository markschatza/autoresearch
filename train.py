from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prepare import PRIMARY_METRIC, compute_metrics, load_split_bundle


RESULTS_DIR = Path(__file__).resolve().parent / "results"
LATEST_RESULT_PATH = RESULTS_DIR / "latest.json"

SEED = 42

# Future AutoResearch loops may edit the model and preprocessing below.
MODEL_CONFIG = {
    "model_name": "logistic_regression",
    "solver": "liblinear",
    "C": 1.0,
    "max_iter": 400,
}


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    classifier = LogisticRegression(
        solver=MODEL_CONFIG["solver"],
        C=MODEL_CONFIG["C"],
        max_iter=MODEL_CONFIG["max_iter"],
        random_state=SEED,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def load_prior_best_result() -> dict[str, Any] | None:
    ensure_results_dir()
    candidates: list[dict[str, Any]] = []
    for path in RESULTS_DIR.glob("*.json"):
        if path.name == "latest.json":
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if "metrics" in payload and "validation" in payload["metrics"]:
            candidates.append(payload)

    if not candidates:
        return None
    return max(candidates, key=lambda payload: payload["metrics"]["validation"][PRIMARY_METRIC])


def write_result(payload: dict[str, Any]) -> Path:
    ensure_results_dir()
    run_id = payload["run_id"]
    result_path = RESULTS_DIR / f"{run_id}.json"
    result_path.write_text(json.dumps(payload, indent=2))
    LATEST_RESULT_PATH.write_text(json.dumps(payload, indent=2))
    return result_path


def main() -> None:
    np.random.seed(SEED)
    bundle = load_split_bundle()
    pipeline = build_pipeline(bundle.numeric_features, bundle.categorical_features)
    pipeline.fit(bundle.X_train, bundle.y_train)

    val_probs = pipeline.predict_proba(bundle.X_val)[:, 1]
    test_probs = pipeline.predict_proba(bundle.X_test)[:, 1]
    validation_metrics = compute_metrics(bundle.y_val, val_probs)
    test_metrics = compute_metrics(bundle.y_test, test_probs)

    prior_best = load_prior_best_result()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "run_id": timestamp,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_metric": PRIMARY_METRIC,
        "model": deepcopy(MODEL_CONFIG),
        "data": {
            "dataset_name": bundle.metadata["dataset"]["dataset_name"],
            "split_seed": bundle.metadata["split"]["seed"],
            "split_sizes": bundle.metadata["split"]["sizes"],
        },
        "metrics": {
            "validation": validation_metrics,
            "test": test_metrics,
        },
        "comparison_to_prior_best": None,
    }

    if prior_best is not None:
        previous_score = prior_best["metrics"]["validation"][PRIMARY_METRIC]
        payload["comparison_to_prior_best"] = {
            "prior_best_run_id": prior_best["run_id"],
            "prior_best_val_roc_auc": previous_score,
            "delta_val_roc_auc": validation_metrics[PRIMARY_METRIC] - previous_score,
        }

    result_path = write_result(payload)

    print("Validation metrics")
    print(f"  roc_auc:  {validation_metrics['roc_auc']:.6f}")
    print(f"  accuracy: {validation_metrics['accuracy']:.6f}")
    print(f"  log_loss: {validation_metrics['log_loss']:.6f}")
    print("Test metrics")
    print(f"  roc_auc:  {test_metrics['roc_auc']:.6f}")
    print(f"  accuracy: {test_metrics['accuracy']:.6f}")
    print(f"  log_loss: {test_metrics['log_loss']:.6f}")
    if payload["comparison_to_prior_best"] is None:
        print("Prior best: none")
    else:
        comparison = payload["comparison_to_prior_best"]
        print(
            "Prior best: "
            f"{comparison['prior_best_run_id']} "
            f"(delta_val_roc_auc={comparison['delta_val_roc_auc']:+.6f})"
        )
    print(f"Result JSON: {result_path}")


if __name__ == "__main__":
    main()
