from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import openml
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


REPO_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DATASET_DIR = ARTIFACTS_DIR / "dataset"
BASELINE_DIR = ARTIFACTS_DIR / "baseline"
REFERENCE_DIR = ARTIFACTS_DIR / "reference"
OPENML_CACHE_DIR = ARTIFACTS_DIR / "openml_cache"

DATASET_CACHE_PATH = DATASET_DIR / "adult_income.parquet"
DATASET_METADATA_PATH = DATASET_DIR / "dataset_metadata.json"
SPLIT_METADATA_PATH = DATASET_DIR / "split_metadata.json"
SPLIT_INDICES_PATH = DATASET_DIR / "split_indices.npz"
LOCAL_BASELINE_PATH = BASELINE_DIR / "dummy_classifier_metrics.json"
REFERENCE_METRICS_PATH = REFERENCE_DIR / "openml_reference_metrics.json"

DATASET_NAME = "adult"
DATASET_VERSION = 2
TARGET_COLUMN = "class"
POSITIVE_LABEL = ">50K"
RANDOM_STATE = 42
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

PRIMARY_METRIC = "roc_auc"
SECONDARY_METRICS = ("accuracy", "log_loss")
@dataclass(frozen=True)
class SplitBundle:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    numeric_features: list[str]
    categorical_features: list[str]
    metadata: dict[str, Any]


def ensure_directories() -> None:
    for path in (ARTIFACTS_DIR, DATASET_DIR, BASELINE_DIR, REFERENCE_DIR, OPENML_CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def configure_openml_cache() -> None:
    ensure_directories()
    openml.config.set_root_cache_directory(str(OPENML_CACHE_DIR))


def _normalize_missing_values(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in cleaned.columns:
        if cleaned[column].dtype == "object" or pd.api.types.is_string_dtype(cleaned[column]):
            cleaned[column] = cleaned[column].replace("?", pd.NA)
    return cleaned


def _positive_label_from_target(target: pd.Series) -> str:
    values = [str(value).strip() for value in pd.Series(target).dropna().unique()]
    if POSITIVE_LABEL in values:
        return POSITIVE_LABEL
    if ">50K." in values:
        return ">50K."
    raise ValueError(f"Could not locate a positive label in target values: {values}")


def encode_target(target: pd.Series) -> np.ndarray:
    positive_label = _positive_label_from_target(target)
    return (target.astype(str).str.strip() == positive_label).astype(np.int64).to_numpy()


def _build_dataset_metadata(frame: pd.DataFrame, encoded_target: np.ndarray, details: dict[str, Any]) -> dict[str, Any]:
    feature_frame = frame.drop(columns=[TARGET_COLUMN])
    numeric_features = feature_frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in feature_frame.columns if column not in numeric_features]
    positive_rate = float(encoded_target.mean())

    return {
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "target_column": TARGET_COLUMN,
        "positive_label": _positive_label_from_target(frame[TARGET_COLUMN]),
        "n_rows": int(frame.shape[0]),
        "n_features": int(feature_frame.shape[1]),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "positive_rate": positive_rate,
        "openml_dataset_id": int(details["id"]) if "id" in details else None,
    }


def cache_dataset(force_refresh: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    ensure_directories()
    if DATASET_CACHE_PATH.exists() and DATASET_METADATA_PATH.exists() and not force_refresh:
        frame = pd.read_parquet(DATASET_CACHE_PATH)
        metadata = json.loads(DATASET_METADATA_PATH.read_text())
        return frame, metadata

    bunch = fetch_openml(
        name=DATASET_NAME,
        version=DATASET_VERSION,
        as_frame=True,
        parser="auto",
    )
    frame = _normalize_missing_values(bunch.frame.copy())
    frame.to_parquet(DATASET_CACHE_PATH, index=False)

    details = dict(getattr(bunch, "details", {}) or {})
    metadata = _build_dataset_metadata(frame, encode_target(frame[TARGET_COLUMN]), details)
    DATASET_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    return frame, metadata


def create_or_load_fixed_split(
    frame: pd.DataFrame,
    force_refresh: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if SPLIT_INDICES_PATH.exists() and SPLIT_METADATA_PATH.exists() and not force_refresh:
        indices = np.load(SPLIT_INDICES_PATH)
        metadata = json.loads(SPLIT_METADATA_PATH.read_text())
        return indices["train"], indices["val"], indices["test"], metadata

    y = encode_target(frame[TARGET_COLUMN])
    all_indices = np.arange(len(frame))

    outer_split = StratifiedShuffleSplit(n_splits=1, test_size=TEST_RATIO, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(outer_split.split(all_indices, y))

    inner_y = y[train_val_idx]
    inner_split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO),
        random_state=RANDOM_STATE,
    )
    train_sub_idx, val_sub_idx = next(inner_split.split(train_val_idx, inner_y))
    train_idx = train_val_idx[train_sub_idx]
    val_idx = train_val_idx[val_sub_idx]

    np.savez_compressed(SPLIT_INDICES_PATH, train=train_idx, val=val_idx, test=test_idx)
    metadata = {
        "seed": RANDOM_STATE,
        "split_ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "sizes": {"train": int(train_idx.size), "val": int(val_idx.size), "test": int(test_idx.size)},
    }
    SPLIT_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    return train_idx, val_idx, test_idx, metadata


def load_split_bundle() -> SplitBundle:
    frame, dataset_metadata = cache_dataset()
    train_idx, val_idx, test_idx, split_metadata = create_or_load_fixed_split(frame)
    X = frame.drop(columns=[TARGET_COLUMN]).reset_index(drop=True)
    y = encode_target(frame[TARGET_COLUMN])

    metadata = {
        "dataset": dataset_metadata,
        "split": split_metadata,
        "artifacts_dir": str(ARTIFACTS_DIR),
        "primary_metric": PRIMARY_METRIC,
        "secondary_metrics": list(SECONDARY_METRICS),
    }

    numeric_features = dataset_metadata["numeric_features"]
    categorical_features = dataset_metadata["categorical_features"]

    return SplitBundle(
        X_train=X.iloc[train_idx].reset_index(drop=True),
        X_val=X.iloc[val_idx].reset_index(drop=True),
        X_test=X.iloc[test_idx].reset_index(drop=True),
        y_train=y[train_idx],
        y_val=y[val_idx],
        y_test=y[test_idx],
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        metadata=metadata,
    )


def compute_metrics(y_true: np.ndarray, positive_probs: np.ndarray) -> dict[str, float]:
    clipped = np.clip(np.asarray(positive_probs, dtype=float), 1e-7, 1 - 1e-7)
    predictions = (clipped >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, clipped)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "log_loss": float(log_loss(y_true, np.column_stack([1.0 - clipped, clipped]), labels=[0, 1])),
    }


def run_local_dummy_baseline(force_refresh: bool = False) -> dict[str, Any]:
    if LOCAL_BASELINE_PATH.exists() and not force_refresh:
        return json.loads(LOCAL_BASELINE_PATH.read_text())

    bundle = load_split_bundle()
    model = DummyClassifier(strategy="prior")
    model.fit(bundle.X_train, bundle.y_train)
    val_probs = model.predict_proba(bundle.X_val)[:, 1]
    test_probs = model.predict_proba(bundle.X_test)[:, 1]

    payload = {
        "model_name": "DummyClassifier(strategy='prior')",
        "validation_metrics": compute_metrics(bundle.y_val, val_probs),
        "test_metrics": compute_metrics(bundle.y_test, test_probs),
    }
    LOCAL_BASELINE_PATH.write_text(json.dumps(payload, indent=2))
    return payload


def _series_from_evaluations(eval_frame: pd.DataFrame) -> pd.Series:
    if "value" in eval_frame.columns:
        return eval_frame["value"]
    excluded = {"task_id", "run_id", "setup_id", "fold", "repeat", "flow_id", "function"}
    numeric_columns = [column for column in eval_frame.columns if column not in excluded and pd.api.types.is_numeric_dtype(eval_frame[column])]
    if not numeric_columns:
        raise ValueError(f"Could not identify an evaluation score column in: {eval_frame.columns.tolist()}")
    return eval_frame[numeric_columns[0]]


def _select_reference_task(dataset_id: int) -> dict[str, Any] | None:
    tasks = openml.tasks.list_tasks(
        task_type=openml.tasks.TaskType.SUPERVISED_CLASSIFICATION,
        data_id=dataset_id,
        output_format="dataframe",
    )
    if tasks is None or tasks.empty:
        return None

    candidates = tasks.copy()
    candidates["estimation_procedure"] = candidates["estimation_procedure"].fillna("")
    candidates["priority"] = candidates["estimation_procedure"].str.lower().map(
        lambda value: 0 if "crossvalidation" in value else 1 if "holdout" in value else 2
    )
    row = candidates.sort_values(["priority", "tid"]).iloc[0]
    return {
        "task_id": int(row["tid"]),
        "task_name": str(row.get("name", "")),
        "estimation_procedure": str(row.get("estimation_procedure", "")),
    }


def fetch_openml_reference_metrics(force_refresh: bool = False) -> dict[str, Any]:
    configure_openml_cache()
    if REFERENCE_METRICS_PATH.exists() and not force_refresh:
        return json.loads(REFERENCE_METRICS_PATH.read_text())

    _, dataset_metadata = cache_dataset()
    dataset_id = dataset_metadata.get("openml_dataset_id")
    if dataset_id is None:
        payload = {"status": "unavailable", "reason": "missing_openml_dataset_id"}
        REFERENCE_METRICS_PATH.write_text(json.dumps(payload, indent=2))
        return payload

    task_info = _select_reference_task(dataset_id)
    if task_info is None:
        payload = {"status": "unavailable", "reason": "no_classification_task_found", "dataset_id": dataset_id}
        REFERENCE_METRICS_PATH.write_text(json.dumps(payload, indent=2))
        return payload

    metric_specs = {
        "roc_auc": ("area_under_roc_curve", "desc"),
        "accuracy": ("predictive_accuracy", "desc"),
        "log_loss": ("log_loss", "asc"),
    }
    metric_payload: dict[str, Any] = {}

    try:
        for metric_name, (function_name, sort_order) in metric_specs.items():
            eval_frame = openml.evaluations.list_evaluations(
                function=function_name,
                tasks=[task_info["task_id"]],
                size=5,
                sort_order=sort_order,
                output_format="dataframe",
            )
            if eval_frame is None or eval_frame.empty:
                metric_payload[metric_name] = {"status": "missing"}
                continue

            values = _series_from_evaluations(eval_frame)
            top_row = eval_frame.iloc[0]
            metric_payload[metric_name] = {
                "status": "ok",
                "function": function_name,
                "value": float(values.iloc[0]),
                "run_id": int(top_row["run_id"]) if "run_id" in top_row else None,
                "n_entries_considered": int(len(eval_frame)),
            }

        payload = {
            "status": "ok",
            "dataset_id": dataset_id,
            "task": task_info,
            "metrics": metric_payload,
        }
    except Exception as exc:  # pragma: no cover - network and service behavior are external
        payload = {
            "status": "unavailable",
            "dataset_id": dataset_id,
            "task": task_info,
            "reason": f"{type(exc).__name__}: {exc}",
        }

    REFERENCE_METRICS_PATH.write_text(json.dumps(payload, indent=2))
    return payload


def format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _reference_metric_value(reference_payload: dict[str, Any], metric_name: str) -> float | None:
    metric_block = reference_payload.get("metrics", {}).get(metric_name, {})
    if metric_block.get("status") != "ok":
        return None
    return float(metric_block["value"])


def print_comparison_table(local_baseline: dict[str, Any], reference_payload: dict[str, Any]) -> None:
    local_metrics = local_baseline["validation_metrics"]
    rows = [
        ("local_dummy_val", local_metrics["roc_auc"], local_metrics["accuracy"], local_metrics["log_loss"]),
        (
            "openml_reference",
            _reference_metric_value(reference_payload, "roc_auc"),
            _reference_metric_value(reference_payload, "accuracy"),
            _reference_metric_value(reference_payload, "log_loss"),
        ),
    ]

    header = ("source", "roc_auc", "accuracy", "log_loss")
    widths = [
        max(len(header[idx]), max(len(format_metric(row[idx])) if idx else len(row[idx]) for row in rows))
        for idx in range(len(header))
    ]
    print("Comparison")
    print(" | ".join(header[idx].ljust(widths[idx]) for idx in range(len(header))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(
            " | ".join(
                (row[idx] if idx == 0 else format_metric(row[idx])).ljust(widths[idx])
                for idx in range(len(row))
            )
        )


def prepare_all(force_dataset_refresh: bool = False, force_reference_refresh: bool = False) -> dict[str, Any]:
    configure_openml_cache()
    bundle = load_split_bundle()
    local_baseline = run_local_dummy_baseline(force_refresh=force_dataset_refresh)
    reference_payload = fetch_openml_reference_metrics(force_refresh=force_reference_refresh)
    return {
        "bundle_metadata": bundle.metadata,
        "local_baseline": local_baseline,
        "reference_payload": reference_payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the fixed Adult Income evaluation harness.")
    parser.add_argument("--refresh-dataset", action="store_true", help="Re-download the dataset and rebuild cached splits.")
    parser.add_argument("--refresh-reference", action="store_true", help="Re-fetch OpenML reference evaluations.")
    args = parser.parse_args()

    if args.refresh_dataset:
        for path in (DATASET_CACHE_PATH, DATASET_METADATA_PATH, SPLIT_METADATA_PATH, SPLIT_INDICES_PATH, LOCAL_BASELINE_PATH):
            if path.exists():
                path.unlink()

    summary = prepare_all(
        force_dataset_refresh=args.refresh_dataset,
        force_reference_refresh=args.refresh_reference,
    )

    dataset_meta = summary["bundle_metadata"]["dataset"]
    split_meta = summary["bundle_metadata"]["split"]

    print(f"Prepared dataset: {dataset_meta['dataset_name']} (OpenML id={dataset_meta['openml_dataset_id']})")
    print(
        f"Rows={dataset_meta['n_rows']}, features={dataset_meta['n_features']}, "
        f"train/val/test={split_meta['sizes']['train']}/{split_meta['sizes']['val']}/{split_meta['sizes']['test']}"
    )
    print(f"Primary metric: {PRIMARY_METRIC}")
    print(f"Secondary metrics: {', '.join(SECONDARY_METRICS)}")
    print_comparison_table(summary["local_baseline"], summary["reference_payload"])

    reference_status = summary["reference_payload"]["status"]
    print(f"Reference metrics cache: {REFERENCE_METRICS_PATH}")
    print(f"Reference lookup status: {reference_status}")


if __name__ == "__main__":
    main()
