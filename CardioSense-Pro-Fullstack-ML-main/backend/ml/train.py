from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.constants import FEATURE_ORDER, RISK_THRESHOLDS
from ml.dataset_utils import load_training_frame


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train heart disease prediction model")
    parser.add_argument(
        "--raw-dir",
        default=str(root / "data" / "raw"),
        help="Directory containing raw CSV files",
    )
    parser.add_argument(
        "--processed-dir",
        default=str(root / "data" / "processed"),
        help="Directory to store processed training data",
    )
    parser.add_argument(
        "--artifact-path",
        default=str(root / "model_artifacts" / "heart_model.joblib"),
        help="Output model artifact path",
    )
    parser.add_argument(
        "--metrics-path",
        default=str(root / "model_artifacts" / "metrics.json"),
        help="Output metrics JSON path",
    )
    parser.add_argument(
        "--test-size",
        default=0.2,
        type=float,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--random-state",
        default=42,
        type=int,
        help="Random seed",
    )
    return parser.parse_args()


def evaluate_model(name: str, model, x_train, y_train, x_test, y_test) -> dict:
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_test)[:, 1]
    else:
        decision = model.decision_function(x_test)
        proba = 1.0 / (1.0 + np.exp(-decision))

    return {
        "name": name,
        "model": model,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }


def main() -> None:
    args = parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    processed_dir = Path(args.processed_dir).resolve()
    artifact_path = Path(args.artifact_path).resolve()
    metrics_path = Path(args.metrics_path).resolve()

    processed_dir.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    frame, source_files = load_training_frame(raw_dir)
    frame.to_csv(processed_dir / "canonical_training_data.csv", index=False)

    x = frame[FEATURE_ORDER]
    y = frame["target"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    models = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        random_state=args.random_state,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=args.random_state,
        ),
        "svm_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=2.0,
                        gamma="scale",
                        probability=True,
                        class_weight="balanced",
                        random_state=args.random_state,
                    ),
                ),
            ]
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=args.random_state),
    }

    evaluations = []
    for model_name, model in models.items():
        evaluations.append(evaluate_model(model_name, model, x_train, y_train, x_test, y_test))

    evaluations.sort(key=lambda item: (item["roc_auc"], item["f1"]), reverse=True)
    best = evaluations[0]

    trained_at = datetime.now(timezone.utc).isoformat()

    artifact = {
        "model": best["model"],
        "feature_order": FEATURE_ORDER,
        "model_name": best["name"],
        "thresholds": RISK_THRESHOLDS,
        "trained_at": trained_at,
        "dataset": {
            "rows": int(len(frame)),
            "positive_rate": float(frame["target"].mean()),
            "source_files": [str(path) for path in source_files],
        },
    }

    joblib.dump(artifact, artifact_path)

    metrics_payload = {
        "best_model": best["name"],
        "trained_at": trained_at,
        "dataset_rows": int(len(frame)),
        "test_size": args.test_size,
        "evaluations": [
            {
                "name": item["name"],
                "accuracy": item["accuracy"],
                "precision": item["precision"],
                "recall": item["recall"],
                "f1": item["f1"],
                "roc_auc": item["roc_auc"],
            }
            for item in evaluations
        ],
    }

    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print("Training complete")
    print(f"Best model: {best['name']}")
    print(f"Artifact: {artifact_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
