from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.services.feature_schema import DEFAULT_THRESHOLDS, FEATURE_ORDER

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self) -> None:
        settings = get_settings()
        self.artifact_path = Path(settings.model_artifact_path)
        self.model: Any | None = None
        self.feature_order: list[str] = FEATURE_ORDER.copy()
        self.thresholds: dict[str, float] = DEFAULT_THRESHOLDS.copy()
        self.metadata: dict[str, Any] = {}
        self.reload()

    @property
    def is_model_loaded(self) -> bool:
        return self.model is not None

    @property
    def model_version(self) -> str:
        if not self.is_model_loaded:
            return "heuristic-fallback"
        model_name = str(self.metadata.get("model_name", "trained-model"))
        trained_at = str(self.metadata.get("trained_at", "unknown"))
        return f"{model_name}@{trained_at}"

    def reload(self) -> None:
        if not self.artifact_path.exists():
            logger.warning(
                "Model artifact not found at %s. Using heuristic fallback.",
                self.artifact_path,
            )
            self.model = None
            self.metadata = {}
            self.feature_order = FEATURE_ORDER.copy()
            self.thresholds = DEFAULT_THRESHOLDS.copy()
            return

        try:
            artifact = joblib.load(self.artifact_path)
            self.model = artifact.get("model")
            self.metadata = artifact
            self.feature_order = artifact.get("feature_order", FEATURE_ORDER)
            stored_thresholds = artifact.get("thresholds", {})
            self.thresholds = {
                "low": float(stored_thresholds.get("low", DEFAULT_THRESHOLDS["low"])),
                "high": float(stored_thresholds.get("high", DEFAULT_THRESHOLDS["high"])),
            }
            logger.info("Model loaded successfully from %s", self.artifact_path)
        except Exception:
            logger.exception("Failed to load model artifact. Falling back to heuristic mode.")
            self.model = None
            self.metadata = {}
            self.feature_order = FEATURE_ORDER.copy()
            self.thresholds = DEFAULT_THRESHOLDS.copy()

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        features = {
            key: float(payload.get(key, 0.0))
            for key in FEATURE_ORDER
        }

        if self.is_model_loaded:
            probability, prediction = self._predict_with_model(features)
        else:
            probability, prediction = self._predict_with_heuristic(features)

        risk_level, risk_label = self._resolve_risk_band(probability)

        return {
            "prediction": prediction,
            "probability": probability,
            "risk_level": risk_level,
            "risk_label": risk_label,
            "risk_factors": self._derive_risk_factors(features, risk_level),
            "recommendation": self._recommendation_for(risk_level),
            "model_version": self.model_version,
        }

    def _predict_with_model(self, features: dict[str, float]) -> tuple[float, int]:
        if self.model is None:
            return self._predict_with_heuristic(features)

        row = {
            name: float(features.get(name, 0.0))
            for name in self.feature_order
        }
        frame = pd.DataFrame([row], columns=self.feature_order)

        if hasattr(self.model, "predict_proba"):
            probability = float(self.model.predict_proba(frame)[0][1])
        elif hasattr(self.model, "decision_function"):
            score = float(self.model.decision_function(frame)[0])
            probability = 1.0 / (1.0 + np.exp(-score))
        else:
            prediction = int(self.model.predict(frame)[0])
            probability = 0.75 if prediction == 1 else 0.25

        probability = float(np.clip(probability, 0.0, 1.0))
        prediction = int(probability >= 0.5)
        return probability, prediction

    def _predict_with_heuristic(self, features: dict[str, float]) -> tuple[float, int]:
        means = {
            "age": 54.4,
            "trestbps": 131.7,
            "chol": 246.7,
            "thalach": 149.6,
            "oldpeak": 1.04,
        }
        stds = {
            "age": 9.04,
            "trestbps": 17.6,
            "chol": 51.8,
            "thalach": 22.9,
            "oldpeak": 1.16,
        }

        def norm(value: float, key: str) -> float:
            return (value - means[key]) / stds[key]

        cp_score = [1.2, 0.4, 0.0, 1.0]
        slope_score = [-0.40, 0.10, 0.50]
        thal_score = [-0.5, 0.3, 0.9, 0.4]

        cp = int(features["cp"])
        slope = int(features["slope"])
        thal = int(features["thal"])

        score = 0.2
        score += norm(features["age"], "age") * 0.38
        score += features["sex"] * 0.74
        score += cp_score[cp] if 0 <= cp < len(cp_score) else 0.0
        score += norm(features["trestbps"], "trestbps") * 0.21
        score += norm(features["chol"], "chol") * 0.12
        score += features["fbs"] * 0.18
        score += features["restecg"] * 0.22
        score += norm(features["thalach"], "thalach") * (-0.48)
        score += features["exang"] * 0.90
        score += norm(features["oldpeak"], "oldpeak") * 0.62
        score += slope_score[slope] if 0 <= slope < len(slope_score) else 0.0
        score += features["ca"] * 0.72
        score += thal_score[thal] if 0 <= thal < len(thal_score) else 0.4

        probability = 1.0 / (1.0 + np.exp(-score))
        probability = float(np.clip(probability, 0.0, 1.0))
        prediction = int(probability >= 0.5)
        return probability, prediction

    def _resolve_risk_band(self, probability: float) -> tuple[str, str]:
        low_threshold = self.thresholds["low"]
        high_threshold = self.thresholds["high"]

        if probability >= high_threshold:
            return "high", "High Risk"
        if probability >= low_threshold:
            return "medium", "Moderate Risk"
        return "low", "Low Risk"

    def _recommendation_for(self, risk_level: str) -> str:
        if risk_level == "high":
            return (
                "High risk detected. Immediate consultation with a cardiologist is strongly advised. "
                "Consider stress testing, echocardiography, and advanced imaging."
            )
        if risk_level == "medium":
            return (
                "Moderate risk profile. Schedule a cardiac evaluation soon and adopt prevention-focused lifestyle changes."
            )
        return (
            "Low risk profile. Maintain healthy habits with regular exercise, balanced nutrition, and annual checkups."
        )

    def _derive_risk_factors(self, features: dict[str, float], risk_level: str) -> list[str]:
        factors: list[str] = []
        protective: list[str] = []

        if features["cp"] == 0:
            factors.append("Typical angina pattern present")
        if features["cp"] == 3:
            factors.append("Asymptomatic pattern can indicate silent ischemia")
        if features["ca"] >= 2:
            factors.append(f"{int(features['ca'])} major vessels narrowed")
        if features["exang"] == 1:
            factors.append("Exercise-induced angina present")
        if features["oldpeak"] >= 2:
            factors.append(f"High ST depression ({features['oldpeak']:.1f} mm)")
        if features["thal"] == 2:
            factors.append("Reversible thalassemia defect detected")
        if features["thalach"] < 120:
            factors.append(f"Low max heart rate ({int(features['thalach'])} bpm)")
        if features["trestbps"] > 140:
            factors.append(f"Hypertensive resting BP ({int(features['trestbps'])} mmHg)")
        if features["chol"] > 240:
            factors.append(f"High cholesterol ({int(features['chol'])} mg/dL)")

        if features["ca"] == 0:
            protective.append("No major vessel blockage observed")
        if features["oldpeak"] == 0:
            protective.append("No ST depression during stress")
        if features["thalach"] > 150:
            protective.append("Strong max heart rate response")
        if features["chol"] < 200:
            protective.append("Desirable cholesterol range")
        if features["trestbps"] <= 120:
            protective.append("Normal resting blood pressure")

        if risk_level == "low":
            if protective:
                return protective[:4]
            return ["Overall feature profile falls in low-risk zone"]

        if factors:
            return factors[:4]

        return ["Risk determined by combined weighted feature profile"]


model_service = ModelService()
