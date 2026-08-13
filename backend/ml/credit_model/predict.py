"""Credit model inference — loads trained XGBoost and predicts default probability."""
import logging
import numpy as np
import joblib
from pathlib import Path
from backend.ml.credit_model.features import engineer_features, CreditFeatures

logger = logging.getLogger(__name__)
MODEL_PATH = Path("models/credit/credit_model.pkl")
SCALER_PATH = Path("models/credit/scaler.pkl")


class CreditModelPredictor:
    def __init__(self):
        self._model = None
        self._scaler = None

    def _load(self):
        if self._model is None:
            if MODEL_PATH.exists():
                self._model = joblib.load(MODEL_PATH)
                self._scaler = joblib.load(SCALER_PATH)
                logger.info("✅ Credit model loaded from disk")
            else:
                logger.warning("⚠️ No trained model found. Using heuristic scorer.")

    async def predict(self, features: dict) -> tuple[float, float]:
        """Returns (credit_score_0_to_1, default_probability)."""
        self._load()
        feat_obj = engineer_features(features)
        X = feat_obj.to_array().reshape(1, -1)

        if self._model is not None:
            if self._scaler:
                X = self._scaler.transform(X)
            proba = self._model.predict_proba(X)[0][1]  # P(default)
            score = 1.0 - proba  # Higher score = lower risk
        else:
            # Heuristic fallback
            dti = features.get("debt_to_income_ratio", 0.5)
            score = max(0.1, min(0.95, 1.0 - dti))

        return float(score), float(1.0 - score)