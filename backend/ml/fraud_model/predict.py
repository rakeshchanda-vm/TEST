"""Fraud detection model — Isolation Forest + rule-based scoring."""
import logging
import numpy as np
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)
MODEL_PATH = Path("models/fraud/fraud_model.pkl")

class FraudModelPredictor:
    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None and MODEL_PATH.exists():
            self._model = joblib.load(MODEL_PATH)
            logger.info("✅ Fraud model loaded")

    async def predict_proba(self, features: dict) -> float:
        """Returns fraud probability 0.0-1.0."""
        self._load()
        X = np.array([[
            features.get("monthly_income", 0) / 100000,
            features.get("loan_amount", 0) / 1000000,
            features.get("bank_balance_avg", 0) / 100000,
            features.get("dti_ratio", 0.5),
        ]])

        if self._model is not None:
            score = self._model.decision_function(X)[0]
            # Isolation Forest: negative score = more anomalous
            fraud_prob = max(0.0, min(1.0, (-score + 0.5)))
            return fraud_prob

        # Heuristic fallback
        dti = features.get("dti_ratio", 0.5)
        return min(0.9, max(0.0, dti - 0.3))