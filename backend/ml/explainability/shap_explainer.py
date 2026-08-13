"""SHAP-based model explainability for RBI compliance and transparency."""
import logging
import numpy as np
import shap
import joblib
from pathlib import Path
from backend.ml.credit_model.features import FEATURE_NAMES, engineer_features

logger = logging.getLogger(__name__)
MODEL_PATH = Path("models/credit/credit_model.pkl")

class SHAPExplainer:
    def __init__(self):
        self._explainer = None

    def _load(self):
        if self._explainer is None and MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            self._explainer = shap.TreeExplainer(model)
            logger.info("✅ SHAP explainer initialized")

    async def explain(self, features: dict) -> dict:
        """
        Generate SHAP values for a single prediction.
        Returns feature → shap_value mapping.
        RBI requires explainability for credit decisions.
        """
        self._load()
        feat_obj = engineer_features(features)
        X = feat_obj.to_array().reshape(1, -1)

        if self._explainer is not None:
            shap_values = self._explainer.shap_values(X)[0]
            return {name: float(val) for name, val in zip(FEATURE_NAMES, shap_values)}

        # Fallback: simple feature importance heuristic
        return {
            "debt_to_income_ratio": -0.3 * features.get("debt_to_income_ratio", 0.5),
            "monthly_income_norm":  0.2 * min(1.0, features.get("monthly_income", 0) / 100000),
            "loan_amount_norm":     -0.1 * min(1.0, features.get("loan_amount", 0) / 1000000),
        }

    def generate_explanation_report(self, shap_vals: dict, decision: str) -> str:
        """Generate human-readable explanation for RBI compliance documentation."""
        top_positive = [(k, v) for k, v in shap_vals.items() if v > 0]
        top_negative = [(k, v) for k, v in shap_vals.items() if v < 0]
        top_positive.sort(key=lambda x: -x[1])
        top_negative.sort(key=lambda x: x[1])

        report = f"Credit Decision: {decision}\n\n"
        report += "Factors supporting approval:\n"
        for feat, val in top_positive[:3]:
            report += f"  • {feat.replace('_', ' ').title()}: +{val:.3f}\n"
        report += "\nFactors against approval:\n"
        for feat, val in top_negative[:3]:
            report += f"  • {feat.replace('_', ' ').title()}: {val:.3f}\n"

        return report