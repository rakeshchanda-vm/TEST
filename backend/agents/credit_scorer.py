"""
Credit Scorer Agent.
Combines bureau score + internal ML model (XGBoost) + LLM reasoning.
Returns credit score, risk tier, and SHAP-based explanations.
"""
import logging
import numpy as np
from backend.core.models import LoanApplicationState
from backend.ml.credit_model.predict import CreditModelPredictor
from backend.ml.explainability.shap_explainer import SHAPExplainer

logger = logging.getLogger(__name__)

RISK_TIERS = {
    (750, 900): "PRIME",
    (700, 749): "NEAR_PRIME",
    (650, 699): "SUBPRIME",
    (0, 649):   "HIGH_RISK",
}

def get_risk_tier(score: int) -> str:
    for (low, high), tier in RISK_TIERS.items():
        if low <= score <= high:
            return tier
    return "HIGH_RISK"


class CreditScorerAgent:
    def __init__(self):
        self.model = CreditModelPredictor()
        self.explainer = SHAPExplainer()

    async def score(self, state: LoanApplicationState) -> LoanApplicationState:
        logger.info(f"[CreditScorer] Scoring application {state.application_id}")

        # Build feature vector from extracted data
        features = self._build_features(state)

        # ML model prediction
        score, proba = await self.model.predict(features)
        state.internal_score = float(score)

        # SHAP explanations
        shap_vals = await self.explainer.explain(features)
        state.shap_values = shap_vals
        state.score_factors = self._format_shap_factors(shap_vals)

        # Combine with bureau score (weighted average)
        bureau = state.bureau_score or 650  # default if bureau unavailable
        combined = int(0.6 * bureau + 0.4 * (score * 900))
        state.credit_score = combined

        # Determine risk tier
        tier = get_risk_tier(combined)
        state.audit_log = [{
            "agent": "credit_scorer",
            "application_id": state.application_id,
            "bureau_score": bureau,
            "internal_score": round(score, 4),
            "combined_score": combined,
            "risk_tier": tier,
            "top_factors": state.score_factors[:3],
        }]

        logger.info(f"[CreditScorer] Score: {combined} | Tier: {tier}")
        return state

    def _build_features(self, state: LoanApplicationState) -> dict:
        """Build ML feature vector from state."""
        return {
            "monthly_income":        state.monthly_income,
            "monthly_obligations":   state.monthly_obligations,
            "debt_to_income_ratio":  state.debt_to_income_ratio,
            "bank_balance_avg":      state.bank_balance_avg,
            "loan_amount":           state.loan_amount,
            "loan_tenure":           state.loan_tenure_months,
            "cash_flow_score":       state.cash_flow_score,
            "loan_to_income":        state.loan_amount / max(state.monthly_income * 12, 1),
            "emi_to_income":         state.monthly_obligations / max(state.monthly_income, 1),
        }

    def _format_shap_factors(self, shap_vals: dict) -> list[dict]:
        """Convert SHAP values to human-readable factors."""
        factors = []
        for feature, value in sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True):
            factors.append({
                "feature": feature.replace("_", " ").title(),
                "impact": "positive" if value > 0 else "negative",
                "magnitude": abs(round(value, 4)),
            })
        return factors[:10]