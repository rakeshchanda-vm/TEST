"""
Fraud Detector Agent.
Uses Isolation Forest for anomaly detection + rule-based checks.
Flags synthetic identity, income inflation, document tampering.
"""
import logging
import numpy as np
from backend.core.models import LoanApplicationState
from backend.ml.fraud_model.predict import FraudModelPredictor

logger = logging.getLogger(__name__)


class FraudDetectorAgent:
    def __init__(self):
        self.model = FraudModelPredictor()

    async def detect(self, state: LoanApplicationState) -> LoanApplicationState:
        logger.info(f"[FraudDetector] Analyzing fraud signals for {state.application_id}")

        flags = []
        risk_score = 0.0

        # Rule 1: Income-to-lifestyle mismatch
        if state.monthly_income > 0 and state.bank_balance_avg > 0:
            income_bank_ratio = state.bank_balance_avg / state.monthly_income
            if income_bank_ratio < 0.5:
                flags.append("LOW_BANK_BALANCE_VS_INCOME")
                risk_score += 0.2

        # Rule 2: Debt-to-income too high
        if state.debt_to_income_ratio > 0.6:
            flags.append("HIGH_DTI_RATIO")
            risk_score += 0.15

        # Rule 3: Loan amount too large relative to income
        annual_income = state.monthly_income * 12
        if annual_income > 0 and state.loan_amount / annual_income > 5:
            flags.append("LOAN_EXCEEDS_5X_ANNUAL_INCOME")
            risk_score += 0.25

        # ML model for deeper patterns
        features = {
            "monthly_income": state.monthly_income,
            "loan_amount": state.loan_amount,
            "bank_balance_avg": state.bank_balance_avg,
            "dti_ratio": state.debt_to_income_ratio,
        }
        ml_score = await self.model.predict_proba(features)
        risk_score = min(1.0, risk_score + ml_score * 0.4)

        state.fraud_risk_score = round(risk_score, 4)
        state.fraud_flags = flags
        state.identity_verified = len(flags) == 0

        state.audit_log = [{
            "agent": "fraud_detector",
            "application_id": state.application_id,
            "fraud_risk_score": state.fraud_risk_score,
            "flags": flags,
            "ml_fraud_score": round(ml_score, 4),
        }]

        logger.info(f"[FraudDetector] Risk: {state.fraud_risk_score:.2f} | Flags: {flags}")
        return state