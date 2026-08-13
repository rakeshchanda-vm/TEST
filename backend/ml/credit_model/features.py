"""Feature engineering for credit scoring model."""
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class CreditFeatures:
    """All features used by the credit scoring model."""
    monthly_income: float
    monthly_obligations: float
    debt_to_income_ratio: float
    bank_balance_avg: float
    loan_amount: float
    loan_tenure_months: int
    cash_flow_score: float
    loan_to_income_ratio: float = 0.0
    emi_to_income_ratio: float = 0.0
    balance_volatility: float = 0.0
    savings_rate: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.monthly_income / 100000,           # normalized
            self.monthly_obligations / 100000,
            self.debt_to_income_ratio,
            self.bank_balance_avg / 100000,
            self.loan_amount / 1000000,
            self.loan_tenure_months / 360,
            self.cash_flow_score,
            self.loan_to_income_ratio,
            self.emi_to_income_ratio,
            self.balance_volatility,
            self.savings_rate,
        ], dtype=np.float32)


FEATURE_NAMES = [
    "monthly_income_norm",
    "monthly_obligations_norm",
    "debt_to_income_ratio",
    "bank_balance_avg_norm",
    "loan_amount_norm",
    "loan_tenure_norm",
    "cash_flow_score",
    "loan_to_income_ratio",
    "emi_to_income_ratio",
    "balance_volatility",
    "savings_rate",
]


def engineer_features(raw: dict) -> CreditFeatures:
    """Build CreditFeatures from raw extracted data."""
    income = raw.get("monthly_income", 0)
    obligations = raw.get("monthly_obligations", 0)
    balance = raw.get("bank_balance_avg", 0)
    loan_amount = raw.get("loan_amount", 0)

    return CreditFeatures(
        monthly_income=income,
        monthly_obligations=obligations,
        debt_to_income_ratio=obligations / max(income, 1),
        bank_balance_avg=balance,
        loan_amount=loan_amount,
        loan_tenure_months=raw.get("loan_tenure_months", 60),
        cash_flow_score=raw.get("cash_flow_score", 0.5),
        loan_to_income_ratio=loan_amount / max(income * 12, 1),
        emi_to_income_ratio=obligations / max(income, 1),
        balance_volatility=raw.get("balance_volatility", 0.2),
        savings_rate=max(0, (income - obligations) / max(income, 1)),
    )