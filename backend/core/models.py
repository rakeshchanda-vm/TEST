"""BFSI Underwriting Agent State — tracks full loan application lifecycle."""
from typing import Annotated, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
import operator


class LoanApplicationState(BaseModel):
    """Complete state for a loan underwriting workflow."""

    # Conversation
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # Application identifiers
    application_id: str = ""
    applicant_id: str = ""
    applicant_name: str = ""          # Full name of applicant
    pan_number: str = ""              # PAN card number (KYC)
    loan_type: str = ""               # personal, home, business, vehicle, msme
    loan_amount: float = 0.0
    tenure_months: int = 0            # Alias for loan_tenure_months (UI-facing)
    loan_tenure_months: int = 0

    # Document processing results
    documents_received: list[str] = Field(default_factory=list)
    documents_validated: bool = False
    extracted_data: dict[str, Any] = Field(default_factory=dict)

    # Financial analysis
    monthly_income: float = 0.0
    monthly_obligations: float = 0.0
    debt_to_income_ratio: float = 0.0
    bank_balance_avg: float = 0.0
    cash_flow_score: float = 0.0
    financial_analysis: dict = Field(default_factory=dict)

    # Credit scoring
    credit_score: float = 0.0
    bureau_score: int = 0
    internal_score: float = 0.0
    score_factors: list[dict] = Field(default_factory=list)

    # Fraud signals
    fraud_risk_score: float = 0.0
    fraud_flags: list[str] = Field(default_factory=list)
    identity_verified: bool = False
    bank_account_verified: bool = False

    # Compliance
    kyc_status: str = ""
    pmla_check: bool = False
    rbi_compliant: bool = False
    compliance_flags: list[str] = Field(default_factory=list)

    # Decision
    decision: str = ""
    decision_reason: str = ""
    approved_amount: float = 0.0
    approved_tenure: int = 0
    interest_rate: float = 0.0
    confidence: float = 0.0

    # Explainability
    shap_values: dict = Field(default_factory=dict)
    decision_explanation: str = ""
    key_factors: list[str] = Field(default_factory=list)

    # Audit trail (immutable append)
    audit_log: Annotated[list[dict], operator.add] = Field(default_factory=list)

    # Flow control
    current_agent: str = ""
    next_agent: str = ""
    error: str | None = None
    processing_time_ms: int = 0