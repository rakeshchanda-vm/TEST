"""
Decision Agent — Final underwriting decision with full explainability.
Synthesizes all agent signals into approve/reject/review with SHAP-based reasoning.
"""
import json
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from backend.core.models import LoanApplicationState
from backend.core.prompts import DECISION_PROMPT

logger = logging.getLogger(__name__)

class DecisionAgent:
    def __init__(self, llm: ChatOllama):
        self.llm = llm

    async def decide(self, state: LoanApplicationState) -> LoanApplicationState:
        logger.info(f"[DecisionAgent] Making final decision for {state.application_id}")

        summary = {
            "application_id":       state.application_id,
            "loan_type":            state.loan_type,
            "requested_amount":     state.loan_amount,
            "monthly_income":       state.monthly_income,
            "credit_score":         state.credit_score,
            "bureau_score":         state.bureau_score,
            "fraud_risk_score":     state.fraud_risk_score,
            "fraud_flags":          state.fraud_flags,
            "dti_ratio":            state.debt_to_income_ratio,
            "rbi_compliant":        state.rbi_compliant,
            "kyc_status":           state.kyc_status,
            "compliance_flags":     state.compliance_flags,
            "top_score_factors":    state.score_factors[:5],
        }

        messages = [
            SystemMessage(content=DECISION_PROMPT.format(summary=json.dumps(summary, indent=2))),
            HumanMessage(content="Make the final underwriting decision."),
        ]

        response = await self.llm.ainvoke(messages)

        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            result = json.loads(content.strip())

            state.decision = result.get("decision", "MANUAL_REVIEW")
            state.approved_amount = float(result.get("approved_amount", 0))
            state.interest_rate = float(result.get("interest_rate", 0))
            state.approved_tenure = int(result.get("approved_tenure_months", 0))
            state.confidence = float(result.get("confidence", 0.5))
            state.key_factors = result.get("key_factors", [])
            state.decision_reason = result.get("decision_reason", "")
            state.decision_explanation = result.get("internal_notes", "")

        except Exception as e:
            logger.error(f"[DecisionAgent] Parse error: {e}")
            state.decision = "MANUAL_REVIEW"
            state.decision_reason = "System error during decisioning. Escalated for manual review."
            state.confidence = 0.0

        state.audit_log = [{
            "agent": "decision_agent",
            "application_id": state.application_id,
            "decision": state.decision,
            "confidence": state.confidence,
            "approved_amount": state.approved_amount,
            "interest_rate": state.interest_rate,
        }]

        logger.info(f"[DecisionAgent] Decision: {state.decision} | Confidence: {state.confidence:.0%} | Amount: ₹{state.approved_amount:,.0f}")
        return state