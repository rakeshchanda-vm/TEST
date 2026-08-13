"""
Compliance Checker Agent.
Verifies RBI guidelines, KYC norms, PMLA, and DPDP Act 2023 compliance.
"""
import json
import logging
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from backend.core.models import LoanApplicationState
from backend.config.config import get_settings
from langchain_ollama import ChatOllama
from backend.core.prompts import COMPLIANCE_PROMPT

logger = logging.getLogger(__name__)

RBI_RULES = {
    "max_dti_ratio":            0.50,
    "min_loan_amount_personal": 10000,
    "max_loan_amount_personal": 5000000,
    "min_age":                  21,
    "max_age":                  65,
    "kyc_mandatory":            True,
    "pmla_threshold":           200000,
}

class ComplianceCheckerAgent:
    def __init__(self, llm: Optional[BaseChatModel] = None):
        """
        Args:
            llm: Optional LangChain chat model. Lazily created on first use if None.
        """
        self._llm = llm

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            s = get_settings()
            self._llm = ChatOllama(model=s.llm_model, max_tokens=1024)
        return self._llm

    @llm.setter
    def llm(self, val: BaseChatModel) -> None:
        self._llm = val

    async def check(self, state: LoanApplicationState) -> dict:
        logger.info(f"[ComplianceChecker] Checking compliance for {state.application_id}")

        flags: list[str] = []
        critical_flags: list[str] = []

        if state.debt_to_income_ratio > RBI_RULES["max_dti_ratio"]:
            flags.append(f"DTI_EXCEEDS_RBI_LIMIT_{state.debt_to_income_ratio:.0%}")
        if state.loan_amount > 0:
            if state.loan_amount < RBI_RULES["min_loan_amount_personal"]:
                flags.append("LOAN_BELOW_MINIMUM")
            if state.loan_amount > RBI_RULES["max_loan_amount_personal"]:
                critical_flags.append("LOAN_EXCEEDS_REGULATORY_CAP")
            if state.loan_amount > RBI_RULES["pmla_threshold"]:
                flags.append("PMLA_ENHANCED_DUE_DILIGENCE_REQUIRED")

        has_pan = state.identity_verified or "pan" in state.documents_received or "pan_card" in state.documents_received
        has_aadhaar = "aadhaar" in state.documents_received or "aadhaar_card" in state.documents_received
        kyc_status = "complete" if (has_pan and has_aadhaar) else "incomplete"
        rbi_compliant = len(critical_flags) == 0 and state.debt_to_income_ratio <= RBI_RULES["max_dti_ratio"]
        pmla_check = state.loan_amount > RBI_RULES["pmla_threshold"]

        app_data = {
            "loan_amount": state.loan_amount, "monthly_income": state.monthly_income,
            "dti_ratio": state.debt_to_income_ratio, "kyc_docs": state.documents_received,
            "loan_type": state.loan_type, "identity_verified": state.identity_verified,
        }
        try:
            messages = [
                SystemMessage(content=COMPLIANCE_PROMPT.format(application_data=json.dumps(app_data))),
                HumanMessage(content="Perform full compliance review."),
            ]
            response = await self.llm.ainvoke(messages)
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            result = json.loads(content.strip())
            rbi_compliant = result.get("rbi_compliant", rbi_compliant) and not critical_flags
            kyc_status = result.get("kyc_status", kyc_status)
            pmla_check = result.get("pmla_required", pmla_check)
            all_flags = flags + critical_flags + result.get("compliance_flags", []) + \
                        ["CRITICAL_" + f for f in result.get("critical_flags", [])]
        except Exception as e:
            logger.warning(f"[ComplianceChecker] LLM parse failed, using rule-based: {e}")
            all_flags = flags + critical_flags

        audit_entry = {
            "agent": "compliance_checker",
            "application_id": state.application_id,
            "rbi_compliant": rbi_compliant,
            "kyc_status": kyc_status,
            "flags": all_flags,
        }
        logger.info(f"[ComplianceChecker] RBI compliant: {rbi_compliant} | Flags: {all_flags}")

        return {
            "rbi_compliant": rbi_compliant,
            "kyc_status": kyc_status,
            "pmla_check": pmla_check,
            "compliance_flags": all_flags,
            "audit_log": [audit_entry],
        }