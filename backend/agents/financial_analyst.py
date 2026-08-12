
"""
Financial Analyst Agent — extracts income, obligations, ratios from documents.
"""
import json
import re
from typing import Any
from backend.core.prompts import FINANCIAL_ANALYST_PROMPT
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from backend.core.models import LoanApplicationState
from backend.config.config import settings

class FinancialAnalystAgent:
    def __init__(self, llm=None):
        self.llm = ChatOllama(
            model=settings.llm_model,
            max_tokens=2048,
        )

    def _extract_financials_heuristic(self, extracted_data: dict) -> dict:
        """Rule-based extraction fallback when LLM is unavailable."""
        monthly_income = 0.0
        monthly_obligations = 0.0
        bank_balance_avg = 0.0

        # Salary slip extraction
        salary_data = extracted_data.get("salary_slip", {})
        if salary_data:
            net_salary = salary_data.get("net_salary", 0) or salary_data.get("net_pay", 0)
            monthly_income = float(net_salary) if net_salary else 0.0

        # ITR extraction
        itr_data = extracted_data.get("itr", {})
        if itr_data and not monthly_income:
            annual = itr_data.get("gross_total_income", 0) or itr_data.get("total_income", 0)
            monthly_income = float(annual) / 12 if annual else 0.0

        # Bank statement extraction
        bank_data = extracted_data.get("bank_statement", {})
        if bank_data:
            avg_bal = bank_data.get("average_balance", 0) or bank_data.get("avg_monthly_balance", 0)
            bank_balance_avg = float(avg_bal) if avg_bal else 0.0

            # Infer obligations from outward debits pattern
            avg_debit = bank_data.get("average_monthly_debit", 0)
            if avg_debit and monthly_income:
                # Conservative: assume 30% of debits are obligations
                monthly_obligations = float(avg_debit) * 0.3

        dti = (monthly_obligations / monthly_income) if monthly_income > 0 else 0.0

        red_flags = []
        if dti > 0.5:
            red_flags.append(f"High DTI ratio: {dti:.1%}")
        if bank_balance_avg < monthly_income * 0.5:
            red_flags.append("Low average bank balance relative to income")

        return {
            "monthly_income": monthly_income,
            "annual_income": monthly_income * 12,
            "income_source": "salary" if salary_data else ("business" if itr_data else "unknown"),
            "income_stability": "stable",
            "monthly_obligations": monthly_obligations,
            "debt_to_income_ratio": round(dti, 4),
            "bank_balance_avg": bank_balance_avg,
            "bank_balance_min": bank_balance_avg * 0.7,
            "gst_annual_turnover": None,
            "financial_stability_score": max(0, min(100, int(80 - dti * 100))),
            "red_flags": red_flags,
            "analysis_notes": "Heuristic extraction — LLM analysis unavailable",
        }

    async def analyze(self, state: LoanApplicationState) -> dict[str, Any]:
        extracted_data = state.extracted_data or {}

        if not extracted_data:
            return {
                "monthly_income": 0.0,
                "monthly_obligations": 0.0,
                "debt_to_income_ratio": 0.0,
                "bank_balance_avg": 0.0,
                "audit_log": [
                    {
                        "stage": "financial_analyst",
                        "status": "skipped",
                        "reason": "No extracted document data available",
                    }
                ],
            }

        # Serialize extracted data for LLM
        doc_summary = json.dumps(extracted_data, indent=2, default=str)

        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=FINANCIAL_ANALYST_PROMPT),
                    HumanMessage(
                        content=f"Loan Type: {state.loan_type}\nLoan Amount Requested: ₹{state.loan_amount:,.0f}\n\nExtracted Document Data:\n{doc_summary}"
                    ),
                ]
            )

            raw = response.content.strip()
            # Extract JSON block
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                financials = json.loads(json_match.group())
            else:
                financials = self._extract_financials_heuristic(extracted_data)

        except Exception as e:
            financials = self._extract_financials_heuristic(extracted_data)
            financials["analysis_notes"] = f"LLM error ({e}), used heuristic extraction"

        # Build state updates
        updates: dict[str, Any] = {
            "monthly_income": financials.get("monthly_income", 0.0),
            "monthly_obligations": financials.get("monthly_obligations", 0.0),
            "debt_to_income_ratio": financials.get("debt_to_income_ratio", 0.0),
            "bank_balance_avg": financials.get("bank_balance_avg", 0.0),
            "audit_log": [
                {
                    "stage": "financial_analyst",
                    "status": "completed",
                    "monthly_income": financials.get("monthly_income"),
                    "dti_ratio": financials.get("debt_to_income_ratio"),
                    "stability_score": financials.get("financial_stability_score"),
                    "red_flags": financials.get("red_flags", []),
                    "income_source": financials.get("income_source"),
                }
            ],
        }

        return updates


async def financial_analyst_node(state: LoanApplicationState) -> dict[str, Any]:
    agent = FinancialAnalystAgent()
    return await agent.analyze(state)