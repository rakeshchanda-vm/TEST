"""
BFSI Loan Underwriting — LangGraph Multi-Agent Workflow.

Flow: document_validator → financial_analyst → credit_scorer
      → fraud_detector → compliance_checker → decision_agent

Runs in parallel where possible (fraud + compliance after financial analysis).
"""
import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_ollama import ChatOllama

from backend.core.models import LoanApplicationState
from backend.agents.document_validator import DocumentValidatorAgent
from backend.agents.financial_analyst import FinancialAnalystAgent
from backend.agents.credit_scorer import CreditScorerAgent
from backend.agents.fraud_detector import FraudDetectorAgent
from backend.agents.compliance_checker import ComplianceCheckerAgent
from backend.agents.decision_agent import DecisionAgent
from backend.config.config import settings

logger = logging.getLogger(__name__)


def route_after_documents(state: LoanApplicationState) -> str:
    """Route after document validation."""
    if state.error or not state.documents_validated:
        return "reject_incomplete"
    return "financial_analyst"


def route_after_financial(state: LoanApplicationState) -> list[str]:
    """After financial analysis, run fraud + credit in parallel."""
    return ["credit_scorer", "fraud_detector"]


def route_after_parallel(state: LoanApplicationState) -> str:
    """After parallel checks, run compliance."""
    return "compliance_checker"


def route_to_decision(state: LoanApplicationState) -> str:
    """Final routing to decision."""
    if state.compliance_flags and "CRITICAL" in str(state.compliance_flags):
        return "auto_reject"
    if state.fraud_risk_score > 0.85:
        return "auto_reject"
    return "decision_agent"


async def reject_incomplete(state: LoanApplicationState) -> LoanApplicationState:
    state.decision = "rejected"
    state.decision_reason = "Incomplete or invalid documents. Please resubmit."
    state.audit_log = [{"event": "auto_reject_incomplete_docs", "application_id": state.application_id}]
    return state


async def auto_reject(state: LoanApplicationState) -> LoanApplicationState:
    state.decision = "rejected"
    state.decision_reason = (
        f"Application auto-rejected due to: "
        f"fraud_score={state.fraud_risk_score:.2f}, "
        f"compliance_flags={state.compliance_flags}"
    )
    state.audit_log = [{"event": "auto_reject", "reason": state.decision_reason}]
    return state


def build_underwriting_graph(checkpointer=None) -> StateGraph:
    """Build the complete loan underwriting graph."""
    llm = ChatOllama(
        model=settings.llm_model,
        temperature=0,
    )

    doc_validator   = DocumentValidatorAgent(llm=llm)
    fin_analyst     = FinancialAnalystAgent(llm=llm)
    credit_scorer   = CreditScorerAgent()
    fraud_detector  = FraudDetectorAgent()
    compliance      = ComplianceCheckerAgent(llm=llm)
    decision        = DecisionAgent(llm=llm)

    graph = StateGraph(LoanApplicationState)

    # Add nodes
    graph.add_node("document_validator",  doc_validator.validate)
    graph.add_node("financial_analyst",   fin_analyst.analyze)
    graph.add_node("credit_scorer",       credit_scorer.score)
    graph.add_node("fraud_detector",      fraud_detector.detect)
    graph.add_node("compliance_checker",  compliance.check)
    graph.add_node("decision_agent",      decision.decide)
    graph.add_node("reject_incomplete",   reject_incomplete)
    graph.add_node("auto_reject",         auto_reject)

    # Edges
    graph.add_edge(START, "document_validator")
    graph.add_conditional_edges("document_validator", route_after_documents)
    graph.add_edge("financial_analyst", "credit_scorer")
    graph.add_edge("financial_analyst", "fraud_detector")
    graph.add_edge("credit_scorer", "compliance_checker")
    graph.add_edge("fraud_detector", "compliance_checker")
    graph.add_conditional_edges("compliance_checker", route_to_decision)
    graph.add_edge("decision_agent", END)
    graph.add_edge("reject_incomplete", END)
    graph.add_edge("auto_reject", END)

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["decision_agent"],  # Pause for human review if needed
    )