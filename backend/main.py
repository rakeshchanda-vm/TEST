"""BFSI Credit Intelligence Platform — FastAPI Application."""
import logging
import uuid
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.config.config import settings
from backend.agents.graph import build_underwriting_graph
from backend.core.models import LoanApplicationState
from fastapi import FastAPI
from backend.core.models import LoanApplicationRequest, LoanDecisionResponse

logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🏦 Starting BFSI Credit Intelligence Platform")
    app.state.graph = build_underwriting_graph()
    logger.info("✅ Underwriting graph compiled | Kafka producer ready")
    yield
    logger.info("🛑 Shutting down BFSI platform")


app = FastAPI(
    title="BFSI Credit Intelligence Platform",
    description="AI-powered loan underwriting with 6 specialized agents",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/api/v1/loans/underwrite", response_model=LoanDecisionResponse)
async def underwrite_loan(request: LoanApplicationRequest):
    """
    Submit a loan application for AI-powered underwriting.
    Runs through 6 specialized agents: validation → financial → credit → fraud → compliance → decision.
    Average processing time: 60-120 seconds for full pipeline.
    """
    import time
    start_time = time.time()
    application_id = f"LOAN-{uuid.uuid4().hex[:8].upper()}"

    initial_state = LoanApplicationState(
        application_id=application_id,
        applicant_id=request.applicant_id,
        loan_type=request.loan_type,
        loan_amount=request.loan_amount,
        loan_tenure_months=request.loan_tenure_months,
        documents_received=request.documents_received,
        monthly_income=request.monthly_income,
        monthly_obligations=request.monthly_obligations,
        bank_balance_avg=request.bank_balance_avg,
        bureau_score=request.bureau_score,
        debt_to_income_ratio=request.monthly_obligations / max(request.monthly_income, 1),
    )

    config = {"configurable": {"thread_id": application_id}}
    final_state = await app.state.graph.ainvoke(initial_state.model_dump(), config=config)

    processing_ms = int((time.time() - start_time) * 1000)

    return LoanDecisionResponse(
        application_id=application_id,
        decision=final_state.get("decision", "MANUAL_REVIEW"),
        approved_amount=final_state.get("approved_amount", 0),
        interest_rate=final_state.get("interest_rate", 0),
        tenure_months=final_state.get("approved_tenure", 0),
        confidence=final_state.get("confidence", 0),
        key_factors=final_state.get("key_factors", []),
        decision_reason=final_state.get("decision_reason", ""),
        shap_explanation=final_state.get("shap_values", {}),
        rbi_compliant=final_state.get("rbi_compliant", False),
        processing_time_ms=processing_ms,
    )


@app.get("/api/v1/health/live")
async def liveness(): return {"status": "ok"}

@app.get("/api/v1/health/ready")
async def readiness(): return {"status": "ready", "graph": "compiled"}