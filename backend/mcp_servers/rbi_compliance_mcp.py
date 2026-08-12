"""
RBI Compliance Rules MCP Server.
Exposes RBI Master Directions and NBFC regulations as queryable tools.
"""
import json
import logging
from mcp.server.fastmcp import FastMCP
from backend.config.config import settings
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Mount
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)
mcp = FastMCP(name="RBI Compliance MCP", version="1.0.0")

RBI_MASTER_DIRECTIONS = {
    "personal_loan": {
        "max_dti":              0.50,
        "max_loan_tenure_months": 84,
        "min_age":              21,
        "max_age":              65,
        "kyc_mandatory":        True,
        "pmla_threshold":       200000,
        "penal_charges_cap":    0.02,  # Max 2% penal rate
    },
    "home_loan": {
        "max_ltv":              0.90,
        "min_age":              21,
        "max_age":              70,
        "kyc_mandatory":        True,
        "mortgage_registration": True,
    },
    "business_loan": {
        "max_dti":              0.65,
        "gst_mandatory_above":  10000000,
        "itr_years_required":   2,
    },
}

PMLA_RULES = {
    "cash_transaction_threshold":    1000000,  # ₹10L
    "suspicious_transaction_report":  200000,   # ₹2L
    "ctr_threshold":                 1000000,  # CTR filing
    "enhanced_due_diligence":        True,
}


@mcp.tool()
async def check_rbi_compliance(
    loan_type: str,
    loan_amount: float,
    dti_ratio: float,
    applicant_age: int,
    has_kyc: bool,
) -> str:
    """
    Check if a loan application meets RBI Master Direction requirements.

    Returns:
        JSON with compliance status and violated rules
    """
    rules = RBI_MASTER_DIRECTIONS.get(loan_type, RBI_MASTER_DIRECTIONS["personal_loan"])
    violations = []
    warnings = []

    if dti_ratio > rules.get("max_dti", 0.5):
        violations.append(f"DTI {dti_ratio:.0%} exceeds RBI limit of {rules['max_dti']:.0%}")

    if applicant_age < rules.get("min_age", 21):
        violations.append(f"Age {applicant_age} below minimum {rules['min_age']}")

    if applicant_age > rules.get("max_age", 65):
        violations.append(f"Age {applicant_age} exceeds maximum {rules['max_age']}")

    if not has_kyc and rules.get("kyc_mandatory"):
        violations.append("KYC incomplete — mandatory under RBI Master Directions")

    if loan_amount > PMLA_RULES["suspicious_transaction_report"]:
        warnings.append("Enhanced due diligence required under PMLA 2002")

    return json.dumps({
        "compliant": len(violations) == 0,
        "violations": violations,
        "warnings": warnings,
        "applicable_rules": f"RBI Master Directions for {loan_type.replace('_', ' ').title()}",
    })


@mcp.tool()
async def get_fair_lending_guidelines() -> str:
    """Return RBI Fair Lending Guidelines for transparent credit decisions."""
    return json.dumps({
        "guidelines": [
            "Communicate loan decision within 30 days of complete application",
            "Provide specific reasons for rejection (RBI mandate since 2023)",
            "Interest rate must be disclosed as APR",
            "No discriminatory lending based on religion, caste, or gender",
            "Grievance redressal mechanism mandatory",
            "Credit bureau reporting within 30 days of disbursement",
        ],
        "source": "RBI Master Circular on Customer Service 2024",
    })

session_manager = StreamableHTTPSessionManager(app=mcp._mcp_server, json_response=True, stateless=True)

@asynccontextmanager
async def lifespan(app: Starlette):
    async with session_manager.run():
        yield
app = Starlette(routes=[Mount("/mcp", app=session_manager.handle_request)],lifespan=lifespan,)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.rbi_compliance_mcp_port)