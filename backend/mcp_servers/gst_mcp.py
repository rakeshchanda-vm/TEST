"""
GST MCP Server — GSTIN verification and GST return analysis for MSME lending.
"""
import hashlib
import re
import random
from datetime import datetime
from contextlib import asynccontextmanager
from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Mount
from backend.config.config import settings

mcp = FastMCP("GST Intelligence Server")

GSTIN_PATTERN = re.compile(r"^[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$")


def _deterministic_gst_data(gstin: str) -> dict:
    seed = int(hashlib.md5(gstin.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    annual_turnover = rng.randint(2_000_000, 50_000_000)  # 20L to 5Cr
    monthly_turnover = annual_turnover / 12

    # Filing compliance: some months may be missing
    months_filed = rng.randint(8, 12)
    compliance_rate = months_filed / 12

    return {
        "gstin": gstin,
        "trade_name": f"BUSINESS_{gstin[-4:]} ENTERPRISES",
        "legal_name": f"M/S ENTERPRISE {gstin[2:7]}",
        "state_code": gstin[:2],
        "registration_date": f"20{rng.randint(18,23)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
        "taxpayer_type": rng.choice(["Regular", "Composition", "Regular"]),
        "status": rng.choice(["Active", "Active", "Active", "Cancelled"]),
        "annual_turnover": annual_turnover,
        "monthly_avg_turnover": monthly_turnover,
        "gst3b_months_filed": months_filed,
        "gstr1_months_filed": rng.randint(months_filed - 1, months_filed),
        "compliance_rate": compliance_rate,
        "input_tax_credit_utilized": annual_turnover * rng.uniform(0.05, 0.15),
        "tax_paid_annual": annual_turnover * rng.uniform(0.05, 0.18),
    }


@mcp.tool()
async def verify_gstin(gstin: str) -> dict:
    """
    Verify GSTIN registration status and fetch basic taxpayer details.
    """
    gstin = gstin.strip().upper()

    if not GSTIN_PATTERN.match(gstin):
        return {
            "valid": False,
            "error": "Invalid GSTIN format. Expected: 2-digit state code + PAN + entity code + Z + checksum",
        }

    data = _deterministic_gst_data(gstin)

    return {
        "valid": True,
        "gstin": gstin,
        "trade_name": data["trade_name"],
        "legal_name": data["legal_name"],
        "registration_date": data["registration_date"],
        "taxpayer_type": data["taxpayer_type"],
        "status": data["status"],
        "state_code": data["state_code"],
        "verified_at": datetime.now().isoformat(),
        "source": "GSTN API (Simulated)",
    }


@mcp.tool()
async def fetch_gst_returns(gstin: str, periods: int = 4) -> dict:
    """
    Fetch GST return filing history and turnover analysis.
    Used for MSME loan underwriting — turnover verification and filing compliance.

    Args:
        gstin: Valid GSTIN number
        periods: Number of quarterly periods to analyze (default: 4 = 1 year)
    """
    gstin = gstin.strip().upper()

    if not GSTIN_PATTERN.match(gstin):
        return {"error": "Invalid GSTIN format"}

    data = _deterministic_gst_data(gstin)

    # Quarterly breakdown
    quarters = []
    for i in range(periods):
        quarter_turnover = data["annual_turnover"] / 4 * random.uniform(0.7, 1.3)
        quarters.append({
            "period": f"Q{(i % 4) + 1}-FY{25 - (i // 4)}",
            "turnover": round(quarter_turnover, 2),
            "tax_paid": round(quarter_turnover * 0.09, 2),
            "gstr1_filed": True,
            "gstr3b_filed": random.random() > 0.1,
            "filing_date": f"2025-{(12 - i*3):02d}-15",
        })

    return {
        "gstin": gstin,
        "legal_name": data["legal_name"],
        "summary": {
            "annual_turnover": data["annual_turnover"],
            "monthly_avg_turnover": round(data["monthly_avg_turnover"], 2),
            "tax_paid_annual": round(data["tax_paid_annual"], 2),
            "itc_utilized": round(data["input_tax_credit_utilized"], 2),
            "compliance_rate": data["compliance_rate"],
            "gst3b_months_filed": data["gst3b_months_filed"],
            "gstr1_months_filed": data["gstr1_months_filed"],
        },
        "quarterly_returns": quarters,
        "risk_assessment": {
            "filing_compliance": "good" if data["compliance_rate"] >= 0.9 else ("fair" if data["compliance_rate"] >= 0.7 else "poor"),
            "turnover_trend": "stable",
            "status": data["status"],
        },
        "lending_summary": {
            "eligible_loan_amount": round(data["annual_turnover"] * 0.2, 2),
            "turnover_verified": True,
            "compliance_score": round(data["compliance_rate"] * 100),
        },
    }


session_manager = StreamableHTTPSessionManager(
    app=mcp._mcp_server,
    event_store=None,
    json_response=True,
    stateless=True,
)

@asynccontextmanager
async def lifespan(app: Starlette):
    async with session_manager.run():
        yield

app = Starlette(
    routes=[Mount("/mcp", app=session_manager.handle_request)],lifespan=lifespan,
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.gst_mcp_port)