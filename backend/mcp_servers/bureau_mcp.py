"""
Credit Bureau MCP Server.
Simulates CIBIL/Experian/Equifax bureau score access.
In production: replace with actual bureau API integration.
"""
import json
import random
import logging
import hashlib
from mcp.server.fastmcp import FastMCP
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Mount
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from backend.config.config import settings
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)
mcp = FastMCP(name="Credit Bureau MCP Server", version="1.0.0")


@mcp.tool()
async def fetch_bureau_score(
    pan_number: str,
    dob: str,
    phone: str | None = None,
) -> str:
    """
    Fetch credit bureau score for an individual.
    Connects to CIBIL/Experian API in production.

    Args:
        pan_number: PAN card number (masked in logs)
        dob: Date of birth in YYYY-MM-DD format
        phone: Mobile number for additional verification

    Returns:
        JSON with bureau score, account summary, and payment history
    """
    # Deterministic mock based on PAN hash (consistent for same applicant)
    pan_hash = int(hashlib.md5(pan_number.encode()).hexdigest(), 16)
    base_score = 550 + (pan_hash % 350)  # Score between 550-900

    # Simulate bureau response structure
    response = {
        "bureau": "CIBIL",
        "score": base_score,
        "score_version": "V2",
        "report_date": "2026-05-01",
        "accounts_summary": {
            "total_accounts": (pan_hash % 10) + 1,
            "active_accounts": (pan_hash % 5) + 1,
            "closed_accounts": (pan_hash % 5),
            "overdue_accounts": 1 if base_score < 650 else 0,
        },
        "payment_history": {
            "on_time_payments_pct": min(100, 60 + (base_score - 550) // 3),
            "missed_payments_last_12m": max(0, (700 - base_score) // 50),
        },
        "credit_utilization": round(random.uniform(0.1, 0.7), 2),
        "oldest_account_years": (pan_hash % 15) + 1,
        "inquiries_last_6m": pan_hash % 4,
        "negative_factors": [] if base_score > 700 else ["Late payment in last 24 months"],
    }
    logger.info(f"[BureauMCP] Score fetched for PAN: {pan_number[:4]}****")
    return json.dumps(response)


@mcp.tool()
async def fetch_bureau_report_details(pan_number: str) -> str:
    """
    Fetch detailed bureau report — full account-level breakdown.

    Args:
        pan_number: PAN card number

    Returns:
        Detailed account history and credit behavior
    """
    pan_hash = int(hashlib.md5(pan_number.encode()).hexdigest(), 16)
    accounts = []
    for i in range((pan_hash % 5) + 1):
        accounts.append({
            "account_type": ["Personal Loan", "Credit Card", "Home Loan", "Two Wheeler"][i % 4],
            "lender": ["HDFC Bank", "SBI", "Axis Bank", "ICICI"][i % 4],
            "sanctioned_amount": [50000, 200000, 2000000, 75000][i % 4],
            "current_balance": [20000, 80000, 1500000, 30000][i % 4],
            "emi": [2000, 5000, 15000, 2500][i % 4],
            "status": "Active" if i < 3 else "Closed",
            "dpd_last_12m": 0 if i == 0 else (30 if pan_hash % 5 == 0 else 0),
        })
    return json.dumps({"accounts": accounts, "total": len(accounts)})

session_manager = StreamableHTTPSessionManager(app=mcp._mcp_server, json_response=True, stateless=True)

@asynccontextmanager
async def lifespan(app: Starlette):
    async with session_manager.run():
        yield
app = Starlette(routes=[Mount("/mcp", app=session_manager.handle_request)],lifespan=lifespan,)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.bureau_mcp_port)