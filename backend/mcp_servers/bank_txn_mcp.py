"""
Bank Transaction MCP Server — account aggregation & transaction analysis.
Simulates AA (Account Aggregator) framework integration per RBI guidelines.
"""

import hashlib
import random
from datetime import datetime, timedelta
from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Mount
from backend.config.config import settings
from contextlib import asynccontextmanager


mcp = FastMCP("Bank Transaction Analyzer")


def _mock_transactions(account_number: str, months: int = 6) -> list[dict]:
    """Generate deterministic mock transactions based on account number for defined number of months.
    Parameter
    account_number = Account number of the applicant
    months = numnber of months to retrive transcations record
    """
    seed = int(hashlib.md5(account_number.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    transactions = []
    base_salary = rng.randint(40000, 200000)
    base_date = datetime.now()

    for m in range(months):
        month_start = base_date - timedelta(days=30 * (months - m))
        # Salary credit
        salary_date = month_start + timedelta(days=rng.randint(1, 5))
        transactions.append({
            "date": salary_date.strftime("%Y-%m-%d"),
            "description": "NEFT SALARY CREDIT",
            "amount": base_salary * rng.uniform(0.95, 1.05),
            "type": "credit",
            "balance": base_salary * rng.uniform(1.2, 2.5),
            "category": "salary",
        })

        # EMI debits
        emi_amount = base_salary * rng.uniform(0.1, 0.35)
        emi_date = month_start + timedelta(days=rng.randint(3, 7))
        transactions.append({
            "date": emi_date.strftime("%Y-%m-%d"),
            "description": f"EMI/{rng.randint(100000, 999999)}/HDFC BANK",
            "amount": emi_amount,
            "type": "debit",
            "balance": base_salary * rng.uniform(0.8, 1.5),
            "category": "emi",
        })

        # Utility / monthly expenses
        for _ in range(rng.randint(5, 15)):
            categories = ["grocery", "utility", "dining", "transport", "shopping"]
            cat = rng.choice(categories)
            txn_date = month_start + timedelta(days=rng.randint(1, 28))
            transactions.append({
                "date": txn_date.strftime("%Y-%m-%d"),
                "description": f"UPI/{cat.upper()}/txn{rng.randint(1000, 9999)}",
                "amount": rng.uniform(500, 5000),
                "type": "debit",
                "balance": base_salary * rng.uniform(0.3, 1.8),
                "category": cat,
            })

    return sorted(transactions, key=lambda x: x["date"])


@mcp.tool()
async def fetch_bank_statement(
    account_number: str,
    months: int = 6,
) -> dict:
    """
    Fetch bank statement analysis via Account Aggregator framework.
    Returns income, expense, balance trends, and obligation detection.
    """
    if not account_number or len(account_number) < 8:
        return {"error": "Invalid account number"}

    transactions = _mock_transactions(account_number, months)

    credits = [t for t in transactions if t["type"] == "credit"]
    debits = [t for t in transactions if t["type"] == "debit"]

    total_credits = sum(t["amount"] for t in credits)
    total_debits = sum(t["amount"] for t in debits)
    avg_monthly_credit = total_credits / months
    avg_monthly_debit = total_debits / months

    # Detect EMI obligations
    emi_txns = [t for t in debits if t["category"] == "emi"]
    total_emi = sum(t["amount"] for t in emi_txns)
    avg_monthly_emi = total_emi / months

    # Balance stats
    balances = [t["balance"] for t in transactions]
    avg_balance = sum(balances) / len(balances)
    min_balance = min(balances)
    max_balance = max(balances)

    # Salary regularity check
    salary_txns = [t for t in credits if "salary" in t["description"].lower() or t["category"] == "salary"]
    salary_months = len(salary_txns)
    salary_regularity = "regular" if salary_months >= months - 1 else "irregular"

    # Bounce detection (simplified — min balance dips)
    bounce_risk = "high" if min_balance < avg_monthly_credit * 0.1 else ("medium" if min_balance < avg_monthly_credit * 0.3 else "low")

    return {
        "account_number_masked": f"XXXX{account_number[-4:]}",
        "analysis_period_months": months,
        "summary": {
            "total_credits": round(total_credits, 2),
            "total_debits": round(total_debits, 2),
            "avg_monthly_credit": round(avg_monthly_credit, 2),
            "avg_monthly_debit": round(avg_monthly_debit, 2),
            "avg_monthly_emi": round(avg_monthly_emi, 2),
            "avg_balance": round(avg_balance, 2),
            "min_balance": round(min_balance, 2),
            "max_balance": round(max_balance, 2),
            "net_monthly_savings": round(avg_monthly_credit - avg_monthly_debit, 2),
        },
        "income_analysis": {
            "salary_regularity": salary_regularity,
            "salary_months_found": salary_months,
            "primary_income_source": "salary" if salary_months > 0 else "business_receipts",
        },
        "obligation_analysis": {
            "detected_emi_per_month": round(avg_monthly_emi, 2),
            "obligation_to_income_ratio": round(avg_monthly_emi / avg_monthly_credit, 3) if avg_monthly_credit > 0 else 0,
            "bounce_risk": bounce_risk,
        },
        "transaction_count": len(transactions),
        "data_source": "Account Aggregator (RBI Licensed)",
        "fetched_at": datetime.now().isoformat(),
    }


@mcp.tool()
async def detect_obligations(account_number: str) -> dict:
    """
    Detect recurring debits (EMIs, subscriptions, insurance) from transaction history.
    Returns itemized list of detected obligations.
    """
    transactions = _mock_transactions(account_number, months=6)
    emi_txns = [t for t in transactions if t["category"] == "emi"]

    obligations = []
    for txn in emi_txns[:5]:  # Show up to 5 EMIs
        obligations.append({
            "description": txn["description"],
            "monthly_amount": round(txn["amount"], 2),
            "type": "emi",
            "detected_date": txn["date"],
        })

    total_monthly = sum(o["monthly_amount"] for o in obligations)

    return {
        "detected_obligations": obligations,
        "total_monthly_obligations": round(total_monthly, 2),
        "obligation_count": len(obligations),
    }


# Starlette ASGI app
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

app = Starlette(routes=[Mount("/mcp", app=session_manager.handle_request)],lifespan=lifespan,)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.bank_txn_mcp_port)