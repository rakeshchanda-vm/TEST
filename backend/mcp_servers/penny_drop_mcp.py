"""
Penny Drop MCP Server — bank account verification via penny drop / IMPS.
Also handles Aadhaar e-KYC, PAN verification, and identity checks.
"""
import hashlib
import re
from datetime import datetime

from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Mount
from backend.config.config import settings
from contextlib import asynccontextmanager

mcp = FastMCP("Identity & Account Verification")

PAN_PATTERN = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]{1}$")
AADHAAR_PATTERN = re.compile(r"^\d{12}$")


@mcp.tool()
async def verify_bank_account(
    account_number: str,
    ifsc_code: str,
    account_holder_name: str,
) -> dict:
    """
    Verify bank account ownership via penny drop (₹1 IMPS credit).
    Returns account holder name from bank records for KYC matching.
    """
    if not account_number or len(account_number) < 9:
        return {"verified": False, "error": "Invalid account number"}

    # Deterministic mock: hash account details
    hash_val = int(hashlib.md5(f"{account_number}{ifsc_code}".encode()).hexdigest()[:6], 16)
    is_valid = hash_val % 10 > 1  # 80% success rate

    if not is_valid:
        return {
            "verified": False,
            "error": "Account not found or IFSC mismatch",
            "account_number_masked": f"XXXX{account_number[-4:]}",
        }

    # Name matching (simplified fuzzy)
    name_parts = account_holder_name.upper().split()
    mock_name = " ".join(reversed(name_parts)) if len(name_parts) > 1 else account_holder_name.upper()

    name_match_score = 0.9 if account_holder_name.upper() in mock_name or mock_name in account_holder_name.upper() else 0.75

    return {
        "verified": True,
        "account_number_masked": f"XXXX{account_number[-4:]}",
        "ifsc_code": ifsc_code,
        "bank_name": _get_bank_from_ifsc(ifsc_code),
        "account_holder_name_from_bank": mock_name,
        "name_match_score": name_match_score,
        "name_match_status": "matched" if name_match_score >= 0.8 else "partial_match",
        "account_type": "savings",
        "penny_drop_amount": 1.0,
        "penny_drop_txn_id": f"IMPS{hash_val:012d}",
        "verified_at": datetime.now().isoformat(),
    }


@mcp.tool()
async def verify_pan(pan_number: str, name: str, dob: str) -> dict:
    """
    Verify PAN card via NSDL/UTIITSL API.
    Returns PAN status, name as per IT records, and filing history indicator.
    """
    pan_number = pan_number.strip().upper()

    if not PAN_PATTERN.match(pan_number):
        return {"valid": False, "error": "Invalid PAN format. Expected: AAAAA9999A"}

    # Deterministic result
    hash_val = int(hashlib.md5(pan_number.encode()).hexdigest()[:6], 16)
    is_active = hash_val % 10 > 0  # 90% active

    # PAN type from 4th character
    pan_type_map = {
        "P": "individual",
        "C": "company",
        "H": "hindu_undivided_family",
        "F": "firm",
        "A": "association",
    }
    pan_type = pan_type_map.get(pan_number[3], "individual")

    return {
        "valid": True,
        "pan_number": pan_number,
        "status": "active" if is_active else "inactive",
        "pan_type": pan_type,
        "name_as_per_pan": name.upper(),
        "dob_match": True,
        "aadhaar_linked": hash_val % 3 > 0,  # 67% linked
        "itr_filer": hash_val % 4 > 0,  # 75% ITR filer
        "last_itr_filed": "AY2024-25" if hash_val % 2 == 0 else "AY2023-24",
        "verified_at": datetime.now().isoformat(),
        "source": "NSDL PAN Verification (Simulated)",
    }


@mcp.tool()
async def verify_aadhaar_otp(aadhaar_number: str, otp: str, name: str) -> dict:
    """
    Verify Aadhaar via OTP-based e-KYC (UIDAI API).
    NOTE: In production, this initiates OTP → user enters OTP → verify flow.
    Returns demographic data for KYC.
    """
    aadhaar_clean = re.sub(r"[\s-]", "", aadhaar_number)

    if not AADHAAR_PATTERN.match(aadhaar_clean):
        return {"verified": False, "error": "Invalid Aadhaar number (must be 12 digits)"}

    # Mask Aadhaar per UIDAI rules
    masked = f"XXXX XXXX {aadhaar_clean[-4:]}"

    # Mock OTP verification (in prod: actual UIDAI API call)
    otp_valid = otp == "123456" or len(otp) == 6  # Dev mode: any 6-digit OTP passes

    if not otp_valid:
        return {"verified": False, "error": "OTP verification failed"}

    hash_val = int(hashlib.md5(aadhaar_clean.encode()).hexdigest()[:6], 16)

    return {
        "verified": True,
        "aadhaar_masked": masked,
        "name_as_per_aadhaar": name.upper(),
        "gender": "M" if hash_val % 2 == 0 else "F",
        "address_state": "Maharashtra",
        "mobile_linked": True,
        "kyc_status": "completed",
        "aadhaar_age_years": 25 + (hash_val % 40),
        "verified_at": datetime.now().isoformat(),
        "compliance": {
            "dpdp_consent_obtained": True,
            "data_minimization_applied": True,
            "retention_period_days": 180,
        },
        "source": "UIDAI e-KYC (Simulated)",
    }


@mcp.tool()
async def check_ckyc(pan_number: str) -> dict:
    """
    Check Central KYC Registry (CKYCR) for existing KYC record.
    Returns CKYC number if available, reducing KYC burden for repeat customers.
    """
    pan_number = pan_number.strip().upper()
    hash_val = int(hashlib.md5(pan_number.encode()).hexdigest()[:8], 16)

    has_ckyc = hash_val % 3 > 0  # 67% have existing CKYC

    if has_ckyc:
        ckyc_id = f"6{hash_val % 10000000000000:013d}"
        return {
            "ckyc_found": True,
            "ckyc_number": ckyc_id,
            "kyc_compliant": True,
            "kyc_date": "2023-06-15",
            "kyc_type": "full_kyc",
            "risk_category": "low",
            "source": "CKYCR (Simulated)",
        }

    return {
        "ckyc_found": False,
        "message": "No existing CKYC record. Full KYC required.",
        "source": "CKYCR (Simulated)",
    }


def _get_bank_from_ifsc(ifsc: str) -> str:
    bank_map = {
        "HDFC": "HDFC Bank",
        "SBIN": "State Bank of India",
        "ICIC": "ICICI Bank",
        "UTIB": "Axis Bank",
        "PUNB": "Punjab National Bank",
        "KKBK": "Kotak Mahindra Bank",
    }
    prefix = ifsc[:4].upper()
    return bank_map.get(prefix, f"Bank ({prefix})")


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
    uvicorn.run(app, host="0.0.0.0", port=settings.penny_drop_mcp_port)