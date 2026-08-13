"""
BFSI Credit Intelligence — Application Configuration.
"""
import os
from functools import lru_cache
from pydantic_settings import BaseSettings,SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    # LLM Configutations
    provider:str = "ollama"
    llm_model:str = "llama3.2:3b"
    ocr_model:str = "minicpm-v:latest"
            
    # MCP Server ports
    bureau_mcp_port: int = 9001
    bank_txn_mcp_port: int = 9002
    gst_mcp_port: int = 9003
    rbi_compliance_mcp_port: int = 9004
    penny_drop_mcp_port: int = 9005

    # Security / PII
    encryption_key: str = os.getenv("ENCRYPTION_KEY", "default_dev_key_change_in_prod!!")
    enable_pii_masking: bool = os.getenv("ENABLE_PII_MASKING", "true").lower() == "true"
    dpdp_compliance_mode: str = os.getenv("DPDP_COMPLIANCE_MODE", "strict")
    audit_log_bucket: str = os.getenv("AUDIT_LOG_BUCKET", "bfsi-audit-logs")

    # MLflow
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow_experiment_name: str = "bfsi-credit-underwriting"


    # Underwriting
    max_loan_amount: float = 100_000_000
    min_loan_amount: float = 10_000
    fraud_risk_auto_reject_threshold: float = 0.85
    max_dti_ratio: float = 0.50

    # App
    environment: str = "development"
    debug: bool = False

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()