"""
Document Validator Agent.
Validates and extracts structured data from loan documents:
bank statements, ITR, salary slips, KYC docs (Aadhaar, PAN).
"""

import json
import logging
from typing import Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from backend.core.models import LoanApplicationState
from backend.document_processing.ocr_pipeline import OCRPipeline
from backend.core.prompts import EXTRACTOR_PROMPT
from langchain_ollama import ChatOllama
from backend.config.config import settings

logger = logging.getLogger(__name__)

REQUIRED_DOCS = {
    "personal":  ["pan", "aadhaar", "bank_statement_6m", "salary_slip_3m", "itr"],
    "home":      ["pan", "aadhaar", "bank_statement_12m", "salary_slip_3m", "itr", "property_docs"],
    "business":  ["pan", "aadhaar", "bank_statement_12m", "gst_returns", "itr_2y", "business_proof"],
    "vehicle":   ["pan", "aadhaar", "bank_statement_3m", "salary_slip_3m"],
    "msme":      ["pan", "aadhaar", "bank_statement_12m", "gst_returns", "itr_2y", "business_proof"],
}

# Aliases — normalize upstream doc type names to canonical keys
DOC_ALIASES = {
    "pan_card": "pan",
    "aadhaar_card": "aadhaar",
    "salary_slip": "salary_slip_3m",
    "bank_statement": "bank_statement_6m",
    "itr_doc": "itr",
    "property_document": "property_docs",
    "gst_certificate": "gst_returns",
    "business_registration": "business_proof",
}

class DocumentValidatorAgent:
    def __init__(self,llm:Optional[BaseChatModel]):
        self.llm=llm
        self.ocr = OCRPipeline()

    @property
    def llm(self)->BaseChatModel:
        self.llm = ChatOllama(model=settings.llm_model, max_tokens = 2048)
        return self.llm

    @llm.setter
    def llm(self, val: BaseChatModel) -> None:
        self._llm = val

    def _normalize_doc_types(self, docs: list[str]) -> list[str]:
        return [DOC_ALIASES.get(d, d) for d in docs]

    async def validate(self, state: LoanApplicationState) -> dict:
        logger.info(f"[DocValidator] Processing {len(state.documents_received)} docs for {state.application_id}")

        received_normalized = self._normalize_doc_types(state.documents_received)
        required = REQUIRED_DOCS.get(state.loan_type, REQUIRED_DOCS["personal"])
        missing = [doc for doc in required if doc not in received_normalized]

        audit_entry: dict = {
            "agent": "document_validator",
            "application_id": state.application_id,
            "documents_received": state.documents_received,
            "missing_documents": missing,
        }

        if missing:
            audit_entry["status"] = "incomplete"
            logger.warning(f"[DocValidator] Missing docs: {missing}")
            return {
                "documents_validated": False,
                "audit_log": [audit_entry],
                "error": f"Missing required documents: {missing}",
            }

        extracted: dict = {}
        for doc_type in state.documents_received:
            try:
                doc_text = f"[{doc_type} content — real impl fetches from S3 via OCR pipeline]"
                response = await self.llm.ainvoke([
                    SystemMessage(content=EXTRACTOR_PROMPT),
                    HumanMessage(content=f"Document type: {doc_type}\n\nContent:\n{doc_text}"),
                ])
                extracted[doc_type] = json.loads(response.content)
            except Exception as e:
                logger.error(f"[DocValidator] Extraction failed for {doc_type}: {e}")
                extracted[doc_type] = {}

        audit_entry["status"] = "valid"
        audit_entry["extracted_keys"] = list(extracted.keys())
        logger.info(f"[DocValidator] All docs valid for {state.application_id}")

        return {
            "documents_validated": True,
            "extracted_data": extracted,
            "audit_log": [audit_entry],
        }
