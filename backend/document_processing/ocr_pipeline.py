"""
OCR Pipeline for loan documents.
Uses PyMuPDF for PDF text extraction + minicpm-v:latest for scanned/image PDFs.
Handles: bank statements, salary slips, ITR, KYC documents.
"""

import logging
import base64
from pathlib import Path
import fitz  # PyMuPDF
import pymupdf4llm
import asyncio
import os
logger = logging.getLogger(__name__)
import ollama
from backend.core.prompts import OCR_MODEL_PROMPT
from backend.config.config import settings

class OCR_Pipeline:
    """
    Mulit strategy document text extractor
    """

    async def extract_text(self,file_path:str)->str:
        """
        Extract text from a document using the best available strategy.
        Strategy order: PyMuPDF native →  OCR (for scanned)
        """
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"File not found: {file_path}")
            return ""

        if path.suffix.lower() == ".pdf":
            return await self._extract_pdf(str(path))
        else:
            return await self._extract_image(str(path))

    async def _extract_pdf(self,file_path)->str:
        try:
            markdown_text = pymupdf4llm.to_markdown(file_path)
            text_parts = []
            if len(markdown_text)<50:
                ocr_text = await self._extract_ocr(file_path)
        except Exception as e:
            logger.error(f"PDF extraction failed for {file_path}: {e}")
            return ""

    async def _extract_ocr(path:str):
        markdown = ""
        doc = fitz.open(path)
        for i, page in enumerate(doc):
            
            img = f"/tmp/page_{i}.png"
            page.get_pixmap(dpi=150).save(img)

            MODEL = settings.ocr_model
            response = ollama.chat(
                model=MODEL,
                messages=[{ "role": "user",
                            "content": OCR_MODEL_PROMPT,
                            "images": [img]
                    }])

            markdown += f"\n\n## Page {i+1}\n\n"
            markdown += response["message"]["content"]
            os.remove(img)
        return markdown

    async def _extract_image(path:str)-> str:
        try:
            MODEL = settings.ocr_model
            response = ollama.chat(
                model=MODEL,
                messages=[{ "role": "user",
                            "content": OCR_MODEL_PROMPT,
                            "images": [path]
                    }])

            markdown += response["message"]["content"]
            return markdown
        except Exception as e:
            logger.error(f"Image OCR failed: {e}")
            return ""