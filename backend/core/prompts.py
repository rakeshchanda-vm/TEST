OCR_MODEL_PROMPT = """
Extract all text from this document image and in Return Markdown.
Preserve headings, tables, and bullet points. Only give extracted text. 
NO EXPLANATION Required
"""

EXTRACTOR_PROMPT = """You are a financial document extraction agent.
Extract structured data from the provided document text.

For BANK STATEMENT extract: average monthly balance, monthly inflow/outflow, bounced cheques, EMI debits.
For SALARY SLIP extract: gross salary, net salary, employer name, employment type.
For ITR extract: annual taxable income, tax paid, assessment year.

Return ONLY valid JSON. No explanations."""