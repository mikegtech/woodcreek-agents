from __future__ import annotations
from io import BytesIO
from pypdf import PdfReader

def pdf_to_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            parts.append(t.strip())
    return "\n\n".join(parts)
