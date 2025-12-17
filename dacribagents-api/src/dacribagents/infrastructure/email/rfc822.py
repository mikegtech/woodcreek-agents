from __future__ import annotations
from dataclasses import dataclass
from email.message import Message

@dataclass(frozen=True)
class ExtractedAttachment:
    filename: str
    content_type: str
    content_bytes: bytes

def extract_attachments(em: Message) -> list[ExtractedAttachment]:
    out: list[ExtractedAttachment] = []
    for part in em.walk():
        if part.is_multipart():
            continue

        filename = part.get_filename()
        disp = (part.get("Content-Disposition") or "").lower()

        # capture explicit attachments + common inline-with-filename cases
        if not filename and "attachment" not in disp:
            continue

        payload = part.get_payload(decode=True) or b""
        if not payload:
            continue

        out.append(
            ExtractedAttachment(
                filename=filename or "attachment.bin",
                content_type=part.get_content_type(),
                content_bytes=payload,
            )
        )
    return out
