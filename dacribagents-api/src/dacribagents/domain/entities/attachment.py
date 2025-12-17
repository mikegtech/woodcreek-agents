from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class AttachmentRef:
    filename: str
    content_type: str
    size_bytes: int

    # Where the bytes live
    bucket: str
    key: str
    etag: Optional[str] = None
