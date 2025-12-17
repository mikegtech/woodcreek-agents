from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Optional

@dataclass(frozen=True)
class EmailMessage:
    provider: str
    account: str
    folder: str
    message_id: str
    thread_id: Optional[str]
    subject: str
    sender: str
    to: list[str]
    cc: list[str]
    date: datetime
    text: str
    metadata: Mapping[str, str]  # any extra normalized fields
