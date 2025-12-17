from __future__ import annotations
from typing import Protocol
from dacribagents.domain.entities.email_message import EmailMessage

class VectorStore(Protocol):
    def upsert_email(self, msg: EmailMessage, embedding: list[float]) -> None: ...
