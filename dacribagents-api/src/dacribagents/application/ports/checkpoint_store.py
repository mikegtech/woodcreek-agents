from __future__ import annotations
from typing import Optional, Protocol
from dacribagents.application.ports.email_source import EmailCursor

class CheckpointStore(Protocol):
    def load(self, account: str, folder: str) -> Optional[EmailCursor]: ...
    def save(self, account: str, cursor: EmailCursor) -> None: ...
