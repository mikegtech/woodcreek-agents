from __future__ import annotations
from typing import Protocol
from dacribagents.domain.entities.attachment import AttachmentRef

class AttachmentStore(Protocol):
    def put(self, *, account: str, message_id: str, filename: str, content_type: str, data: bytes) -> AttachmentRef: ...
