from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional

@dataclass(frozen=True)
class EmailCursor:
    # IMAP cursor: UIDVALIDITY + last seen UID for folder
    folder: str
    uidvalidity: int
    last_uid: int

@dataclass(frozen=True)
class RawEmail:
    provider: str
    account: str
    folder: str
    uid: int
    rfc822_bytes: bytes

class EmailSource:
    def fetch_since(self, cursor: Optional[EmailCursor]) -> tuple[list[RawEmail], EmailCursor]:
        raise NotImplementedError
