from __future__ import annotations
import imaplib
from dataclasses import dataclass
from typing import Optional

from dacribagents.application.ports.email_source import EmailCursor, RawEmail, EmailSource

WORKMAIL_IMAP_HOST = "imap.mail.us-east-1.awsapps.com"
WORKMAIL_IMAP_PORT = 993

@dataclass
class WorkMailImapConfig:
    region: str
    email: str
    password: str
    folder: str = "INBOX"

class WorkMailImapEmailSource(EmailSource):
    def __init__(self, cfg: WorkMailImapConfig) -> None:
        self.cfg = cfg

    def _connect(self) -> imaplib.IMAP4_SSL:
        # WorkMail requires IMAPS; no STARTTLS.
        conn = imaplib.IMAP4_SSL(WORKMAIL_IMAP_HOST, WORKMAIL_IMAP_PORT)
        conn.login(self.cfg.email, self.cfg.password)
        return conn

    def fetch_since(self, cursor: Optional[EmailCursor]) -> tuple[list[RawEmail], EmailCursor]:
        conn = self._connect()
        try:
            typ, _ = conn.select(self.cfg.folder, readonly=True)
            if typ != "OK":
                raise RuntimeError(f"Failed to select folder {self.cfg.folder}")

            typ, data = conn.response("UIDVALIDITY")
            uidvalidity = int(data[0].decode().split()[-1]) if data and data[0] else 0

            last_uid = 0
            if cursor and cursor.uidvalidity == uidvalidity and cursor.folder == self.cfg.folder:
                last_uid = cursor.last_uid

            # Search for new messages by UID
            # UID search uses: UID <command> not SEARCH <command>
            # We use "UID SEARCH {last+1}:*" to get UIDs >= last+1
            start = last_uid + 1
            typ, uids_data = conn.uid("SEARCH", None, f"{start}:*")
            if typ != "OK":
                raise RuntimeError("UID SEARCH failed")

            uids = []
            if uids_data and uids_data[0]:
                uids = [int(x) for x in uids_data[0].split()]

            raws: list[RawEmail] = []
            max_uid = last_uid

            for uid in uids:
                typ, msg_data = conn.uid("FETCH", str(uid), "(RFC822)")
                if typ != "OK" or not msg_data or not msg_data[0]:
                    continue

                # msg_data structure: [(b'UID ... RFC822 {bytes}', b'raw_bytes'), b')']
                rfc822_bytes = msg_data[0][1]
                raws.append(RawEmail(
                    provider="workmail_imap",
                    account=self.cfg.email,
                    folder=self.cfg.folder,
                    uid=uid,
                    rfc822_bytes=rfc822_bytes,
                ))
                if uid > max_uid:
                    max_uid = uid

            new_cursor = EmailCursor(folder=self.cfg.folder, uidvalidity=uidvalidity, last_uid=max_uid)
            return raws, new_cursor
        finally:
            try:
                conn.logout()
            except Exception:
                pass
