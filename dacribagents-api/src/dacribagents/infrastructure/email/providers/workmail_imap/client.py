from __future__ import annotations
import imaplib
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from dacribagents.application.ports.email_source import EmailCursor, RawEmail, EmailSource

WORKMAIL_IMAP_HOST = "imap.mail.us-east-1.awsapps.com"
WORKMAIL_IMAP_PORT = 993
INGESTED_FLAG = "$Ingested"


@dataclass
class WorkMailImapConfig:
    region: str
    email: str
    password: str
    folder: str = "INBOX"

    @property
    def target_folder(self) -> str:
        """Folder based on email local part: agents@woodcreek.me → agents"""
        return self.email.split("@")[0]


class WorkMailImapEmailSource(EmailSource):
    def __init__(self, cfg: WorkMailImapConfig) -> None:
        self.cfg = cfg
        self._conn: Optional[imaplib.IMAP4_SSL] = None

    def _connect(self) -> imaplib.IMAP4_SSL:
        if self._conn is None:
            self._conn = imaplib.IMAP4_SSL(WORKMAIL_IMAP_HOST, WORKMAIL_IMAP_PORT)
            self._conn.login(self.cfg.email, self.cfg.password)
        return self._conn

    def disconnect(self) -> None:
        if self._conn:
            try:
                self._conn.logout()
            except Exception:
                pass
            self._conn = None

    def ensure_folder_exists(self, folder: str) -> None:
        """Create folder if it doesn't exist."""
        conn = self._connect()
        typ, folders = conn.list()
        
        folder_names = []
        if typ == "OK" and folders:
            for f in folders:
                if f:
                    parts = f.decode().split(' "/" ')
                    if len(parts) > 1:
                        folder_names.append(parts[1].strip('"'))

        if folder not in folder_names:
            logger.info(f"Creating folder: {folder}")
            typ, _ = conn.create(folder)
            if typ != "OK":
                logger.warning(f"Could not create folder {folder}")
            else:
                conn.subscribe(folder)

    def fetch_unprocessed(self) -> list[tuple[int, RawEmail]]:
        """Fetch emails NOT flagged as ingested from source folder."""
        conn = self._connect()

        typ, _ = conn.select(self.cfg.folder, readonly=False)
        if typ != "OK":
            raise RuntimeError(f"Failed to select folder {self.cfg.folder}")

        # Search for emails WITHOUT the $Ingested flag
        typ, uids_data = conn.uid("SEARCH", None, f"UNKEYWORD {INGESTED_FLAG}")
        if typ != "OK":
            raise RuntimeError("UID SEARCH failed")

        uids = []
        if uids_data and uids_data[0]:
            uids = [int(x) for x in uids_data[0].split()]

        logger.info(f"Found {len(uids)} unprocessed emails in {self.cfg.folder}")

        results: list[tuple[int, RawEmail]] = []
        for uid in uids:
            typ, msg_data = conn.uid("FETCH", str(uid), "(RFC822)")
            if typ != "OK" or not msg_data or not msg_data[0]:
                continue

            rfc822_bytes = msg_data[0][1]
            results.append((
                uid,
                RawEmail(
                    provider="workmail_imap",
                    account=self.cfg.email,
                    folder=self.cfg.folder,
                    uid=uid,
                    rfc822_bytes=rfc822_bytes,
                )
            ))

        return results

    def mark_processed(self, uid: int) -> bool:
        """Move email to target folder and add $Ingested flag."""
        conn = self._connect()
        target = self.cfg.target_folder

        # Ensure target folder exists
        self.ensure_folder_exists(target)

        # Select source folder
        typ, _ = conn.select(self.cfg.folder, readonly=False)
        if typ != "OK":
            logger.error(f"Failed to select {self.cfg.folder}")
            return False

        # Copy to target folder
        typ, _ = conn.uid("COPY", str(uid), target)
        if typ != "OK":
            logger.error(f"Failed to copy UID {uid} to {target}")
            return False

        # Mark original as deleted
        typ, _ = conn.uid("STORE", str(uid), "+FLAGS", "(\\Deleted)")
        if typ != "OK":
            logger.warning(f"Failed to mark UID {uid} as deleted")

        # Expunge to remove from source
        conn.expunge()

        # Now select target folder and flag the message
        typ, _ = conn.select(target, readonly=False)
        if typ != "OK":
            logger.warning(f"Could not select {target} to add flag")
            return True  # Move succeeded, flag failed - still ok

        # Find the message we just copied (most recent)
        typ, uids_data = conn.uid("SEARCH", None, "ALL")
        if typ == "OK" and uids_data and uids_data[0]:
            # Get the highest UID (most recently added)
            new_uids = [int(x) for x in uids_data[0].split()]
            if new_uids:
                new_uid = max(new_uids)
                conn.uid("STORE", str(new_uid), "+FLAGS", f"({INGESTED_FLAG})")
                logger.debug(f"Flagged UID {new_uid} as ingested in {target}")

        logger.info(f"Moved UID {uid} to {target} and flagged as ingested")
        return True

    def reset_flags(self, folder: Optional[str] = None) -> int:
        """Remove $Ingested flag from all emails in folder (for reprocessing).
        
        Args:
            folder: Folder to reset. Defaults to target_folder (agents, hoa, etc.)
        """
        conn = self._connect()
        target = folder or self.cfg.target_folder

        typ, _ = conn.select(target, readonly=False)
        if typ != "OK":
            logger.warning(f"Could not select {target}")
            return 0

        # Find all emails WITH the flag
        typ, uids_data = conn.uid("SEARCH", None, f"KEYWORD {INGESTED_FLAG}")
        if typ != "OK" or not uids_data or not uids_data[0]:
            return 0

        uids = [int(x) for x in uids_data[0].split()]
        for uid in uids:
            conn.uid("STORE", str(uid), "-FLAGS", f"({INGESTED_FLAG})")

        logger.info(f"Reset {len(uids)} emails in {target} for reprocessing")
        return len(uids)

    # Legacy compatibility
    def fetch_since(self, cursor: Optional[EmailCursor]) -> tuple[list[RawEmail], EmailCursor]:
        results = self.fetch_unprocessed()
        raws = [r[1] for r in results]
        max_uid = max((r[0] for r in results), default=0)
        cursor = EmailCursor(folder=self.cfg.folder, uidvalidity=0, last_uid=max_uid)
        return raws, cursor