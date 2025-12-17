from __future__ import annotations

from email import policy
from email.parser import BytesParser
from typing import Optional

from loguru import logger

from dacribagents.application.ports.vector_store import VectorStore
from dacribagents.infrastructure.attachments.store import AttachmentStore
from dacribagents.infrastructure.email.rfc822 import extract_attachments
from dacribagents.infrastructure.email.providers.workmail_imap.mapper import rfc822_to_email_message
from dacribagents.infrastructure.email.providers.workmail_imap.client import WorkMailImapEmailSource
from dacribagents.domain.entities.attachment import AttachmentRef


class IngestEmailUseCase:
    """Ingest emails with folder + flag state management.

    Flow:
    1. Fetch unflagged emails from source folder (INBOX)
    2. Process (embed, store attachments)
    3. Move to account folder (agents/, hoa/, etc.)
    4. Flag as $Ingested

    To reprocess: remove flag (stays in folder), move back to INBOX
    """

    def __init__(
        self,
        source: WorkMailImapEmailSource,
        store: VectorStore,
        embedder,
        attachment_store: Optional[AttachmentStore] = None,
        max_attachment_mb: float = 25.0,
    ) -> None:
        self.source = source
        self.store = store
        self.embedder = embedder
        self.attachment_store = attachment_store
        self.max_attachment_bytes = int(max_attachment_mb * 1024 * 1024)

    def run(self) -> int:
        """Process all unprocessed emails."""
        emails = self.source.fetch_unprocessed()
        logger.info(f"Found {len(emails)} emails to process")

        count = 0
        for uid, raw in emails:
            try:
                success = self._process_email(raw)
                if success:
                    self.source.mark_processed(uid)
                    count += 1
            except Exception as e:
                logger.error(f"Failed to process UID {uid}: {e}")

        self.source.disconnect()
        return count

    def _process_email(self, raw) -> bool:
        """Process a single email. Returns True on success."""
        em = BytesParser(policy=policy.default).parsebytes(raw.rfc822_bytes)
        message_id = em.get("Message-Id", "unknown")

        # Extract and store attachments
        attachment_refs: list[AttachmentRef] = []
        if self.attachment_store:
            extracted = extract_attachments(em)
            for att in extracted:
                if len(att.content_bytes) > self.max_attachment_bytes:
                    logger.warning(
                        f"Skipping attachment {att.filename} ({len(att.content_bytes)} bytes) - exceeds max"
                    )
                    continue

                try:
                    ref = self.attachment_store.put(
                        account=raw.account,
                        message_id=message_id,
                        filename=att.filename,
                        content_type=att.content_type,
                        data=att.content_bytes,
                    )
                    attachment_refs.append(ref)
                    logger.info(f"Stored attachment: {att.filename} ({len(att.content_bytes)} bytes)")
                except Exception as e:
                    logger.error(f"Failed to store attachment {att.filename}: {e}")

        # Build email message
        msg = rfc822_to_email_message(
            raw.provider,
            raw.account,
            raw.folder,
            raw.rfc822_bytes,
            attachments=attachment_refs,
        )

        if not msg.text and not msg.subject:
            logger.debug(f"Skipping empty email {message_id}")
            return True

        # Embed and store
        embedding = self.embedder.embed(msg.text or msg.subject)
        self.store.upsert_email(msg, embedding)
        logger.info(f"Ingested: {msg.subject[:50]}...")

        return True