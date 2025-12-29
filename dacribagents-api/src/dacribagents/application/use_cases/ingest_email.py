"""Ingest emails into vector store with recipient filtering."""

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
    2. Filter by recipient (To/CC must match configured email)
    3. Process (embed, store attachments)
    4. Move to account folder (agents/, hoa/, etc.)
    5. Flag as $Ingested

    To reprocess: remove flag (stays in folder), move back to INBOX
    """

    def __init__(
        self,
        source: WorkMailImapEmailSource,
        store: VectorStore,
        embedder,
        attachment_store: Optional[AttachmentStore] = None,
        max_attachment_mb: float = 25.0,
        require_exact_recipient: bool = True,
        include_cc: bool = True,
        recipient_alias: Optional[str] = None,
    ) -> None:
        """Initialize the email ingestion use case.
        
        Args:
            source: Email source (WorkMail IMAP)
            store: Vector store for embeddings
            embedder: Text embedder
            attachment_store: Optional S3 store for attachments
            max_attachment_mb: Max attachment size in MB
            require_exact_recipient: If True, only process emails where 
                                     recipient_alias (or source email) is in To/CC
            include_cc: If True, also match emails where recipient 
                        is in CC (not just To)
            recipient_alias: Email address to match for recipient filtering.
                            If not provided, uses source.cfg.email
        """
        self.source = source
        self.store = store
        self.embedder = embedder
        self.attachment_store = attachment_store
        self.max_attachment_bytes = int(max_attachment_mb * 1024 * 1024)
        self.require_exact_recipient = require_exact_recipient
        self.include_cc = include_cc
        
        # The email address to match for recipient filtering
        # Use alias if provided, otherwise fall back to login email
        self.recipient_email = (recipient_alias or source.cfg.email).lower().strip()

    def run(self) -> int:
        """Process all unprocessed emails."""
        emails = self.source.fetch_unprocessed()
        logger.info(f"Found {len(emails)} emails to process (matching: {self.recipient_email})")

        count = 0
        skipped = 0
        for uid, raw in emails:
            try:
                result = self._process_email(raw)
                if result == "processed":
                    self.source.mark_processed(uid)
                    count += 1
                elif result == "skipped_recipient":
                    # Still mark as processed so we don't keep checking it
                    self.source.mark_processed(uid)
                    skipped += 1
                elif result == "skipped_empty":
                    self.source.mark_processed(uid)
                    skipped += 1
            except Exception as e:
                logger.error(f"Failed to process UID {uid}: {e}")

        self.source.disconnect()
        
        if skipped > 0:
            logger.info(f"Skipped {skipped} emails (not addressed to {self.recipient_email})")
        
        return count

    def _is_addressed_to_us(self, to_list: list[str], cc_list: list[str]) -> bool:
        """Check if email is addressed to our recipient email address.
        
        Args:
            to_list: List of To recipients
            cc_list: List of CC recipients
            
        Returns:
            True if recipient_email is in To (or CC if include_cc=True)
        """
        # Normalize for comparison
        def extract_email(addr: str) -> str:
            """Extract email from 'Name <email@domain.com>' format."""
            addr = addr.lower().strip()
            if '<' in addr and '>' in addr:
                start = addr.index('<') + 1
                end = addr.index('>')
                return addr[start:end].strip()
            return addr
        
        # Check To field
        for recipient in to_list:
            if extract_email(recipient) == self.recipient_email:
                return True
        
        # Check CC field if enabled
        if self.include_cc:
            for recipient in cc_list:
                if extract_email(recipient) == self.recipient_email:
                    return True
        
        return False

    def _process_email(self, raw) -> str:
        """Process a single email.
        
        Returns:
            'processed' - Email was ingested
            'skipped_recipient' - Email not addressed to us
            'skipped_empty' - Email has no content
        """
        em = BytesParser(policy=policy.default).parsebytes(raw.rfc822_bytes)
        message_id = em.get("Message-Id", "unknown")
        subject = (em.get("Subject") or "")[:50]

        # Extract To and CC for recipient filtering
        to_list = [x.strip() for x in (em.get_all("To") or [])]
        cc_list = [x.strip() for x in (em.get_all("Cc") or [])]

        # Filter by recipient if enabled
        if self.require_exact_recipient:
            if not self._is_addressed_to_us(to_list, cc_list):
                logger.debug(
                    f"Skipping email not addressed to {self.recipient_email}: "
                    f"To={to_list}, CC={cc_list}, Subject={subject}"
                )
                return "skipped_recipient"

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
            return "skipped_empty"

        # Embed and store
        embedding = self.embedder.embed(msg.text or msg.subject)
        self.store.upsert_email(msg, embedding)
        logger.info(f"Ingested: {msg.subject[:50]}... (to: {self.recipient_email})")

        return "processed"