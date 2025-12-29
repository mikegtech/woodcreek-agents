"""Email ingestion worker - polls multiple mailboxes at configurable intervals."""

from __future__ import annotations

import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime

from loguru import logger

from dacribagents.application.use_cases.ingest_email import IngestEmailUseCase
from dacribagents.infrastructure.email.providers.workmail_imap.client import (
    WorkMailImapConfig,
    WorkMailImapEmailSource,
)
from dacribagents.infrastructure import get_milvus_client
from dacribagents.infrastructure.stores import MilvusEmailStore
from dacribagents.infrastructure.embeddings import EmbeddingsFactory
from dacribagents.infrastructure.attachments.s3_store import s3_store_from_env


@dataclass
class MailboxConfig:
    """Configuration for a single mailbox."""
    name: str
    email: str
    password: str
    folder: str = "INBOX"
    email_alias: str | None = None  # Email to match for recipient filtering


@dataclass
class WorkerStats:
    """Track worker statistics."""
    total_processed: int = 0
    total_errors: int = 0
    last_poll: datetime | None = None
    polls_completed: int = 0
    by_mailbox: dict[str, int] = None

    def __post_init__(self):
        if self.by_mailbox is None:
            self.by_mailbox = {}


class EmailWorker:
    """
    Multi-mailbox email ingestion worker.
    
    Polls configured mailboxes at regular intervals and ingests
    new emails into Milvus for RAG retrieval.
    """

    def __init__(
        self,
        mailboxes: list[MailboxConfig],
        poll_interval_minutes: int = 5,
        region: str = "us-east-1",
    ):
        self.mailboxes = mailboxes
        self.poll_interval = poll_interval_minutes * 60  # Convert to seconds
        self.region = region
        self.running = False
        self.stats = WorkerStats()
        
        # Shared infrastructure
        self._milvus = None
        self._store = None
        self._embedder = None
        self._attachment_store = None

    def _init_infrastructure(self) -> None:
        """Initialize shared infrastructure components."""
        logger.info("Initializing infrastructure...")
        
        self._milvus = get_milvus_client()
        self._milvus.connect()
        
        self._store = MilvusEmailStore(self._milvus)
        self._embedder = EmbeddingsFactory.from_env()
        
        # Optional attachment storage
        if os.getenv("ATTACHMENTS_ENABLED", "false").lower() == "true":
            self._attachment_store = s3_store_from_env()
            logger.info("Attachment storage enabled (S3)")
        
        logger.info("Infrastructure initialized")

    def _process_mailbox(self, mailbox: MailboxConfig) -> int:
        """Process a single mailbox. Returns count of processed emails."""
        logger.info(f"Processing mailbox: {mailbox.name} ({mailbox.email})")
        if mailbox.email_alias:
            logger.info(f"  Recipient filter alias: {mailbox.email_alias}")
        
        cfg = WorkMailImapConfig(
            region=self.region,
            email=mailbox.email,
            password=mailbox.password,
            folder=mailbox.folder,
        )
        
        source = WorkMailImapEmailSource(cfg)
        
        max_attachment_mb = float(os.getenv("ATTACHMENTS_MAX_MB", "25"))
        
        # Recipient filtering options
        require_exact_recipient = os.getenv("EMAIL_REQUIRE_EXACT_RECIPIENT", "true").lower() == "true"
        include_cc = os.getenv("EMAIL_INCLUDE_CC", "true").lower() == "true"
        
        uc = IngestEmailUseCase(
            source=source,
            store=self._store,
            embedder=self._embedder,
            attachment_store=self._attachment_store,
            max_attachment_mb=max_attachment_mb,
            require_exact_recipient=require_exact_recipient,
            include_cc=include_cc,
            recipient_alias=mailbox.email_alias,
        )
        
        try:
            count = uc.run()
            logger.info(f"Mailbox {mailbox.name}: ingested {count} emails")
            return count
        except Exception as e:
            logger.error(f"Mailbox {mailbox.name} failed: {e}")
            raise
        finally:
            source.disconnect()

    def _poll_all_mailboxes(self) -> None:
        """Poll all configured mailboxes once."""
        self.stats.last_poll = datetime.now()
        logger.info(f"Starting poll cycle #{self.stats.polls_completed + 1}")
        
        for mailbox in self.mailboxes:
            try:
                count = self._process_mailbox(mailbox)
                self.stats.total_processed += count
                self.stats.by_mailbox[mailbox.name] = (
                    self.stats.by_mailbox.get(mailbox.name, 0) + count
                )
            except Exception as e:
                self.stats.total_errors += 1
                logger.error(f"Error processing {mailbox.name}: {e}")
        
        self.stats.polls_completed += 1
        self._log_stats()

    def _log_stats(self) -> None:
        """Log current worker statistics."""
        logger.info(
            f"Worker stats: "
            f"polls={self.stats.polls_completed}, "
            f"processed={self.stats.total_processed}, "
            f"errors={self.stats.total_errors}, "
            f"by_mailbox={self.stats.by_mailbox}"
        )

    def _handle_shutdown(self, signum, frame) -> None:
        """Handle graceful shutdown."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def run(self) -> int:
        """Run the worker loop."""
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        
        logger.info(f"Email worker starting with {len(self.mailboxes)} mailbox(es)")
        logger.info(f"Poll interval: {self.poll_interval // 60} minutes")
        for mb in self.mailboxes:
            logger.info(f"  - {mb.name}: {mb.email}")
        
        # Initialize infrastructure once
        try:
            self._init_infrastructure()
        except Exception as e:
            logger.error(f"Failed to initialize infrastructure: {e}")
            return 1
        
        self.running = True
        
        # Initial poll
        self._poll_all_mailboxes()
        
        # Main loop
        while self.running:
            logger.debug(f"Sleeping for {self.poll_interval} seconds...")
            
            # Sleep in small increments to respond to signals quickly
            sleep_remaining = self.poll_interval
            while sleep_remaining > 0 and self.running:
                sleep_time = min(sleep_remaining, 10)
                time.sleep(sleep_time)
                sleep_remaining -= sleep_time
            
            if self.running:
                self._poll_all_mailboxes()
        
        logger.info("Worker shutdown complete")
        self._log_stats()
        return 0


def get_mailboxes_from_env() -> list[MailboxConfig]:
    """
    Load mailbox configurations from environment variables.
    
    Supports two formats:
    
    1. Single mailbox (legacy):
       WORKMAIL_EMAIL=agents@woodcreek.me
       WORKMAIL_PASSWORD=xxx
       WORKMAIL_EMAIL_ALIAS=agents@woodcreek.me  # Optional: address to match
    
    2. Multiple mailboxes:
       WORKMAIL_MAILBOXES=agents,solar,hoa
       WORKMAIL_AGENTS_EMAIL=agents@woodcreek.me
       WORKMAIL_AGENTS_PASSWORD=xxx
       WORKMAIL_AGENTS_EMAIL_ALIAS=agents@woodcreek.me  # Optional: address to match
       WORKMAIL_SOLAR_EMAIL=solar@woodcreek.me
       WORKMAIL_SOLAR_PASSWORD=xxx
       WORKMAIL_SOLAR_EMAIL_ALIAS=solar@woodcreek.me  # Optional
       ...
    """
    mailboxes = []
    
    # Check for multi-mailbox config
    mailbox_names = os.getenv("WORKMAIL_MAILBOXES", "").strip()
    
    if mailbox_names:
        # Multi-mailbox mode
        for name in mailbox_names.split(","):
            name = name.strip().upper()
            email = os.getenv(f"WORKMAIL_{name}_EMAIL")
            password = os.getenv(f"WORKMAIL_{name}_PASSWORD")
            folder = os.getenv(f"WORKMAIL_{name}_FOLDER", "INBOX")
            email_alias = os.getenv(f"WORKMAIL_{name}_EMAIL_ALIAS")
            
            if email and password:
                mailboxes.append(MailboxConfig(
                    name=name.lower(),
                    email=email,
                    password=password,
                    folder=folder,
                    email_alias=email_alias,
                ))
                alias_info = f" (alias: {email_alias})" if email_alias else ""
                logger.info(f"Configured mailbox: {name.lower()} ({email}){alias_info}")
            else:
                logger.warning(f"Mailbox {name} missing email or password, skipping")
    
    else:
        # Single mailbox mode (legacy)
        email = os.getenv("WORKMAIL_EMAIL")
        password = os.getenv("WORKMAIL_PASSWORD")
        folder = os.getenv("WORKMAIL_FOLDER", "INBOX")
        email_alias = os.getenv("WORKMAIL_EMAIL_ALIAS")
        
        if email and password:
            name = email.split("@")[0]
            mailboxes.append(MailboxConfig(
                name=name,
                email=email,
                password=password,
                folder=folder,
                email_alias=email_alias,
            ))
            alias_info = f" (alias: {email_alias})" if email_alias else ""
            logger.info(f"Configured single mailbox: {name} ({email}){alias_info}")
    
    return mailboxes


def main() -> int:
    """Entry point for the email worker."""
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=os.getenv("LOG_LEVEL", "INFO"),
    )
    
    logger.info("=" * 60)
    logger.info("Woodcreek Email Worker")
    logger.info("=" * 60)
    
    # Get configuration
    poll_minutes = int(os.getenv("EMAIL_POLL_MINUTES", "5"))
    region = os.getenv("WORKMAIL_REGION", "us-east-1")
    
    mailboxes = get_mailboxes_from_env()
    
    if not mailboxes:
        logger.error("No mailboxes configured! Set WORKMAIL_EMAIL/PASSWORD or WORKMAIL_MAILBOXES")
        return 1
    
    # Create and run worker
    worker = EmailWorker(
        mailboxes=mailboxes,
        poll_interval_minutes=poll_minutes,
        region=region,
    )
    
    return worker.run()


if __name__ == "__main__":
    raise SystemExit(main())