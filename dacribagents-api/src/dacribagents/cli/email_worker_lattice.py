"""
Woodcreek Email Worker → Lattice Publisher

This worker:
1. Polls WorkMail IMAP for multiple aliases (agents@, solar@, hoa@)
2. Filters emails by To/CC recipient matching
3. Publishes raw RFC822 to Lattice Kafka (lattice.mail.raw.v1)
4. Marks emails as processed via IMAP flags

Lattice then handles: parsing → chunking → embedding → Milvus upsert
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import signal
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from typing import Optional

from confluent_kafka import Producer, KafkaError
from loguru import logger

from dacribagents.infrastructure.email.providers.workmail_imap.client import (
    WorkMailImapConfig,
    WorkMailImapEmailSource,
)


# =============================================================================
# Lattice Envelope
# =============================================================================

@dataclass
class LatticeEnvelope:
    """Lattice Kafka message envelope."""
    
    message_id: str
    schema_version: str = "v1"
    domain: str = "mail"
    stage: str = "raw"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    tenant_id: str = ""
    account_id: str = ""
    alias: Optional[str] = None
    
    source_service: str = "woodcreek-email-worker"
    source_version: str = "1.0.0"
    
    data_classification: str = "internal"
    pii: bool = True
    
    trace_id: Optional[str] = None
    
    payload: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "schema_version": self.schema_version,
            "domain": self.domain,
            "stage": self.stage,
            "created_at": self.created_at,
            "tenant_id": self.tenant_id,
            "account_id": self.account_id,
            "alias": self.alias,
            "source": {
                "service": self.source_service,
                "version": self.source_version,
            },
            "data_classification": self.data_classification,
            "pii": self.pii,
            "trace_id": self.trace_id or str(uuid.uuid4()),
            "payload": self.payload,
        }


# =============================================================================
# Kafka Publisher
# =============================================================================

class LatticeKafkaPublisher:
    """Publishes emails to Lattice via Kafka."""
    
    def __init__(
        self,
        brokers: str,
        username: str,
        password: str,
        topic: str = "lattice.mail.raw.v1",
        tenant_id: str = "woodcreek",
    ):
        self.topic = topic
        self.tenant_id = tenant_id
        
        config = {
            "bootstrap.servers": brokers,
            "security.protocol": "SASL_SSL",
            "sasl.mechanisms": "PLAIN",
            "sasl.username": username,
            "sasl.password": password,
            # Reliability settings
            "acks": "all",
            "retries": 3,
            "retry.backoff.ms": 1000,
            # Batching for efficiency
            "batch.size": 65536,
            "linger.ms": 100,
        }
        
        self.producer = Producer(config)
        self._delivery_errors: list[str] = []
        
        logger.info(f"Kafka publisher initialized for topic {topic}")
    
    def _delivery_callback(self, err, msg):
        """Handle delivery confirmation."""
        if err:
            logger.error(f"Delivery failed: {err}")
            self._delivery_errors.append(str(err))
        else:
            logger.debug(f"Delivered to {msg.topic()}[{msg.partition()}] @ {msg.offset()}")
    
    def publish(
        self,
        rfc822_bytes: bytes,
        account_id: str,
        alias: str,
        provider_message_id: str,
        folder: str = "INBOX",
        received_at: Optional[datetime] = None,
    ) -> str:
        """
        Publish a raw email to Lattice.
        
        Returns:
            message_id of the envelope
        """
        # Generate content hash for deduplication
        content_hash = hashlib.sha256(rfc822_bytes).hexdigest()
        
        # Build payload matching lattice.mail.raw.v1 schema
        payload = {
            "provider": "imap",
            "provider_message_id": provider_message_id,
            "folder": folder,
            "received_at": (received_at or datetime.now(timezone.utc)).isoformat(),
            "raw_rfc822_base64": base64.b64encode(rfc822_bytes).decode("utf-8"),
            "size_bytes": len(rfc822_bytes),
            "content_hash": content_hash,
        }
        
        # Create envelope
        envelope = LatticeEnvelope(
            message_id=str(uuid.uuid4()),
            tenant_id=self.tenant_id,
            account_id=account_id,
            alias=alias,
            payload=payload,
        )
        
        # Kafka key for partitioning (same email always goes to same partition)
        key = f"{self.tenant_id}:{account_id}:{provider_message_id}".encode()
        
        # Publish
        self.producer.produce(
            topic=self.topic,
            key=key,
            value=json.dumps(envelope.to_dict()).encode("utf-8"),
            callback=self._delivery_callback,
        )
        
        return envelope.message_id
    
    def flush(self, timeout: float = 30.0) -> int:
        """Flush pending messages. Returns number of messages still in queue."""
        return self.producer.flush(timeout)
    
    def get_errors(self) -> list[str]:
        """Get delivery errors since last check."""
        errors = self._delivery_errors.copy()
        self._delivery_errors.clear()
        return errors


# =============================================================================
# Mailbox Configuration
# =============================================================================

@dataclass
class MailboxConfig:
    """Configuration for a single mailbox."""
    name: str
    email: str
    password: str
    email_alias: str  # For To/CC filtering
    account_id: str   # Lattice account ID
    folder: str = "INBOX"
    target_folder: Optional[str] = None  # Move processed emails here
    
    def __post_init__(self):
        if not self.target_folder:
            self.target_folder = self.name


# =============================================================================
# Email Worker
# =============================================================================

class EmailWorkerLattice:
    """
    Email worker that publishes to Lattice Kafka.
    
    Polls IMAP mailboxes, filters by recipient, publishes to Kafka.
    """
    
    def __init__(
        self,
        mailboxes: list[MailboxConfig],
        publisher: LatticeKafkaPublisher,
        poll_interval_minutes: int = 5,
        imap_host: str = "imap.mail.us-east-1.awsapps.com",
        imap_port: int = 993,
    ):
        self.mailboxes = mailboxes
        self.publisher = publisher
        self.poll_interval = poll_interval_minutes * 60
        self.imap_host = imap_host
        self.imap_port = imap_port
        
        self._running = False
        self._poll_count = 0
        
        # Stats
        self.stats = {
            "total_published": 0,
            "total_skipped": 0,
            "total_errors": 0,
            "by_mailbox": {},
        }
    
    def _setup_signals(self):
        """Setup graceful shutdown handlers."""
        def handle_signal(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self._running = False
        
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)
    
    def _extract_recipient_filter(self, alias: str) -> str:
        """Extract the email address to filter on."""
        # If alias is just the local part, append domain
        if "@" not in alias:
            return f"{alias}@woodcreek.me"
        return alias.lower()
    
    def _check_recipient_match(self, rfc822_bytes: bytes, alias: str) -> bool:
        """Check if email is addressed to the alias."""
        try:
            em = BytesParser(policy=policy.default).parsebytes(rfc822_bytes)
            
            # Get To and CC addresses
            to_addrs = em.get_all("To") or []
            cc_addrs = em.get_all("Cc") or []
            
            # Extract email addresses
            def extract_email(addr_str: str) -> str:
                if "<" in addr_str and ">" in addr_str:
                    return addr_str.split("<")[1].split(">")[0].lower()
                return addr_str.strip().lower()
            
            all_recipients = []
            for addr in to_addrs + cc_addrs:
                all_recipients.append(extract_email(str(addr)))
            
            filter_addr = self._extract_recipient_filter(alias)
            return filter_addr in all_recipients
            
        except Exception as e:
            logger.warning(f"Failed to parse email for recipient check: {e}")
            return False
    
    def _process_mailbox(self, config: MailboxConfig) -> tuple[int, int, int]:
        """
        Process a single mailbox.
        
        Returns: (published, skipped, errors)
        """
        logger.info(f"Processing mailbox: {config.name} ({config.email})")
        logger.info(f"  Recipient filter: {config.email_alias}")
        
        # Create IMAP source
        # Note: WorkMailImapConfig.target_folder is a computed property from email
        imap_config = WorkMailImapConfig(
            region="us-east-1",
            email=config.email,
            password=config.password,
            folder=config.folder,
        )
        
        source = WorkMailImapEmailSource(imap_config)
        
        try:
            # Fetch unprocessed emails
            emails = source.fetch_unprocessed()
            logger.info(f"  Found {len(emails)} unprocessed emails")
            
            published = 0
            skipped = 0
            errors = 0
            
            for uid, raw in emails:
                try:
                    # Check recipient filter
                    if not self._check_recipient_match(raw.rfc822_bytes, config.email_alias):
                        logger.debug(f"  Skipping UID {uid} - not addressed to {config.email_alias}")
                        skipped += 1
                        # DO NOT mark as processed - let the correct mailbox handler get it
                        continue
                    
                    # Publish to Lattice
                    message_id = self.publisher.publish(
                        rfc822_bytes=raw.rfc822_bytes,
                        account_id=config.account_id,
                        alias=config.name,
                        provider_message_id=f"{config.email}:{uid}",
                        folder=config.folder,
                    )
                    
                    logger.info(f"  Published UID {uid} as {message_id}")
                    
                    # Mark as processed in IMAP
                    source.mark_processed(uid)
                    published += 1
                    
                except Exception as e:
                    logger.error(f"  Failed to process UID {uid}: {e}")
                    errors += 1
            
            # Flush Kafka to ensure delivery
            remaining = self.publisher.flush(timeout=10.0)
            if remaining > 0:
                logger.warning(f"  {remaining} messages still pending after flush")
            
            # Check for delivery errors
            kafka_errors = self.publisher.get_errors()
            if kafka_errors:
                logger.error(f"  Kafka delivery errors: {kafka_errors}")
                errors += len(kafka_errors)
            
            return published, skipped, errors
            
        finally:
            source.disconnect()
    
    def run_once(self) -> dict:
        """Run a single poll cycle across all mailboxes."""
        self._poll_count += 1
        logger.info(f"Starting poll cycle #{self._poll_count}")
        
        cycle_stats = {
            "published": 0,
            "skipped": 0,
            "errors": 0,
        }
        
        for config in self.mailboxes:
            try:
                published, skipped, errors = self._process_mailbox(config)
                
                cycle_stats["published"] += published
                cycle_stats["skipped"] += skipped
                cycle_stats["errors"] += errors
                
                # Update per-mailbox stats
                if config.name not in self.stats["by_mailbox"]:
                    self.stats["by_mailbox"][config.name] = 0
                self.stats["by_mailbox"][config.name] += published
                
            except Exception as e:
                logger.error(f"Failed to process mailbox {config.name}: {e}")
                cycle_stats["errors"] += 1
        
        # Update totals
        self.stats["total_published"] += cycle_stats["published"]
        self.stats["total_skipped"] += cycle_stats["skipped"]
        self.stats["total_errors"] += cycle_stats["errors"]
        
        logger.info(
            f"Poll cycle #{self._poll_count} complete: "
            f"published={cycle_stats['published']}, "
            f"skipped={cycle_stats['skipped']}, "
            f"errors={cycle_stats['errors']}"
        )
        logger.info(f"Cumulative stats: {self.stats}")
        
        return cycle_stats
    
    def run_forever(self):
        """Run continuous polling loop."""
        self._setup_signals()
        self._running = True
        
        logger.info(f"Starting email worker (poll interval: {self.poll_interval}s)")
        logger.info(f"Mailboxes: {[m.name for m in self.mailboxes]}")
        
        while self._running:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"Poll cycle failed: {e}")
            
            if self._running:
                logger.info(f"Sleeping {self.poll_interval}s until next poll...")
                # Sleep in small increments to respond to signals
                for _ in range(self.poll_interval):
                    if not self._running:
                        break
                    time.sleep(1)
        
        logger.info("Email worker stopped")
        logger.info(f"Final stats: {self.stats}")


# =============================================================================
# Factory Functions
# =============================================================================

def create_publisher_from_env() -> "LatticeKafkaPublisher":
    """Create Kafka publisher from environment variables.
    
    Uses the lattice publisher module which supports claim check pattern.
    """
    from dacribagents.infrastructure.lattice.publisher import get_lattice_publisher
    return get_lattice_publisher()


def create_mailboxes_from_env() -> list[MailboxConfig]:
    """Create mailbox configs from environment variables."""
    mailbox_names = os.getenv("WORKMAIL_MAILBOXES", "agents").split(",")
    mailboxes = []
    
    for name in mailbox_names:
        name = name.strip()
        prefix = f"WORKMAIL_{name.upper()}_"
        
        email = os.environ.get(f"{prefix}EMAIL")
        password = os.environ.get(f"{prefix}PASSWORD")
        
        if not email or not password:
            logger.warning(f"Skipping mailbox {name}: missing EMAIL or PASSWORD")
            continue
        
        mailboxes.append(MailboxConfig(
            name=name,
            email=email,
            password=password,
            email_alias=os.getenv(f"{prefix}EMAIL_ALIAS", email),
            account_id=os.getenv(f"{prefix}ACCOUNT_ID", f"workmail-{name}"),
            folder=os.getenv(f"{prefix}FOLDER", "INBOX"),
            target_folder=os.getenv(f"{prefix}TARGET_FOLDER", name),
        ))
    
    return mailboxes


# =============================================================================
# CLI Entry Points
# =============================================================================

def main():
    """Main entry point for continuous polling."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Woodcreek Email Worker → Lattice")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--interval", type=int, default=None, help="Poll interval in minutes")
    args = parser.parse_args()
    
    # Create components
    publisher = create_publisher_from_env()
    mailboxes = create_mailboxes_from_env()
    
    if not mailboxes:
        logger.error("No mailboxes configured!")
        return 1
    
    poll_interval = args.interval or int(os.getenv("EMAIL_POLL_MINUTES", "5"))
    
    worker = EmailWorkerLattice(
        mailboxes=mailboxes,
        publisher=publisher,
        poll_interval_minutes=poll_interval,
    )
    
    if args.once:
        worker.run_once()
    else:
        worker.run_forever()
    
    return 0


def backfill():
    """Backfill entry point for historical emails."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Backfill historical emails to Lattice")
    parser.add_argument("--mailbox", required=True, help="Mailbox name (agents, solar, hoa)")
    parser.add_argument("--folder", default="INBOX", help="IMAP folder to backfill")
    parser.add_argument("--limit", type=int, default=100, help="Max emails to backfill")
    parser.add_argument("--since-days", type=int, default=None, help="Only emails from last N days")
    args = parser.parse_args()
    
    # Get mailbox config
    mailboxes = create_mailboxes_from_env()
    config = next((m for m in mailboxes if m.name == args.mailbox), None)
    
    if not config:
        logger.error(f"Mailbox '{args.mailbox}' not found. Available: {[m.name for m in mailboxes]}")
        return 1
    
    publisher = create_publisher_from_env()
    
    logger.info(f"Backfilling {args.mailbox} from folder {args.folder}")
    
    # Create IMAP config for backfill (different from incremental)
    imap_config = WorkMailImapConfig(
        region="us-east-1",
        email=config.email,
        password=config.password,
        folder=args.folder,
    )
    
    source = WorkMailImapEmailSource(imap_config)
    
    try:
        # For backfill, we need to fetch ALL emails (not just unprocessed)
        # This requires accessing the underlying IMAP connection
        import imaplib
        from datetime import timedelta
        
        # Connect and get the underlying IMAP connection
        conn = source._connect()  # Use _connect() method to get connection
        conn.select(args.folder, readonly=True)
        
        # Build search criteria
        if args.since_days:
            since_date = (datetime.now() - timedelta(days=args.since_days)).strftime("%d-%b-%Y")
            search_criteria = f'(SINCE {since_date})'
        else:
            search_criteria = "ALL"
        
        typ, uids_data = conn.uid("SEARCH", None, search_criteria)
        if typ != "OK":
            logger.error("IMAP search failed")
            return 1
        
        all_uids = uids_data[0].split()
        uids_to_process = all_uids[-args.limit:]  # Take last N
        
        logger.info(f"Found {len(all_uids)} emails, processing {len(uids_to_process)}")
        
        published = 0
        skipped = 0
        
        for uid in uids_to_process:
            uid_str = uid.decode() if isinstance(uid, bytes) else uid
            
            typ, msg_data = conn.uid("FETCH", uid, "(RFC822)")
            if typ != "OK" or not msg_data or not msg_data[0]:
                continue
            
            rfc822_bytes = msg_data[0][1]
            
            # Check recipient filter
            if not _check_recipient_match_static(rfc822_bytes, config.email_alias):
                logger.debug(f"Skipping UID {uid_str} - not addressed to {config.email_alias}")
                skipped += 1
                continue
            
            try:
                message_id = publisher.publish(
                    rfc822_bytes=rfc822_bytes,
                    account_id=config.account_id,
                    alias=config.name,
                    provider_message_id=f"{config.email}:{uid_str}",
                    folder=args.folder,
                )
                logger.info(f"Published UID {uid_str} as {message_id}")
                published += 1
            except Exception as e:
                logger.error(f"Failed to publish UID {uid_str}: {e}")
        
        publisher.flush(timeout=30.0)
        
        logger.info(f"Backfill complete: published={published}, skipped={skipped}")
        
    finally:
        source.disconnect()
    
    return 0


def _check_recipient_match_static(rfc822_bytes: bytes, alias: str) -> bool:
    """Static version of recipient check for backfill."""
    try:
        em = BytesParser(policy=policy.default).parsebytes(rfc822_bytes)
        
        to_addrs = em.get_all("To") or []
        cc_addrs = em.get_all("Cc") or []
        
        def extract_email(addr_str: str) -> str:
            if "<" in addr_str and ">" in addr_str:
                return addr_str.split("<")[1].split(">")[0].lower()
            return addr_str.strip().lower()
        
        all_recipients = [extract_email(str(addr)) for addr in to_addrs + cc_addrs]
        
        filter_addr = alias.lower() if "@" in alias else f"{alias.lower()}@woodcreek.me"
        return filter_addr in all_recipients
        
    except Exception:
        return False


if __name__ == "__main__":
    import sys
    sys.exit(main())