"""One-shot email ingestion from WorkMail with recipient filtering."""

from __future__ import annotations

import argparse
import os

from dacribagents.application.use_cases.ingest_email import IngestEmailUseCase
from dacribagents.infrastructure.email.providers.workmail_imap.client import (
    WorkMailImapConfig,
    WorkMailImapEmailSource,
)
from dacribagents.infrastructure import get_milvus_client
from dacribagents.infrastructure.stores import MilvusEmailStore
from dacribagents.infrastructure.embeddings import EmbeddingsFactory
from dacribagents.infrastructure.attachments.s3_store import s3_store_from_env


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest emails from WorkMail")
    parser.add_argument("--reset", action="store_true", help="Reset flags in target folder for reprocessing")
    parser.add_argument("--folder", default=None, help="Override folder to reset (default: account folder)")
    parser.add_argument("--no-filter", action="store_true", help="Disable recipient filtering (process all emails)")
    parser.add_argument("--to-only", action="store_true", help="Only match To field, not CC")
    args = parser.parse_args()

    cfg = WorkMailImapConfig(
        region=os.environ["WORKMAIL_REGION"],
        email=os.environ["WORKMAIL_EMAIL"],
        password=os.environ["WORKMAIL_PASSWORD"],
        folder=os.getenv("WORKMAIL_FOLDER", "INBOX"),
    )

    source = WorkMailImapEmailSource(cfg)

    # Handle reset flag
    if args.reset:
        count = source.reset_flags(folder=args.folder)
        print(f"Reset {count} emails in {args.folder or cfg.target_folder} for reprocessing")
        source.disconnect()
        return 0

    milvus = get_milvus_client()
    milvus.connect()

    store = MilvusEmailStore(milvus)
    embedder = EmbeddingsFactory.from_env()

    # Attachment storage (optional)
    attachment_store = None
    if os.getenv("ATTACHMENTS_ENABLED", "false").lower() == "true":
        attachment_store = s3_store_from_env()

    max_attachment_mb = float(os.getenv("ATTACHMENTS_MAX_MB", "25"))
    
    # Recipient filtering (default: enabled)
    require_exact_recipient = not args.no_filter
    if os.getenv("EMAIL_REQUIRE_EXACT_RECIPIENT", "").lower() == "false":
        require_exact_recipient = False
    
    include_cc = not args.to_only
    if os.getenv("EMAIL_INCLUDE_CC", "").lower() == "false":
        include_cc = False

    uc = IngestEmailUseCase(
        source=source,
        store=store,
        embedder=embedder,
        attachment_store=attachment_store,
        max_attachment_mb=max_attachment_mb,
        require_exact_recipient=require_exact_recipient,
        include_cc=include_cc,
    )

    print(f"Ingesting emails for: {cfg.email}")
    print(f"Recipient filtering: {'enabled' if require_exact_recipient else 'disabled'}")
    if require_exact_recipient:
        print(f"Include CC: {'yes' if include_cc else 'no (To only)'}")
    
    count = uc.run()
    print(f"Ingested {count} emails from {cfg.folder} → {cfg.target_folder}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())