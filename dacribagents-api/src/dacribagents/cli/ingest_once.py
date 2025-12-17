from __future__ import annotations

import os

from dacribagents.application.use_cases.ingest_email import IngestEmailUseCase
from dacribagents.infrastructure.email.providers.workmail_imap.client import (
    WorkMailImapConfig,
    WorkMailImapEmailSource,
)

from dacribagents.infrastructure import (
    get_milvus_client
)

def main() -> int:
    cfg = WorkMailImapConfig(
        region=os.environ["WORKMAIL_REGION"],
        email=os.environ["WORKMAIL_EMAIL"],
        password=os.environ["WORKMAIL_PASSWORD"],
        folder=os.getenv("WORKMAIL_FOLDER", "INBOX"),
    )

    source = WorkMailImapEmailSource(cfg)

    milvus = get_milvus_client()
    store = MilvusEmailStore(milvus)
    checkpoints = MilvusCheckpointStore(milvus)

    embedder = EmbeddingsFactory.from_env()

    uc = IngestEmailUseCase(
        source=source,
        store=store,
        checkpoints=checkpoints,
        embedder=embedder,
    )

    count = uc.run(account=cfg.email, folder=cfg.folder)
    print(f"Ingested {count} messages from {cfg.folder} for {cfg.email}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
