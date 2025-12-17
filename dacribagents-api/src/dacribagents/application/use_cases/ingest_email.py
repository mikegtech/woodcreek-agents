from __future__ import annotations
from dacribagents.application.ports.email_source import EmailSource
from dacribagents.application.ports.vector_store import VectorStore
from dacribagents.application.ports.checkpoint_store import CheckpointStore
from dacribagents.infrastructure.email.providers.workmail_imap.mapper import rfc822_to_email_message

class IngestEmailUseCase:
    def __init__(self, source: EmailSource, store: VectorStore, checkpoints: CheckpointStore, embedder) -> None:
        self.source = source
        self.store = store
        self.checkpoints = checkpoints
        self.embedder = embedder

    def run(self, account: str, folder: str) -> int:
        cursor = self.checkpoints.load(account, folder)
        raws, new_cursor = self.source.fetch_since(cursor)

        count = 0
        for raw in raws:
            msg = rfc822_to_email_message(raw.provider, raw.account, raw.folder, raw.rfc822_bytes)
            if not msg.text and not msg.subject:
                continue

            embedding = self.embedder.embed(msg.text or msg.subject)
            self.store.upsert_email(msg, embedding)
            count += 1

        self.checkpoints.save(account, new_cursor)
        return count
