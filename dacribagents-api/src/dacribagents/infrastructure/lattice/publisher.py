"""
Woodcreek Agents + Lattice Integration Examples

This module shows how to:
1. Publish emails to Lattice via Kafka (lattice.mail.raw.v1)
2. Use LatticeEmailRetriever in agent RAG pipelines
3. Configure agents for different aliases (solar, hoa, agents)
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from confluent_kafka import Producer
from loguru import logger

from lattice_retriever import LatticeEmailRetriever, get_lattice_retriever


# =============================================================================
# Part 1: Publishing to Lattice (for backfill or custom ingestion)
# =============================================================================

class LatticeEmailPublisher:
    """
    Publishes raw emails to Lattice via Kafka.
    
    Use this for:
    - Backfilling historical emails
    - Custom email sources not supported by Lattice connectors
    - Testing the pipeline
    """
    
    def __init__(
        self,
        kafka_brokers: str,
        kafka_username: str,
        kafka_password: str,
        tenant_id: str = "woodcreek",
        source_service: str = "woodcreek-agents",
        source_version: str = "1.0.0",
    ):
        self.tenant_id = tenant_id
        self.source_service = source_service
        self.source_version = source_version
        
        # Confluent Cloud config
        self.producer = Producer({
            "bootstrap.servers": kafka_brokers,
            "security.protocol": "SASL_SSL",
            "sasl.mechanisms": "PLAIN",
            "sasl.username": kafka_username,
            "sasl.password": kafka_password,
        })
        
        self.topic = "lattice.mail.raw.v1"
    
    def _create_envelope(
        self,
        payload: dict,
        account_id: str,
        alias: str | None = None,
        trace_id: str | None = None,
    ) -> dict:
        """
        Create a Lattice envelope for the message.
        
        All messages to Lattice workers must be wrapped in this envelope format.
        """
        return {
            # Required envelope fields
            "message_id": str(uuid.uuid4()),
            "schema_version": "v1",
            "domain": "mail",
            "stage": "raw",
            "created_at": datetime.now(timezone.utc).isoformat(),
            
            # Source identification
            "source": {
                "service": self.source_service,
                "version": self.source_version,
            },
            
            # Tenant/account context
            "tenant_id": self.tenant_id,
            "account_id": account_id,
            "alias": alias,
            
            # Data classification
            "data_classification": "internal",
            "pii": True,  # Emails contain PII
            
            # Optional tracing
            "trace_id": trace_id or str(uuid.uuid4()),
            
            # The actual payload
            "payload": payload,
        }
    
    def publish_raw_email(
        self,
        rfc822_bytes: bytes,
        account_id: str,
        provider: str = "imap",
        provider_message_id: str | None = None,
        alias: str | None = None,
        folder: str = "INBOX",
        received_at: datetime | None = None,
    ) -> str:
        """
        Publish a raw RFC822 email to Lattice.
        
        Args:
            rfc822_bytes: Raw email bytes (RFC822 format)
            account_id: Account identifier (e.g., "workmail-agents")
            provider: Email provider (imap, gmail)
            provider_message_id: Unique ID from provider (UID for IMAP)
            alias: Email alias for routing (solar, hoa, agents)
            folder: Source folder
            received_at: When email was received
        
        Returns:
            message_id of the published envelope
        """
        import base64
        import hashlib
        
        # Generate provider_message_id if not provided
        if not provider_message_id:
            provider_message_id = hashlib.sha256(rfc822_bytes).hexdigest()[:32]
        
        # Create payload matching lattice.mail.raw.v1 schema
        payload = {
            "provider": provider,
            "provider_message_id": provider_message_id,
            "folder": folder,
            "received_at": (received_at or datetime.now(timezone.utc)).isoformat(),
            "raw_rfc822_base64": base64.b64encode(rfc822_bytes).decode("utf-8"),
            "size_bytes": len(rfc822_bytes),
        }
        
        # Wrap in envelope
        envelope = self._create_envelope(
            payload=payload,
            account_id=account_id,
            alias=alias,
        )
        
        # Publish to Kafka
        message_id = envelope["message_id"]
        
        def delivery_callback(err, msg):
            if err:
                logger.error(f"Failed to deliver message {message_id}: {err}")
            else:
                logger.info(f"Published {message_id} to {msg.topic()}[{msg.partition()}]")
        
        self.producer.produce(
            topic=self.topic,
            key=f"{self.tenant_id}:{account_id}:{provider_message_id}".encode(),
            value=json.dumps(envelope).encode(),
            callback=delivery_callback,
        )
        
        # Flush to ensure delivery
        self.producer.flush()
        
        return message_id
    
    def backfill_from_imap(
        self,
        host: str,
        email: str,
        password: str,
        account_id: str,
        alias: str | None = None,
        folder: str = "INBOX",
        limit: int = 100,
    ) -> int:
        """
        Backfill emails from IMAP directly to Lattice.
        
        Args:
            host: IMAP server hostname
            email: Email address
            password: Password or app password
            account_id: Lattice account ID
            alias: Optional alias for routing
            folder: IMAP folder to read from
            limit: Maximum emails to backfill
        
        Returns:
            Number of emails published
        """
        import imaplib
        
        logger.info(f"Connecting to IMAP {host} as {email}")
        
        conn = imaplib.IMAP4_SSL(host, 993)
        conn.login(email, password)
        conn.select(folder, readonly=True)
        
        # Fetch all message UIDs
        typ, uids_data = conn.uid("SEARCH", None, "ALL")
        if typ != "OK":
            raise RuntimeError("IMAP search failed")
        
        uids = uids_data[0].split()[-limit:]  # Take last N
        logger.info(f"Found {len(uids)} emails to backfill")
        
        count = 0
        for uid in uids:
            typ, msg_data = conn.uid("FETCH", uid, "(RFC822)")
            if typ != "OK" or not msg_data or not msg_data[0]:
                continue
            
            rfc822_bytes = msg_data[0][1]
            
            try:
                self.publish_raw_email(
                    rfc822_bytes=rfc822_bytes,
                    account_id=account_id,
                    provider="imap",
                    provider_message_id=uid.decode(),
                    alias=alias,
                    folder=folder,
                )
                count += 1
            except Exception as e:
                logger.error(f"Failed to publish UID {uid}: {e}")
        
        conn.logout()
        logger.info(f"Backfilled {count} emails to Lattice")
        return count


# =============================================================================
# Part 2: Using Lattice Retriever in Agents
# =============================================================================

class LatticeRAGMixin:
    """
    Mixin class to add Lattice RAG capabilities to agents.
    
    Add this to your agent classes to enable email retrieval via Lattice.
    """
    
    _retriever: LatticeEmailRetriever | None = None
    
    def _get_retriever(self, alias: str | None = None) -> LatticeEmailRetriever:
        """Get or create a Lattice retriever."""
        if self._retriever is None or getattr(self._retriever, 'alias', None) != alias:
            self._retriever = get_lattice_retriever(
                alias=alias,
                tenant_id=os.getenv("LATTICE_TENANT_ID", "woodcreek"),
                top_k=int(os.getenv("RAG_TOP_K", "5")),
                min_score=float(os.getenv("RAG_MIN_SCORE", "0.25")),
            )
        return self._retriever
    
    async def retrieve_email_context(
        self, 
        query: str, 
        alias: str | None = None,
        max_chars: int = 3000,
    ) -> str:
        """
        Retrieve relevant email context for a query.
        
        Args:
            query: User query to search for
            alias: Optional alias filter (solar, hoa, agents)
            max_chars: Maximum context length
        
        Returns:
            Formatted context string for LLM
        """
        retriever = self._get_retriever(alias=alias)
        
        # Get relevant documents
        docs = retriever.get_relevant_documents(query)
        
        if not docs:
            return ""
        
        # Format as context
        return retriever.format_context(docs, max_chars=max_chars)


# Example: Updated Solar Agent with Lattice RAG
class SolarAgentWithLattice(LatticeRAGMixin):
    """
    Solar installation agent with Lattice email RAG.
    
    This agent specializes in solar panel installation queries
    and retrieves context from solar@ emails via Lattice.
    """
    
    SYSTEM_PROMPT_WITH_RAG = """You are a helpful assistant specializing in solar panel installation 
for the Woodcreek community in Fate, Texas. You have access to email correspondence 
about solar installations.

RELEVANT EMAIL CONTEXT:
{context}

Use this context to answer questions about solar installation schedules, permits, 
equipment, contractors, and HOA approval processes. If the context doesn't contain 
relevant information, say so and provide general guidance.

Contact: solar@woodcreek.me
"""
    
    SYSTEM_PROMPT_NO_RAG = """You are a helpful assistant specializing in solar panel installation 
for the Woodcreek community in Fate, Texas.

You can help with general questions about solar installation, but you don't have access 
to specific email correspondence at the moment.

For specific questions about your installation, contact: solar@woodcreek.me
"""
    
    async def handle_query(self, query: str) -> str:
        """Handle a user query about solar installation."""
        # Retrieve context from solar@ emails
        context = await self.retrieve_email_context(
            query=f"solar panel installation {query}",
            alias="solar",
            max_chars=3000,
        )
        
        # Build system prompt
        if context:
            system_prompt = self.SYSTEM_PROMPT_WITH_RAG.format(context=context)
        else:
            system_prompt = self.SYSTEM_PROMPT_NO_RAG
        
        # Call LLM (implementation depends on your LLM setup)
        response = await self._call_llm(system_prompt, query)
        return response
    
    async def _call_llm(self, system_prompt: str, user_query: str) -> str:
        """Call the LLM with system prompt and query."""
        # Placeholder - integrate with your actual LLM
        # e.g., LangGraph, OpenAI, local vLLM
        raise NotImplementedError("Implement LLM call")


# Example: Updated HOA Agent with Lattice RAG
class HOAAgentWithLattice(LatticeRAGMixin):
    """
    HOA compliance agent with Lattice email RAG.
    
    This agent handles HOA rules, violations, and architectural 
    approval queries using hoa@ email context.
    """
    
    SYSTEM_PROMPT_WITH_RAG = """You are a helpful assistant for HOA compliance and community rules
in the Woodcreek community in Fate, Texas.

RELEVANT EMAIL CONTEXT:
{context}

Use this context to answer questions about:
- HOA rules and CC&Rs
- Architectural approval processes
- Violations and remediation
- Community guidelines
- TownSq platform usage

Be accurate and cite specific emails when relevant. If uncertain, recommend
contacting the HOA directly.

Contact: hoa@woodcreek.me
"""
    
    async def handle_query(self, query: str) -> str:
        """Handle a user query about HOA compliance."""
        context = await self.retrieve_email_context(
            query=f"HOA rules compliance {query}",
            alias="hoa",
            max_chars=3000,
        )
        
        if context:
            system_prompt = self.SYSTEM_PROMPT_WITH_RAG.format(context=context)
        else:
            system_prompt = "You are a helpful HOA assistant. Contact hoa@woodcreek.me for specific questions."
        
        response = await self._call_llm(system_prompt, query)
        return response
    
    async def _call_llm(self, system_prompt: str, user_query: str) -> str:
        raise NotImplementedError("Implement LLM call")


# =============================================================================
# Part 3: CLI for Backfill and Testing
# =============================================================================

def main():
    """CLI for Lattice integration testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Lattice Integration CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    # Backfill command
    backfill_parser = subparsers.add_parser("backfill", help="Backfill emails to Lattice")
    backfill_parser.add_argument("--host", required=True, help="IMAP host")
    backfill_parser.add_argument("--email", required=True, help="Email address")
    backfill_parser.add_argument("--password", required=True, help="Password")
    backfill_parser.add_argument("--account-id", required=True, help="Lattice account ID")
    backfill_parser.add_argument("--alias", help="Email alias (solar, hoa, agents)")
    backfill_parser.add_argument("--folder", default="INBOX", help="IMAP folder")
    backfill_parser.add_argument("--limit", type=int, default=100, help="Max emails")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search emails via Lattice")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--alias", help="Filter by alias")
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    
    args = parser.parse_args()
    
    if args.command == "backfill":
        publisher = LatticeEmailPublisher(
            kafka_brokers=os.environ["KAFKA_BROKERS"],
            kafka_username=os.environ["KAFKA_USERNAME"],
            kafka_password=os.environ["KAFKA_PASSWORD"],
        )
        
        count = publisher.backfill_from_imap(
            host=args.host,
            email=args.email,
            password=args.password,
            account_id=args.account_id,
            alias=args.alias,
            folder=args.folder,
            limit=args.limit,
        )
        print(f"Backfilled {count} emails")
    
    elif args.command == "search":
        retriever = get_lattice_retriever(
            alias=args.alias,
            top_k=args.top_k,
        )
        
        docs = retriever.get_relevant_documents(args.query)
        
        print(f"\nFound {len(docs)} results for: {args.query}\n")
        for i, doc in enumerate(docs, 1):
            print(f"--- Result {i} (score: {doc.metadata['score']:.3f}) ---")
            print(f"Subject: {doc.metadata['subject']}")
            print(f"From: {doc.metadata['sender']}")
            print(f"Date: {doc.metadata['date']}")
            print(f"Chunk: {doc.metadata['chunk_index']+1}/{doc.metadata['total_chunks']}")
            print(f"\n{doc.page_content[:500]}...")
            print()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()