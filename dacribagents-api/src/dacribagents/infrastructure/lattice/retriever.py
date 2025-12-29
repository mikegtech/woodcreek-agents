"""
LangChain-compatible retriever for Lattice email chunks.

This retriever queries:
1. Milvus for semantic similarity (chunk-level vectors)
2. Postgres FTS for keyword boosting (optional)
3. Hydrates results with full email context

Usage:
    retriever = LatticeEmailRetriever(
        milvus_uri="http://localhost:19530",
        postgres_dsn="postgresql://lattice:xxx@localhost:5432/lattice",
        tenant_id="woodcreek",
        alias="solar",  # Filter to solar@ emails
    )
    
    docs = retriever.get_relevant_documents("solar panel installation schedule")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from loguru import logger
from pydantic import Field


@dataclass
class ChunkResult:
    """Result from Milvus search."""
    chunk_id: str
    email_id: str
    score: float
    chunk_hash: str
    embedding_version: str | None = None


@dataclass  
class ChunkData:
    """Hydrated chunk data from Postgres."""
    chunk_id: str
    email_id: str
    chunk_text: str
    source_type: str
    chunk_index: int
    total_chunks: int
    section_type: str | None = None


@dataclass
class EmailData:
    """Hydrated email metadata from Postgres."""
    email_id: str
    subject: str
    sender: str
    sender_name: str | None
    sent_at: str
    account_id: str
    provider_message_id: str


class LatticeEmailRetriever(BaseRetriever):
    """
    Hybrid retriever for Lattice email chunks.
    
    Combines Milvus semantic search with optional Postgres FTS boosting
    for improved relevance.
    """
    
    # Connection settings
    milvus_uri: str = Field(default="http://localhost:19530")
    postgres_dsn: str = Field(default="")
    collection: str = Field(default="email_chunks_v1")
    
    # Tenant/account filtering
    tenant_id: str = Field(default="woodcreek")
    account_id: Optional[str] = Field(default=None)
    alias: Optional[str] = Field(default=None)  # solar, hoa, agents
    
    # Search parameters
    top_k: int = Field(default=10)
    min_score: float = Field(default=0.25)
    over_fetch_factor: int = Field(default=2)  # Fetch more for reranking
    
    # FTS settings
    use_fts: bool = Field(default=True)
    fts_boost_weight: float = Field(default=0.2)
    
    # Embedding
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    
    # Internal state (not Pydantic fields)
    _milvus: Any = None
    _embedder: Any = None
    _pg_pool: Any = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_clients()
    
    def _init_clients(self) -> None:
        """Initialize Milvus and Postgres clients."""
        from pymilvus import MilvusClient
        from sentence_transformers import SentenceTransformer
        import psycopg2.pool
        
        # Milvus
        logger.info(f"Connecting to Milvus at {self.milvus_uri}")
        self._milvus = MilvusClient(uri=self.milvus_uri)
        
        # Embedder
        logger.info(f"Loading embedding model: {self.embedding_model}")
        self._embedder = SentenceTransformer(self.embedding_model)
        
        # Postgres connection pool
        if self.postgres_dsn:
            logger.info("Creating Postgres connection pool")
            self._pg_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=5,
                dsn=self.postgres_dsn,
            )
    
    def _embed_query(self, query: str) -> list[float]:
        """Embed a query string."""
        return self._embedder.encode(query, convert_to_numpy=True).tolist()
    
    def _build_filter_expr(self) -> str:
        """Build Milvus filter expression."""
        filters = [f'tenant_id == "{self.tenant_id}"']
        
        if self.account_id:
            filters.append(f'account_id == "{self.account_id}"')
        
        # Alias filtering requires joining with Postgres or storing alias in Milvus
        # For now, we'll do post-filtering if alias is set
        
        return " and ".join(filters)
    
    def _search_milvus(self, embedding: list[float]) -> list[ChunkResult]:
        """Search Milvus for similar chunks."""
        filter_expr = self._build_filter_expr()
        
        results = self._milvus.search(
            collection_name=self.collection,
            data=[embedding],
            limit=self.top_k * self.over_fetch_factor,
            filter=filter_expr,
            output_fields=["email_id", "chunk_id", "chunk_hash", "embedding_version"],
        )
        
        if not results or not results[0]:
            return []
        
        return [
            ChunkResult(
                chunk_id=str(hit["entity"]["chunk_id"]),
                email_id=str(hit["entity"]["email_id"]),
                score=1 - hit["distance"],  # Convert distance to similarity
                chunk_hash=hit["entity"].get("chunk_hash", ""),
                embedding_version=hit["entity"].get("embedding_version"),
            )
            for hit in results[0]
        ]
    
    def _search_fts(self, query: str) -> dict[str, float]:
        """Search Postgres FTS for keyword matches."""
        if not self._pg_pool:
            return {}
        
        conn = self._pg_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        ec.chunk_id::text,
                        ts_rank(ec.fts_vector, plainto_tsquery('english', %s)) as rank
                    FROM email_chunk ec
                    JOIN email e ON ec.email_id = e.id
                    WHERE e.tenant_id = %s
                      AND ec.fts_vector @@ plainto_tsquery('english', %s)
                    ORDER BY rank DESC
                    LIMIT %s
                """, (query, self.tenant_id, query, self.top_k * 2))
                
                return {str(row[0]): float(row[1]) for row in cur.fetchall()}
        finally:
            self._pg_pool.putconn(conn)
    
    def _merge_fts_boost(
        self, 
        milvus_results: list[ChunkResult], 
        fts_scores: dict[str, float]
    ) -> list[ChunkResult]:
        """Boost Milvus scores with FTS ranks."""
        if not fts_scores:
            return milvus_results
        
        # Normalize FTS scores
        max_fts = max(fts_scores.values()) if fts_scores else 1.0
        
        for result in milvus_results:
            fts_score = fts_scores.get(result.chunk_id, 0.0)
            normalized_fts = fts_score / max_fts if max_fts > 0 else 0.0
            
            # Weighted combination
            result.score = (
                result.score * (1 - self.fts_boost_weight) +
                normalized_fts * self.fts_boost_weight
            )
        
        # Re-sort by combined score
        milvus_results.sort(key=lambda x: x.score, reverse=True)
        return milvus_results
    
    def _hydrate_chunk(self, chunk_id: str) -> ChunkData | None:
        """Get full chunk data from Postgres."""
        if not self._pg_pool:
            return None
        
        conn = self._pg_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        chunk_id::text,
                        email_id::text,
                        chunk_text,
                        source_type,
                        chunk_index,
                        total_chunks,
                        section_type
                    FROM email_chunk 
                    WHERE chunk_id = %s::uuid
                """, (chunk_id,))
                
                row = cur.fetchone()
                if not row:
                    return None
                
                return ChunkData(
                    chunk_id=row[0],
                    email_id=row[1],
                    chunk_text=row[2],
                    source_type=row[3],
                    chunk_index=row[4],
                    total_chunks=row[5],
                    section_type=row[6],
                )
        finally:
            self._pg_pool.putconn(conn)
    
    def _hydrate_email(self, email_id: str) -> EmailData | None:
        """Get email metadata from Postgres."""
        if not self._pg_pool:
            return None
        
        conn = self._pg_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        id::text,
                        subject,
                        from_address->>'address' as sender,
                        from_address->>'name' as sender_name,
                        sent_at::text,
                        account_id,
                        provider_message_id
                    FROM email 
                    WHERE id = %s::uuid
                """, (email_id,))
                
                row = cur.fetchone()
                if not row:
                    return None
                
                return EmailData(
                    email_id=row[0],
                    subject=row[1] or "",
                    sender=row[2] or "",
                    sender_name=row[3],
                    sent_at=row[4] or "",
                    account_id=row[5] or "",
                    provider_message_id=row[6] or "",
                )
        finally:
            self._pg_pool.putconn(conn)
    
    def _filter_by_alias(self, results: list[ChunkResult]) -> list[ChunkResult]:
        """Post-filter results by alias (email local part)."""
        if not self.alias or not self._pg_pool:
            return results
        
        # Get email IDs that match the alias
        email_ids = list(set(r.email_id for r in results))
        if not email_ids:
            return results
        
        conn = self._pg_pool.getconn()
        try:
            with conn.cursor() as cur:
                # Check which emails have recipients matching the alias
                cur.execute("""
                    SELECT id::text
                    FROM email
                    WHERE id = ANY(%s::uuid[])
                      AND (
                        -- Check To addresses
                        EXISTS (
                            SELECT 1 FROM jsonb_array_elements(to_addresses) addr
                            WHERE addr->>'address' ILIKE %s
                        )
                        OR
                        -- Check CC addresses  
                        EXISTS (
                            SELECT 1 FROM jsonb_array_elements(cc_addresses) addr
                            WHERE addr->>'address' ILIKE %s
                        )
                      )
                """, (email_ids, f"{self.alias}@%", f"{self.alias}@%"))
                
                matching_email_ids = {row[0] for row in cur.fetchall()}
        finally:
            self._pg_pool.putconn(conn)
        
        return [r for r in results if r.email_id in matching_email_ids]
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """
        Retrieve relevant documents for a query.
        
        This is the main entry point called by LangChain.
        """
        logger.debug(f"Searching for: {query[:100]}...")
        
        # 1. Embed query
        embedding = self._embed_query(query)
        
        # 2. Search Milvus
        milvus_results = self._search_milvus(embedding)
        logger.debug(f"Milvus returned {len(milvus_results)} results")
        
        if not milvus_results:
            return []
        
        # 3. Optional FTS boost
        if self.use_fts and self._pg_pool:
            fts_scores = self._search_fts(query)
            milvus_results = self._merge_fts_boost(milvus_results, fts_scores)
            logger.debug(f"FTS boosted {len(fts_scores)} results")
        
        # 4. Filter by alias if specified
        if self.alias:
            milvus_results = self._filter_by_alias(milvus_results)
            logger.debug(f"Alias filter: {len(milvus_results)} results for {self.alias}@")
        
        # 5. Filter by min score and limit
        filtered = [r for r in milvus_results if r.score >= self.min_score][:self.top_k]
        
        # 6. Hydrate with full context
        documents = []
        for result in filtered:
            chunk_data = self._hydrate_chunk(result.chunk_id)
            email_data = self._hydrate_email(result.email_id)
            
            if not chunk_data:
                logger.warning(f"Could not hydrate chunk {result.chunk_id}")
                continue
            
            # Build document
            doc = Document(
                page_content=chunk_data.chunk_text,
                metadata={
                    # Identifiers
                    "email_id": result.email_id,
                    "chunk_id": result.chunk_id,
                    "chunk_hash": result.chunk_hash,
                    
                    # Scores
                    "score": result.score,
                    
                    # Chunk info
                    "source_type": chunk_data.source_type,
                    "section_type": chunk_data.section_type,
                    "chunk_index": chunk_data.chunk_index,
                    "total_chunks": chunk_data.total_chunks,
                    
                    # Email metadata (if available)
                    "subject": email_data.subject if email_data else "",
                    "sender": email_data.sender if email_data else "",
                    "sender_name": email_data.sender_name if email_data else "",
                    "date": email_data.sent_at if email_data else "",
                    "account_id": email_data.account_id if email_data else "",
                    "provider_message_id": email_data.provider_message_id if email_data else "",
                }
            )
            documents.append(doc)
        
        logger.info(f"Returning {len(documents)} documents for query: {query[:50]}...")
        return documents
    
    def format_context(self, documents: list[Document], max_chars: int = 4000) -> str:
        """
        Format retrieved documents as context for LLM.
        
        Groups chunks by email and formats with metadata.
        """
        if not documents:
            return "No relevant emails found."
        
        # Group by email
        by_email: dict[str, list[Document]] = {}
        for doc in documents:
            email_id = doc.metadata["email_id"]
            if email_id not in by_email:
                by_email[email_id] = []
            by_email[email_id].append(doc)
        
        # Format each email
        context_parts = []
        total_chars = 0
        
        for email_id, chunks in by_email.items():
            # Sort chunks by index
            chunks.sort(key=lambda d: d.metadata.get("chunk_index", 0))
            
            # Get metadata from first chunk
            meta = chunks[0].metadata
            
            email_header = (
                f"[Email from {meta.get('sender', 'unknown')}, "
                f"{meta.get('date', 'unknown date')}]\n"
                f"Subject: {meta.get('subject', 'No subject')}\n"
            )
            
            # Combine chunk content
            content = "\n".join(doc.page_content for doc in chunks)
            
            email_block = f"{email_header}\n{content}\n"
            
            if total_chars + len(email_block) > max_chars:
                # Truncate this email to fit
                remaining = max_chars - total_chars - len(email_header) - 50
                if remaining > 100:
                    content = content[:remaining] + "..."
                    email_block = f"{email_header}\n{content}\n"
                else:
                    break
            
            context_parts.append(email_block)
            total_chars += len(email_block)
        
        return "\n---\n".join(context_parts)


# Factory function
def get_lattice_retriever(
    alias: str | None = None,
    tenant_id: str = "woodcreek",
    **kwargs
) -> LatticeEmailRetriever:
    """
    Create a Lattice retriever with sensible defaults.
    
    Args:
        alias: Filter to specific inbox (solar, hoa, agents)
        tenant_id: Tenant ID for multi-tenant filtering
        **kwargs: Additional arguments passed to LatticeEmailRetriever
    
    Returns:
        Configured LatticeEmailRetriever
    """
    return LatticeEmailRetriever(
        milvus_uri=os.getenv("LATTICE_MILVUS_URI", "http://localhost:19530"),
        postgres_dsn=os.getenv("LATTICE_POSTGRES_DSN", ""),
        tenant_id=tenant_id,
        alias=alias,
        **kwargs,
    )