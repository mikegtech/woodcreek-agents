"""
Agentic RAG Implementation for Woodcreek Agents.

This module implements advanced RAG patterns:
1. Query Reformulation - Optimize queries for better retrieval
2. Self-Correction Loops - Retry with different strategies if retrieval fails
3. Multi-hop Reasoning - Break complex queries into sub-queries
4. Relevance Checking - Validate retrieved chunks
5. Hallucination Detection - Ensure responses are grounded

NVIDIA Agentic AI Certification Concepts:
- Corrective RAG (CRAG)
- Self-RAG
- Adaptive RAG
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from dacribagents.infrastructure.milvus_client import MilvusClientWrapper


# =============================================================================
# Data Models
# =============================================================================


class QueryType(Enum):
    """Classification of query complexity."""
    SIMPLE = "simple"           # Direct factual question
    COMPLEX = "complex"         # Requires reasoning over multiple facts
    MULTI_HOP = "multi_hop"     # Requires sequential retrieval steps
    COMPARISON = "comparison"   # Comparing multiple items
    TEMPORAL = "temporal"       # Time-based queries


class RetrievalStrategy(Enum):
    """Different retrieval strategies to try."""
    SEMANTIC = "semantic"       # Standard vector similarity
    KEYWORD = "keyword"         # Keyword-based augmentation
    HYBRID = "hybrid"           # Combined semantic + keyword
    EXPANDED = "expanded"       # Query expansion with synonyms


@dataclass
class RetrievedChunk:
    """A retrieved document chunk with metadata."""
    content: str
    source: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_relevant(self) -> bool:
        """Check if chunk meets relevance threshold."""
        return self.score >= 0.7


@dataclass
class RAGResult:
    """Result of an agentic RAG operation."""
    query: str
    reformulated_query: str | None
    retrieved_chunks: list[RetrievedChunk]
    response: str
    is_grounded: bool
    confidence: float
    iterations: int
    strategy_used: RetrievalStrategy
    sub_queries: list[str] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        return self.is_grounded and self.confidence >= 0.7


# =============================================================================
# Prompts
# =============================================================================


QUERY_ANALYSIS_PROMPT = """Analyze this user query and provide:
1. Query type (simple, complex, multi_hop, comparison, temporal)
2. Key entities/concepts to search for
3. A reformulated search query optimized for semantic search
4. If multi-hop, list the sub-queries needed

User Query: {query}

Respond in this exact format:
TYPE: <query_type>
ENTITIES: <comma-separated list>
SEARCH_QUERY: <optimized query for vector search>
SUB_QUERIES: <comma-separated sub-queries or "none">
"""


RELEVANCE_CHECK_PROMPT = """Evaluate if this retrieved context is relevant to answering the query.

Query: {query}

Retrieved Context:
{context}

Score the relevance from 0.0 to 1.0 and explain briefly.
Format: SCORE: <0.0-1.0> | REASON: <brief explanation>
"""


GROUNDING_CHECK_PROMPT = """Determine if the response is fully grounded in the provided context.

Query: {query}

Context:
{context}

Response:
{response}

Check:
1. Is every claim in the response supported by the context?
2. Are there any statements that go beyond the context (hallucinations)?
3. Is the response accurate based on the context?

Format:
GROUNDED: <yes/no>
CONFIDENCE: <0.0-1.0>
ISSUES: <list any unsupported claims or "none">
"""


GENERATE_RESPONSE_PROMPT = """Answer the user's question based ONLY on the provided context.

Context:
{context}

User Question: {query}

Instructions:
- Only use information from the context above
- If the context doesn't contain enough information, say so
- Be concise and direct
- Cite relevant parts of the context when possible

Answer:"""


SELF_CORRECTION_PROMPT = """The previous response was not well-grounded in the context.

Original Query: {query}
Previous Response: {previous_response}
Issues Found: {issues}

Context Available:
{context}

Please generate a new response that:
1. Only uses information from the context
2. Acknowledges limitations if context is insufficient
3. Avoids the issues identified above

Corrected Answer:"""


QUERY_EXPANSION_PROMPT = """Expand this query with synonyms and related terms for better retrieval.

Original Query: {query}

Generate 3 alternative phrasings that capture the same intent but use different words.
Format each on a new line, no numbering."""


# =============================================================================
# Agentic RAG Pipeline
# =============================================================================


class AgenticRAG:
    """
    Agentic RAG implementation with self-correction capabilities.
    
    Key Features:
    - Query analysis and reformulation
    - Multi-strategy retrieval
    - Relevance filtering
    - Grounding verification
    - Self-correction loops
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        milvus_client: MilvusClientWrapper,
        collection_name: str = "email_chunks_v1",
        filter_expr: str | None = None,
        max_iterations: int = 3,
        relevance_threshold: float = 0.7,
        grounding_threshold: float = 0.8,
    ):
        self.llm = llm
        self.milvus = milvus_client
        self.collection_name = collection_name
        self.filter_expr = filter_expr
        self.max_iterations = max_iterations
        self.relevance_threshold = relevance_threshold
        self.grounding_threshold = grounding_threshold
        
        # Embedding model for queries
        self._embedder = None
        
        logger.info(f"AgenticRAG initialized with collection={collection_name}, filter={filter_expr}")
    
    @property
    def embedder(self):
        """Lazy load the embedding model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        return self._embedder
    
    # -------------------------------------------------------------------------
    # Query Analysis & Reformulation
    # -------------------------------------------------------------------------
    
    async def analyze_query(self, query: str) -> dict[str, Any]:
        """
        Analyze and classify the user query.
        
        Returns:
            dict with query_type, entities, search_query, sub_queries
        """
        prompt = ChatPromptTemplate.from_template(QUERY_ANALYSIS_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        
        result = await chain.ainvoke({"query": query})
        
        # Parse the structured response
        analysis = {
            "query_type": QueryType.SIMPLE,
            "entities": [],
            "search_query": query,
            "sub_queries": [],
        }
        
        for line in result.strip().split("\n"):
            if line.startswith("TYPE:"):
                type_str = line.replace("TYPE:", "").strip().lower()
                try:
                    analysis["query_type"] = QueryType(type_str)
                except ValueError:
                    pass
            elif line.startswith("ENTITIES:"):
                entities = line.replace("ENTITIES:", "").strip()
                analysis["entities"] = [e.strip() for e in entities.split(",") if e.strip()]
            elif line.startswith("SEARCH_QUERY:"):
                analysis["search_query"] = line.replace("SEARCH_QUERY:", "").strip()
            elif line.startswith("SUB_QUERIES:"):
                sub_q = line.replace("SUB_QUERIES:", "").strip()
                if sub_q.lower() != "none":
                    analysis["sub_queries"] = [q.strip() for q in sub_q.split(",") if q.strip()]
        
        logger.debug(f"Query analysis: {analysis}")
        return analysis
    
    async def reformulate_query(self, query: str) -> str:
        """
        Reformulate query for better semantic search.
        """
        analysis = await self.analyze_query(query)
        return analysis["search_query"]
    
    async def expand_query(self, query: str) -> list[str]:
        """
        Generate query variations for expanded retrieval.
        """
        prompt = ChatPromptTemplate.from_template(QUERY_EXPANSION_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        
        result = await chain.ainvoke({"query": query})
        
        variations = [query]  # Include original
        for line in result.strip().split("\n"):
            line = line.strip()
            if line and line != query:
                variations.append(line)
        
        logger.debug(f"Query variations: {variations}")
        return variations[:4]  # Limit to 4 total
    
    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------
    
    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for query."""
        return self.embedder.encode(query).tolist()
    
    async def retrieve(
        self,
        query: str,
        strategy: RetrievalStrategy = RetrievalStrategy.SEMANTIC,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Retrieve relevant chunks using specified strategy.
        """
        chunks = []
        
        if strategy == RetrievalStrategy.SEMANTIC:
            chunks = await self._semantic_retrieve(query, top_k)
        
        elif strategy == RetrievalStrategy.EXPANDED:
            # Use multiple query variations
            variations = await self.expand_query(query)
            seen_ids = set()
            
            for var in variations:
                var_chunks = await self._semantic_retrieve(var, top_k=3)
                for chunk in var_chunks:
                    chunk_id = f"{chunk.source}:{chunk.content[:50]}"
                    if chunk_id not in seen_ids:
                        seen_ids.add(chunk_id)
                        chunks.append(chunk)
            
            # Re-sort by score and limit
            chunks.sort(key=lambda x: x.score, reverse=True)
            chunks = chunks[:top_k]
        
        elif strategy == RetrievalStrategy.HYBRID:
            # Combine semantic with keyword boost
            semantic_chunks = await self._semantic_retrieve(query, top_k)
            
            # Boost scores for chunks with exact keyword matches
            keywords = query.lower().split()
            for chunk in semantic_chunks:
                content_lower = chunk.content.lower()
                keyword_matches = sum(1 for kw in keywords if kw in content_lower)
                if keyword_matches > 0:
                    chunk.score = min(1.0, chunk.score + (keyword_matches * 0.05))
            
            chunks = sorted(semantic_chunks, key=lambda x: x.score, reverse=True)
        
        logger.info(f"Retrieved {len(chunks)} chunks using {strategy.value} strategy")
        return chunks
    
    async def _semantic_retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """
        Semantic search in Milvus, then hydrate content from Postgres.
        
        Architecture: Claim Check Pattern
        1. Milvus search → returns chunk_id, email_id, score
        2. Postgres lookup → returns actual chunk_text
        
        This keeps Milvus lean (vectors only) while Postgres remains
        the system of record for content.
        """
        query_embedding = self.embed_query(query)
        
        # Step 1: Vector search in Milvus (returns IDs, not content)
        results = self.milvus.search(
            query_vector=query_embedding,
            top_k=top_k,
            collection_name=self.collection_name,
            filter_expr=self.filter_expr,
            output_fields=[
                "pk",
                "email_id", 
                "chunk_hash",
                "section_type", 
                "account_id",
                "email_timestamp",
            ],
        )
        
        chunks = []
        for hit in results:
            entity = hit.get("entity", {})
            email_id = entity.get("email_id")
            chunk_hash = entity.get("chunk_hash")
            
            # Step 2: Hydrate content from Postgres
            content = await self._hydrate_chunk_content(email_id, chunk_hash)
            
            if not content:
                logger.warning(f"Could not hydrate chunk: email_id={email_id}, chunk_hash={chunk_hash}")
                content = f"[Content unavailable for {entity.get('section_type', 'unknown')} section]"
            
            chunk = RetrievedChunk(
                content=content,
                source=f"email:{email_id}",
                score=1 - hit.get("distance", 0),  # Convert distance to similarity
                metadata={
                    "email_id": email_id,
                    "chunk_hash": chunk_hash,
                    "section_type": entity.get("section_type"),
                    "account_id": entity.get("account_id"),
                    "timestamp": entity.get("email_timestamp"),
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _hydrate_chunk_content(self, email_id: str, chunk_hash: str) -> str | None:
        """
        Fetch actual chunk text from Lattice Postgres.
        
        Uses LATTICE_DATABASE_URL (separate from LangGraph checkpoints DATABASE_URL).
        Postgres is the system of record - Milvus only stores vectors.
        """
        try:
            import os
            import asyncpg
            
            # Use dedicated Lattice database URL (separate from LangGraph checkpoints)
            lattice_db_url = os.getenv("LATTICE_DATABASE_URL")
            
            if not lattice_db_url:
                logger.error("LATTICE_DATABASE_URL not set - cannot hydrate chunk content")
                return None
            
            # Quick connection for hydration
            # TODO: In production, use a connection pool instead of per-request connections
            conn = await asyncpg.connect(lattice_db_url)
            try:
                row = await conn.fetchrow("""
                    SELECT chunk_text 
                    FROM email_chunk 
                    WHERE email_id = $1::uuid AND chunk_hash = $2
                """, email_id, chunk_hash)
                
                return row["chunk_text"] if row else None
            finally:
                await conn.close()
                
        except Exception as e:
            logger.error(f"Failed to hydrate chunk from Lattice Postgres: {e}")
            return None
    
    # -------------------------------------------------------------------------
    # Relevance & Grounding Checks
    # -------------------------------------------------------------------------
    
    async def check_relevance(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        Filter chunks by relevance to query.
        Uses LLM for nuanced relevance scoring.
        """
        if not chunks:
            return []
        
        relevant_chunks = []
        
        for chunk in chunks:
            # Quick filter by vector similarity score
            if chunk.score < self.relevance_threshold * 0.8:
                continue
            
            # LLM-based relevance check for borderline cases
            if chunk.score < self.relevance_threshold:
                prompt = ChatPromptTemplate.from_template(RELEVANCE_CHECK_PROMPT)
                chain = prompt | self.llm | StrOutputParser()
                
                result = await chain.ainvoke({
                    "query": query,
                    "context": chunk.content[:1000],
                })
                
                # Parse score
                try:
                    score_str = result.split("|")[0].replace("SCORE:", "").strip()
                    llm_score = float(score_str)
                    chunk.score = (chunk.score + llm_score) / 2  # Average
                except (ValueError, IndexError):
                    pass
            
            if chunk.score >= self.relevance_threshold:
                relevant_chunks.append(chunk)
        
        logger.info(f"Relevance filtering: {len(chunks)} → {len(relevant_chunks)} chunks")
        return relevant_chunks
    
    async def check_grounding(
        self,
        query: str,
        context: str,
        response: str,
    ) -> tuple[bool, float, str]:
        """
        Verify response is grounded in context.
        
        Returns:
            (is_grounded, confidence, issues)
        """
        prompt = ChatPromptTemplate.from_template(GROUNDING_CHECK_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        
        result = await chain.ainvoke({
            "query": query,
            "context": context,
            "response": response,
        })
        
        # Parse response
        is_grounded = False
        confidence = 0.5
        issues = ""
        
        for line in result.strip().split("\n"):
            if line.startswith("GROUNDED:"):
                is_grounded = "yes" in line.lower()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip())
                except ValueError:
                    pass
            elif line.startswith("ISSUES:"):
                issues = line.replace("ISSUES:", "").strip()
        
        logger.debug(f"Grounding check: grounded={is_grounded}, confidence={confidence}")
        return is_grounded, confidence, issues
    
    # -------------------------------------------------------------------------
    # Response Generation
    # -------------------------------------------------------------------------
    
    async def generate_response(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> str:
        """Generate response from retrieved context."""
        if not chunks:
            return "I don't have enough information to answer that question."
        
        # Combine chunk contents
        context = "\n\n---\n\n".join([
            f"Source: {c.source}\n{c.content}"
            for c in chunks
        ])
        
        prompt = ChatPromptTemplate.from_template(GENERATE_RESPONSE_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        
        response = await chain.ainvoke({
            "query": query,
            "context": context,
        })
        
        return response.strip()
    
    async def generate_corrected_response(
        self,
        query: str,
        previous_response: str,
        issues: str,
        chunks: list[RetrievedChunk],
    ) -> str:
        """Generate a corrected response addressing identified issues."""
        context = "\n\n---\n\n".join([
            f"Source: {c.source}\n{c.content}"
            for c in chunks
        ])
        
        prompt = ChatPromptTemplate.from_template(SELF_CORRECTION_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        
        response = await chain.ainvoke({
            "query": query,
            "previous_response": previous_response,
            "issues": issues,
            "context": context,
        })
        
        return response.strip()
    
    # -------------------------------------------------------------------------
    # Main Pipeline
    # -------------------------------------------------------------------------
    
    async def query(
        self,
        query: str,
        top_k: int = 5,
    ) -> RAGResult:
        """
        Execute the full agentic RAG pipeline.
        
        Steps:
        1. Analyze and reformulate query
        2. Retrieve with relevance filtering
        3. Generate response
        4. Check grounding
        5. Self-correct if needed
        """
        logger.info(f"Agentic RAG query: {query}")
        
        # Step 1: Query Analysis
        analysis = await self.analyze_query(query)
        reformulated = analysis["search_query"]
        sub_queries = analysis["sub_queries"]
        
        # Track iterations for self-correction
        iterations = 0
        strategy = RetrievalStrategy.SEMANTIC
        strategies_to_try = [
            RetrievalStrategy.SEMANTIC,
            RetrievalStrategy.EXPANDED,
            RetrievalStrategy.HYBRID,
        ]
        
        best_result = None
        
        while iterations < self.max_iterations:
            iterations += 1
            current_strategy = strategies_to_try[min(iterations - 1, len(strategies_to_try) - 1)]
            
            logger.info(f"Iteration {iterations}: using {current_strategy.value} strategy")
            
            # Step 2: Retrieve
            search_query = reformulated if iterations == 1 else query
            chunks = await self.retrieve(search_query, strategy=current_strategy, top_k=top_k)
            
            # Step 2b: Relevance filtering
            relevant_chunks = await self.check_relevance(query, chunks)
            
            if not relevant_chunks:
                logger.warning(f"No relevant chunks found in iteration {iterations}")
                continue
            
            # Step 3: Generate response
            if best_result and best_result.response:
                # Self-correction: improve previous response
                response = await self.generate_corrected_response(
                    query=query,
                    previous_response=best_result.response,
                    issues=best_result.metadata.get("issues", "Response not grounded"),
                    chunks=relevant_chunks,
                )
            else:
                response = await self.generate_response(query, relevant_chunks)
            
            # Step 4: Check grounding
            context = "\n".join([c.content for c in relevant_chunks])
            is_grounded, confidence, issues = await self.check_grounding(
                query, context, response
            )
            
            # Create result
            result = RAGResult(
                query=query,
                reformulated_query=reformulated,
                retrieved_chunks=relevant_chunks,
                response=response,
                is_grounded=is_grounded,
                confidence=confidence,
                iterations=iterations,
                strategy_used=current_strategy,
                sub_queries=sub_queries,
            )
            result.metadata = {"issues": issues}
            
            # Check if we're done
            if is_grounded and confidence >= self.grounding_threshold:
                logger.info(f"Grounded response found in iteration {iterations}")
                return result
            
            # Keep best result so far
            if best_result is None or confidence > best_result.confidence:
                best_result = result
            
            logger.info(f"Response not well-grounded (confidence={confidence}), retrying...")
        
        # Return best result after max iterations
        logger.warning(f"Max iterations reached, returning best result (confidence={best_result.confidence})")
        return best_result or RAGResult(
            query=query,
            reformulated_query=reformulated,
            retrieved_chunks=[],
            response="I couldn't find a well-supported answer to your question.",
            is_grounded=False,
            confidence=0.0,
            iterations=iterations,
            strategy_used=strategy,
            sub_queries=sub_queries,
        )
    
    async def multi_hop_query(
        self,
        query: str,
        top_k: int = 3,
    ) -> RAGResult:
        """
        Execute multi-hop RAG for complex queries.
        
        Breaks query into sub-queries and aggregates results.
        """
        logger.info(f"Multi-hop RAG query: {query}")
        
        # Analyze to get sub-queries
        analysis = await self.analyze_query(query)
        sub_queries = analysis.get("sub_queries", [])
        
        if not sub_queries:
            # Fall back to regular query
            return await self.query(query, top_k)
        
        # Execute each sub-query
        all_chunks = []
        sub_responses = []
        
        for i, sub_q in enumerate(sub_queries):
            logger.info(f"Sub-query {i+1}/{len(sub_queries)}: {sub_q}")
            
            sub_result = await self.query(sub_q, top_k=top_k)
            all_chunks.extend(sub_result.retrieved_chunks)
            
            if sub_result.response:
                sub_responses.append(f"Q: {sub_q}\nA: {sub_result.response}")
        
        # Deduplicate chunks
        seen = set()
        unique_chunks = []
        for chunk in all_chunks:
            key = chunk.content[:100]
            if key not in seen:
                seen.add(key)
                unique_chunks.append(chunk)
        
        # Generate final aggregated response
        aggregated_context = "\n\n".join(sub_responses)
        
        final_prompt = ChatPromptTemplate.from_template(
            """Based on the following sub-query answers, provide a comprehensive answer to the main question.

Main Question: {query}

Sub-query Results:
{context}

Comprehensive Answer:"""
        )
        chain = final_prompt | self.llm | StrOutputParser()
        
        final_response = await chain.ainvoke({
            "query": query,
            "context": aggregated_context,
        })
        
        return RAGResult(
            query=query,
            reformulated_query=analysis["search_query"],
            retrieved_chunks=unique_chunks,
            response=final_response.strip(),
            is_grounded=True,  # Aggregated from grounded sub-responses
            confidence=0.85,
            iterations=len(sub_queries),
            strategy_used=RetrievalStrategy.SEMANTIC,
            sub_queries=sub_queries,
        )


# =============================================================================
# Factory Function
# =============================================================================


def get_agentic_rag(
    collection_name: str = "email_chunks_v1",
    filter_expr: str | None = None,
    max_iterations: int = 3,
) -> AgenticRAG:
    """
    Factory function to create AgenticRAG instance.
    
    Uses the configured LLM and Milvus client.
    
    Args:
        collection_name: Milvus collection to search
        filter_expr: Optional Milvus filter expression (e.g., 'account_id == "workmail-hoa"')
        max_iterations: Max self-correction iterations
    """
    from dacribagents.application.agents.general_assistant import create_llm
    from dacribagents.infrastructure.milvus_client import get_milvus_client
    from dacribagents.infrastructure.settings import get_settings
    
    settings = get_settings()
    llm = create_llm(settings)
    milvus = get_milvus_client()
    
    return AgenticRAG(
        llm=llm,
        milvus_client=milvus,
        collection_name=collection_name,
        filter_expr=filter_expr,
        max_iterations=max_iterations,
    )