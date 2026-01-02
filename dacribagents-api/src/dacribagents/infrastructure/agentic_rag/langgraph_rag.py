"""
LangGraph-based Agentic RAG Pipeline.

This implements the same agentic RAG patterns using LangGraph's
state machine approach, which is more aligned with NVIDIA's
recommended patterns for agentic systems.

Key Patterns Implemented:
- Corrective RAG (CRAG): Grade documents and correct retrieval
- Self-RAG: Self-reflection and correction
- Adaptive RAG: Choose strategy based on query complexity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger


# =============================================================================
# State Definition
# =============================================================================


class RAGState(TypedDict):
    """State for the Agentic RAG graph."""
    
    # Input
    query: str
    
    # Query processing
    query_type: str  # simple, complex, multi_hop
    reformulated_query: str
    sub_queries: list[str]
    
    # Retrieval
    documents: list[Document]
    relevant_documents: list[Document]
    retrieval_strategy: str  # semantic, expanded, hybrid
    
    # Generation
    response: str
    
    # Grounding check
    is_grounded: bool
    confidence: float
    issues: str
    
    # Control flow
    iteration: int
    max_iterations: int
    should_retry: bool
    final_answer: str


# =============================================================================
# Prompts
# =============================================================================


QUERY_ROUTER_PROMPT = """Analyze this query and determine the best approach.

Query: {query}

Classify as one of:
- "simple": Direct factual question answerable from a single source
- "complex": Requires reasoning across multiple pieces of information  
- "multi_hop": Requires sequential lookups where each step informs the next

Also provide a search-optimized version of the query.

Format:
TYPE: <simple|complex|multi_hop>
SEARCH_QUERY: <optimized query>
SUB_QUERIES: <comma-separated sub-queries if multi_hop, else "none">
"""


DOCUMENT_GRADER_PROMPT = """Grade whether this document is relevant to the query.

Query: {query}

Document:
{document}

Grade as 'relevant' or 'not_relevant'.
A document is relevant if it contains information that helps answer the query.

Grade: """


HALLUCINATION_GRADER_PROMPT = """Assess whether the response is grounded in the documents.

Documents:
{documents}

Response: {response}

Grade as 'grounded' if all claims in the response are supported by the documents.
Grade as 'not_grounded' if any claims go beyond what's in the documents.

Provide a confidence score from 0.0 to 1.0.

Format:
GRADE: <grounded|not_grounded>
CONFIDENCE: <0.0-1.0>
ISSUES: <any unsupported claims, or "none">
"""


ANSWER_GRADER_PROMPT = """Assess whether the response adequately answers the query.

Query: {query}
Response: {response}

Grade as 'adequate' if the response directly answers the question.
Grade as 'inadequate' if it's off-topic, incomplete, or doesn't address the query.

Grade: """


QUERY_REWRITER_PROMPT = """Rewrite this query to improve retrieval results.

Original Query: {query}
Previous Issues: {issues}

The previous attempt didn't find good results. Rewrite the query to:
1. Be more specific or more general as needed
2. Use different terminology
3. Focus on the core information need

Rewritten Query: """


RESPONSE_GENERATOR_PROMPT = """Answer the question based only on the provided context.

Context:
{context}

Question: {query}

Instructions:
- Only use information from the context
- If the context is insufficient, say so clearly
- Be concise and direct

Answer: """


RESPONSE_CORRECTOR_PROMPT = """Correct this response to be better grounded in the context.

Query: {query}

Context:
{context}

Previous Response: {response}

Issues Found: {issues}

Generate a corrected response that:
1. Only makes claims supported by the context
2. Addresses the identified issues
3. Acknowledges limitations if context is insufficient

Corrected Answer: """


# =============================================================================
# LangGraph Nodes
# =============================================================================


class AgenticRAGGraph:
    """
    LangGraph implementation of Agentic RAG.
    
    Graph Structure:
    
        [analyze_query]
              │
              ▼
        [retrieve] ◄────────────┐
              │                 │
              ▼                 │
        [grade_documents]       │
              │                 │
              ▼                 │
        [check_relevance]───────┤ (retry if no relevant docs)
              │                 │
              ▼                 │
        [generate]              │
              │                 │
              ▼                 │
        [check_grounding]───────┘ (retry if not grounded)
              │
              ▼
           [END]
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        retriever: Any,  # Your Milvus retriever
        max_iterations: int = 3,
    ):
        self.llm = llm
        self.retriever = retriever
        self.max_iterations = max_iterations
        self._graph: CompiledStateGraph | None = None
    
    @property
    def graph(self) -> CompiledStateGraph:
        """Build and cache the graph."""
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph
    
    def _build_graph(self) -> CompiledStateGraph:
        """Construct the LangGraph state machine."""
        
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("grade_documents", self._grade_documents_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("check_grounding", self._check_grounding_node)
        workflow.add_node("rewrite_query", self._rewrite_query_node)
        
        # Set entry point
        workflow.set_entry_point("analyze_query")
        
        # Add edges
        workflow.add_edge("analyze_query", "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        
        # Conditional edge after grading documents
        workflow.add_conditional_edges(
            "grade_documents",
            self._decide_after_grading,
            {
                "generate": "generate",
                "rewrite": "rewrite_query",
                "end": END,
            }
        )
        
        workflow.add_edge("rewrite_query", "retrieve")
        workflow.add_edge("generate", "check_grounding")
        
        # Conditional edge after grounding check
        workflow.add_conditional_edges(
            "check_grounding",
            self._decide_after_grounding,
            {
                "end": END,
                "retry": "rewrite_query",
            }
        )
        
        return workflow.compile()
    
    # -------------------------------------------------------------------------
    # Node Implementations
    # -------------------------------------------------------------------------
    
    async def _analyze_query_node(self, state: RAGState) -> dict:
        """Analyze and classify the query."""
        logger.info(f"Analyzing query: {state['query']}")
        
        prompt = ChatPromptTemplate.from_template(QUERY_ROUTER_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        
        result = await chain.ainvoke({"query": state["query"]})
        
        # Parse result
        query_type = "simple"
        search_query = state["query"]
        sub_queries = []
        
        for line in result.strip().split("\n"):
            if line.startswith("TYPE:"):
                query_type = line.replace("TYPE:", "").strip().lower()
            elif line.startswith("SEARCH_QUERY:"):
                search_query = line.replace("SEARCH_QUERY:", "").strip()
            elif line.startswith("SUB_QUERIES:"):
                sq = line.replace("SUB_QUERIES:", "").strip()
                if sq.lower() != "none":
                    sub_queries = [q.strip() for q in sq.split(",")]
        
        return {
            "query_type": query_type,
            "reformulated_query": search_query,
            "sub_queries": sub_queries,
            "iteration": 1,
            "max_iterations": self.max_iterations,
            "retrieval_strategy": "semantic",
        }
    
    async def _retrieve_node(self, state: RAGState) -> dict:
        """Retrieve documents based on query."""
        query = state.get("reformulated_query") or state["query"]
        strategy = state.get("retrieval_strategy", "semantic")
        
        logger.info(f"Retrieving with strategy={strategy}, query={query[:50]}...")
        
        # Use the retriever (assumes it returns Document objects)
        documents = await self._do_retrieval(query, strategy)
        
        return {"documents": documents}
    
    async def _do_retrieval(
        self,
        query: str,
        strategy: str,
    ) -> list[Document]:
        """Perform the actual retrieval."""
        # This wraps your Milvus retriever
        # Adapt based on your actual retriever interface
        
        if hasattr(self.retriever, "aget_relevant_documents"):
            docs = await self.retriever.aget_relevant_documents(query)
        elif hasattr(self.retriever, "search"):
            # For MilvusClientWrapper
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
            query_vec = embedder.encode(query).tolist()
            
            results = self.retriever.search(
                query_vector=query_vec,
                top_k=5,
                output_fields=["text", "subject", "sender"],
            )
            
            docs = [
                Document(
                    page_content=r.get("entity", {}).get("text", ""),
                    metadata={
                        "source": r.get("entity", {}).get("subject", ""),
                        "sender": r.get("entity", {}).get("sender", ""),
                        "score": 1 - r.get("distance", 0),
                    }
                )
                for r in results
            ]
        else:
            docs = []
        
        return docs
    
    async def _grade_documents_node(self, state: RAGState) -> dict:
        """Grade document relevance."""
        logger.info(f"Grading {len(state['documents'])} documents")
        
        relevant_docs = []
        
        for doc in state["documents"]:
            prompt = ChatPromptTemplate.from_template(DOCUMENT_GRADER_PROMPT)
            chain = prompt | self.llm | StrOutputParser()
            
            grade = await chain.ainvoke({
                "query": state["query"],
                "document": doc.page_content[:1000],
            })
            
            if "relevant" in grade.lower() and "not" not in grade.lower():
                relevant_docs.append(doc)
        
        logger.info(f"Relevant documents: {len(relevant_docs)}/{len(state['documents'])}")
        
        return {"relevant_documents": relevant_docs}
    
    async def _generate_node(self, state: RAGState) -> dict:
        """Generate response from relevant documents."""
        docs = state["relevant_documents"]
        
        if not docs:
            return {"response": "I don't have enough information to answer that question."}
        
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        
        # Check if we're correcting a previous response
        if state.get("response") and state.get("issues"):
            prompt = ChatPromptTemplate.from_template(RESPONSE_CORRECTOR_PROMPT)
            response = await (prompt | self.llm | StrOutputParser()).ainvoke({
                "query": state["query"],
                "context": context,
                "response": state["response"],
                "issues": state["issues"],
            })
        else:
            prompt = ChatPromptTemplate.from_template(RESPONSE_GENERATOR_PROMPT)
            response = await (prompt | self.llm | StrOutputParser()).ainvoke({
                "query": state["query"],
                "context": context,
            })
        
        return {"response": response.strip()}
    
    async def _check_grounding_node(self, state: RAGState) -> dict:
        """Check if response is grounded in documents."""
        docs = state["relevant_documents"]
        response = state["response"]
        
        if not docs or not response:
            return {
                "is_grounded": False,
                "confidence": 0.0,
                "issues": "No documents or response",
            }
        
        docs_text = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = ChatPromptTemplate.from_template(HALLUCINATION_GRADER_PROMPT)
        result = await (prompt | self.llm | StrOutputParser()).ainvoke({
            "documents": docs_text,
            "response": response,
        })
        
        # Parse result
        is_grounded = False
        confidence = 0.5
        issues = ""
        
        for line in result.strip().split("\n"):
            if line.startswith("GRADE:"):
                is_grounded = "grounded" in line.lower() and "not" not in line.lower()
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip())
                except ValueError:
                    pass
            elif line.startswith("ISSUES:"):
                issues = line.replace("ISSUES:", "").strip()
        
        logger.info(f"Grounding check: grounded={is_grounded}, confidence={confidence}")
        
        return {
            "is_grounded": is_grounded,
            "confidence": confidence,
            "issues": issues,
            "final_answer": response if is_grounded and confidence >= 0.8 else "",
        }
    
    async def _rewrite_query_node(self, state: RAGState) -> dict:
        """Rewrite query for better retrieval."""
        iteration = state.get("iteration", 1) + 1
        
        logger.info(f"Rewriting query (iteration {iteration})")
        
        prompt = ChatPromptTemplate.from_template(QUERY_REWRITER_PROMPT)
        new_query = await (prompt | self.llm | StrOutputParser()).ainvoke({
            "query": state["query"],
            "issues": state.get("issues", "No relevant results found"),
        })
        
        # Rotate retrieval strategy
        strategies = ["semantic", "expanded", "hybrid"]
        current_idx = strategies.index(state.get("retrieval_strategy", "semantic"))
        next_strategy = strategies[(current_idx + 1) % len(strategies)]
        
        return {
            "reformulated_query": new_query.strip(),
            "iteration": iteration,
            "retrieval_strategy": next_strategy,
        }
    
    # -------------------------------------------------------------------------
    # Conditional Edges
    # -------------------------------------------------------------------------
    
    def _decide_after_grading(self, state: RAGState) -> str:
        """Decide next step after grading documents."""
        relevant_docs = state.get("relevant_documents", [])
        iteration = state.get("iteration", 1)
        max_iter = state.get("max_iterations", 3)
        
        if relevant_docs:
            return "generate"
        elif iteration < max_iter:
            return "rewrite"
        else:
            return "end"
    
    def _decide_after_grounding(self, state: RAGState) -> str:
        """Decide next step after grounding check."""
        is_grounded = state.get("is_grounded", False)
        confidence = state.get("confidence", 0.0)
        iteration = state.get("iteration", 1)
        max_iter = state.get("max_iterations", 3)
        
        if is_grounded and confidence >= 0.8:
            return "end"
        elif iteration < max_iter:
            return "retry"
        else:
            return "end"
    
    # -------------------------------------------------------------------------
    # Public Interface
    # -------------------------------------------------------------------------
    
    async def query(self, query: str) -> dict:
        """
        Execute the agentic RAG pipeline.
        
        Args:
            query: User's question
            
        Returns:
            Final state with response and metadata
        """
        initial_state: RAGState = {
            "query": query,
            "query_type": "",
            "reformulated_query": "",
            "sub_queries": [],
            "documents": [],
            "relevant_documents": [],
            "retrieval_strategy": "semantic",
            "response": "",
            "is_grounded": False,
            "confidence": 0.0,
            "issues": "",
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "should_retry": False,
            "final_answer": "",
        }
        
        final_state = await self.graph.ainvoke(initial_state)
        
        return {
            "query": final_state["query"],
            "response": final_state.get("final_answer") or final_state.get("response", ""),
            "is_grounded": final_state.get("is_grounded", False),
            "confidence": final_state.get("confidence", 0.0),
            "iterations": final_state.get("iteration", 1),
            "documents_used": len(final_state.get("relevant_documents", [])),
        }


# =============================================================================
# Factory
# =============================================================================


def get_agentic_rag_graph(
    max_iterations: int = 3,
) -> AgenticRAGGraph:
    """Create an AgenticRAGGraph instance."""
    from dacribagents.application.agents.general_assistant import create_llm
    from dacribagents.infrastructure.milvus_client import get_milvus_client
    from dacribagents.infrastructure.settings import get_settings
    
    settings = get_settings()
    llm = create_llm(settings)
    milvus = get_milvus_client()
    
    return AgenticRAGGraph(
        llm=llm,
        retriever=milvus,
        max_iterations=max_iterations,
    )