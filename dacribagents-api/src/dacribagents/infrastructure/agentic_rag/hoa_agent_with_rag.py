"""
HOA Compliance Agent with Agentic RAG.

This agent uses the agentic RAG pipeline for:
- Query reformulation for HOA-specific terminology
- Self-correction when responses aren't grounded in CC&Rs
- Multi-hop reasoning for complex compliance questions
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from dacribagents.application.agents.base import AgentState, BaseAgent
from dacribagents.domain.models import AgentType, Message, MessageRole


HOA_SYSTEM_PROMPT = """You are an HOA Compliance specialist for the Woodcreek community in Fate, Texas.

Your expertise includes:
- CC&Rs (Covenants, Conditions & Restrictions)
- ARC (Architectural Review Committee) processes
- HOA violations and appeals
- Community rules and regulations
- TownSq platform guidance

Important guidelines:
- Base your answers on the retrieved HOA documents and CC&Rs
- Always cite the relevant section when possible
- If information isn't in your knowledge base, say so clearly
- Recommend contacting the HOA board for official rulings
- Never provide legal advice

Current context:
- Community: Woodcreek (Fate, TX)
- HOA Management: TownSq platform
- Documents available: CC&Rs, ARC guidelines, community rules
"""


class HOAComplianceAgent(BaseAgent):
    """
    HOA Compliance agent with Agentic RAG capabilities.
    
    Features:
    - RAG over HOA documents (CC&Rs, rules, guidelines)
    - Query reformulation for HOA terminology
    - Self-correction for grounded responses
    - Guardrails integration
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        use_agentic_rag: bool = True,
        use_guardrails: bool = True,
    ):
        super().__init__(AgentType.HOA_COMPLIANCE)
        self.llm = llm
        self.use_agentic_rag = use_agentic_rag
        self.use_guardrails = use_guardrails
        self._rag_pipeline = None
        self._guardrails = None
    
    @property
    def rag_pipeline(self):
        """Lazy load the agentic RAG pipeline."""
        if self._rag_pipeline is None and self.use_agentic_rag:
            try:
                from dacribagents.infrastructure.agentic_rag import get_agentic_rag
                self._rag_pipeline = get_agentic_rag(
                    collection_name="email_chunks_v1",
                    filter_expr='account_id == "workmail-hoa"',
                    max_iterations=3,
                )
                logger.info("Agentic RAG pipeline loaded for HOA agent (filtered by workmail-hoa)")
            except Exception as e:
                logger.warning(f"Could not load agentic RAG: {e}")
        return self._rag_pipeline
    
    @property
    def guardrails(self):
        """Lazy load guardrails."""
        if self._guardrails is None and self.use_guardrails:
            try:
                from dacribagents.infrastructure.guardrails import GuardrailsMiddleware
                self._guardrails = GuardrailsMiddleware(agent_type="hoa")
                logger.info("Guardrails loaded for HOA agent")
            except Exception as e:
                logger.warning(f"Could not load guardrails: {e}")
        return self._guardrails
    
    def _build_graph(self) -> CompiledStateGraph:
        """Build the HOA agent graph."""
        
        async def process_with_rag(state: dict) -> dict:
            """Process query using agentic RAG."""
            messages = state["messages"]
            
            # Get the user's question
            user_message = ""
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    user_message = msg.content
                    break
            
            if not user_message:
                return {"messages": messages}
            
            # Check input guardrails
            if self.guardrails:
                allowed, reason = await self.guardrails.check_input(user_message)
                if not allowed:
                    from langchain_core.messages import AIMessage
                    return {"messages": messages + [AIMessage(content=reason)]}
            
            # Use agentic RAG if available
            response_content = ""
            rag_metadata = {}
            
            if self.rag_pipeline:
                try:
                    rag_result = await self.rag_pipeline.query(user_message)
                    response_content = rag_result.response
                    rag_metadata = {
                        "is_grounded": rag_result.is_grounded,
                        "confidence": rag_result.confidence,
                        "iterations": rag_result.iterations,
                        "sources": [c.source for c in rag_result.retrieved_chunks[:3]],
                    }
                except Exception as e:
                    logger.error(f"RAG error: {e}")
            
            # Fall back to direct LLM if RAG failed
            if not response_content:
                all_messages = [SystemMessage(content=HOA_SYSTEM_PROMPT)] + list(messages)
                response = await self.llm.ainvoke(all_messages)
                response_content = response.content
            
            # Check output guardrails
            if self.guardrails:
                allowed, filtered = await self.guardrails.check_output(response_content)
                if not allowed:
                    response_content = filtered
            
            from langchain_core.messages import AIMessage
            response_msg = AIMessage(
                content=response_content,
                additional_kwargs={"rag_metadata": rag_metadata},
            )
            
            return {"messages": messages + [response_msg]}
        
        # Build graph
        workflow = StateGraph(AgentState)
        workflow.add_node("process", process_with_rag)
        workflow.set_entry_point("process")
        workflow.add_edge("process", END)
        
        return workflow.compile()
    
    async def process(
        self,
        messages: list[Message],
        context: dict[str, Any] | None = None,
    ) -> Message:
        """Process messages and return response."""
        # Convert to LangChain messages
        lc_messages = self._convert_to_langchain_messages(messages)
        
        initial_state: AgentState = {
            "messages": lc_messages,
            "context": context or {},
            "agent_type": self.agent_type.value,
            "thread_id": None,
        }
        
        result = await self.graph.ainvoke(initial_state)
        
        # Extract response
        response_messages = result["messages"]
        if response_messages:
            last_msg = response_messages[-1]
            return self._convert_from_langchain_message(last_msg)
        
        return Message(
            role=MessageRole.ASSISTANT,
            content="I apologize, but I couldn't process your HOA question. Please try again.",
        )
    
    @property
    def model_info(self) -> dict[str, str]:
        """Get model info including RAG status."""
        return {
            "agent_type": "hoa_compliance",
            "agentic_rag": "enabled" if self.rag_pipeline else "disabled",
            "guardrails": "enabled" if self.guardrails else "disabled",
        }


# =============================================================================
# Factory
# =============================================================================


_hoa_agent: HOAComplianceAgent | None = None


def get_hoa_agent(
    use_agentic_rag: bool = True,
    use_guardrails: bool = True,
) -> HOAComplianceAgent:
    """Get or create the HOA Compliance agent."""
    global _hoa_agent
    
    if _hoa_agent is None:
        from dacribagents.application.agents.general_assistant import create_llm
        from dacribagents.infrastructure.settings import get_settings
        
        settings = get_settings()
        llm = create_llm(settings)
        
        _hoa_agent = HOAComplianceAgent(
            llm=llm,
            use_agentic_rag=use_agentic_rag,
            use_guardrails=use_guardrails,
        )
    
    return _hoa_agent