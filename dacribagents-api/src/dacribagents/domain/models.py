"""Domain models for Woodcreek Agents."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class AgentType(str, Enum):
    """Types of agents in the system."""

    HOA_COMPLIANCE = "hoa_compliance"
    SOLAR = "solar"
    HOME_MAINTENANCE = "home_maintenance"
    SECURITY_CAMERAS = "security_cameras"
    SUPERVISOR = "supervisor"
    GENERAL_ASSISTANT = "general_assistant"


class MessageRole(str, Enum):
    """Message roles in conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Message(BaseModel):
    """A message in the conversation."""

    id: UUID = Field(default_factory=uuid4)
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationThread(BaseModel):
    """A conversation thread with an agent."""

    thread_id: str
    agent_type: AgentType
    user_id: str | None = None
    messages: list[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentState(BaseModel):
    """State passed through the agent graph."""

    messages: list[dict[str, Any]] = Field(default_factory=list)
    current_agent: AgentType | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    should_continue: bool = True


class Document(BaseModel):
    """A document for RAG indexing."""

    id: UUID = Field(default_factory=uuid4)
    content: str
    source: str
    doc_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SearchResult(BaseModel):
    """Result from vector similarity search."""

    document_id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)