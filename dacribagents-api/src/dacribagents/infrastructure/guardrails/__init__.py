"""
NeMo Guardrails infrastructure for Woodcreek Agents.

This module provides guardrails that ensure:
- Safe, on-topic conversations
- Input validation (jailbreak prevention, toxicity filtering)
- Output validation (hallucination checking, harmful content blocking)
- Topic-specific rails for HOA, Solar, and General agents
"""

from dacribagents.infrastructure.guardrails.rails import (
    GuardrailedAgent,
    GuardrailsMiddleware,
    get_guarded_general_assistant,
    get_guarded_hoa_agent,
    get_guarded_solar_agent,
)

__all__ = [
    "GuardrailedAgent",
    "GuardrailsMiddleware",
    "get_guarded_general_assistant",
    "get_guarded_hoa_agent",
    "get_guarded_solar_agent",
]
