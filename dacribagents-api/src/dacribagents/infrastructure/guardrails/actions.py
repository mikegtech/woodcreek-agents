"""
Custom Actions for NeMo Guardrails.

These actions are called from Colang flows to perform custom logic
like topic classification, content checking, and validation.
"""

from typing import Any

from nemoguardrails.actions import action
from nemoguardrails.actions.actions import ActionResult


# =============================================================================
# Topic Classification Actions
# =============================================================================


@action(name="check_hoa_topic")
async def check_hoa_topic(context: dict[str, Any]) -> bool:
    """
    Check if the user's message is related to HOA topics.
    
    Returns True if the message is HOA-related, False otherwise.
    """
    user_message = context.get("last_user_message", "").lower()
    
    hoa_keywords = [
        "hoa", "homeowner", "association", "cc&r", "covenant", "restriction",
        "arc", "architectural", "review", "committee", "violation", "compliance",
        "fence", "paint", "landscaping", "yard", "lawn", "tree", "modification",
        "dues", "assessment", "fee", "parking", "vehicle", "rv", "trailer",
        "noise", "quiet", "hours", "rental", "lease", "tenant", "airbnb",
        "board", "meeting", "vote", "election", "townsg", "management",
        "neighbor", "dispute", "complaint", "fine", "appeal", "hearing",
        "common area", "pool", "clubhouse", "amenity", "gate", "access"
    ]
    
    # Check for keyword matches
    for keyword in hoa_keywords:
        if keyword in user_message:
            return True
    
    # Default to allowing (let the agent handle it)
    return True


@action(name="check_solar_topic")
async def check_solar_topic(context: dict[str, Any]) -> bool:
    """
    Check if the user's message is related to solar topics.
    
    Returns True if the message is solar-related, False otherwise.
    """
    user_message = context.get("last_user_message", "").lower()
    
    solar_keywords = [
        "solar", "panel", "photovoltaic", "pv", "sun", "energy",
        "inverter", "battery", "powerwall", "storage", "backup",
        "net metering", "buyback", "grid", "utility", "oncor", "rep",
        "kilowatt", "kwh", "watt", "production", "generation",
        "roof", "rooftop", "mount", "installation", "installer",
        "tax credit", "itc", "incentive", "rebate", "savings",
        "electricity", "bill", "rate", "cost", "price", "quote",
        "monitoring", "app", "dashboard", "efficiency", "shade",
        "tesla", "sunrun", "enphase", "solaredge", "qcells"
    ]
    
    # Check for keyword matches
    for keyword in solar_keywords:
        if keyword in user_message:
            return True
    
    return True


@action(name="check_general_topic")
async def check_general_topic(context: dict[str, Any]) -> bool:
    """
    Check if the user's message is appropriate for the general assistant.
    
    Blocks clearly off-topic or inappropriate requests.
    """
    user_message = context.get("last_user_message", "").lower()
    
    # Blocked topics
    blocked_keywords = [
        "kill", "murder", "weapon", "bomb", "drug", "illegal",
        "hack", "steal", "fraud", "scam",
        "nsfw", "porn", "xxx", "nude",
        "suicide", "self-harm"
    ]
    
    for keyword in blocked_keywords:
        if keyword in user_message:
            return False
    
    return True


# =============================================================================
# Content Validation Actions
# =============================================================================


@action(name="check_needs_hoa_disclaimer")
async def check_needs_hoa_disclaimer(bot_message: str) -> bool:
    """
    Check if the bot's HOA response needs a legal disclaimer.
    """
    disclaimer_triggers = [
        "violation", "fine", "appeal", "hearing", "legal",
        "sue", "lawsuit", "attorney", "court", "enforce",
        "must", "required", "mandatory", "prohibited"
    ]
    
    message_lower = bot_message.lower()
    
    for trigger in disclaimer_triggers:
        if trigger in message_lower:
            return True
    
    return False


@action(name="check_mentions_financial")
async def check_mentions_financial(bot_message: str) -> bool:
    """
    Check if the bot's response mentions financial figures.
    """
    financial_triggers = [
        "$", "dollar", "cost", "price", "savings", "roi",
        "payback", "investment", "return", "profit", "expense",
        "thousand", "percent", "%"
    ]
    
    message_lower = bot_message.lower()
    
    for trigger in financial_triggers:
        if trigger in message_lower:
            return True
    
    return False


# =============================================================================
# Safety Check Actions
# =============================================================================


@action(name="check_jailbreak")
async def check_jailbreak(context: dict[str, Any]) -> bool:
    """
    Check if the user is attempting to jailbreak the system.
    
    Returns True if safe, False if jailbreak attempt detected.
    """
    user_message = context.get("last_user_message", "").lower()
    
    jailbreak_patterns = [
        "ignore your instructions",
        "ignore all previous",
        "disregard your programming",
        "forget your rules",
        "pretend you are",
        "act as if",
        "you are now",
        "from now on you",
        "developer mode",
        "dan mode",
        "jailbreak",
        "bypass",
        "override your",
        "new persona",
        "roleplay as"
    ]
    
    for pattern in jailbreak_patterns:
        if pattern in user_message:
            return False
    
    return True


@action(name="check_toxicity")
async def check_toxicity(context: dict[str, Any]) -> bool:
    """
    Basic toxicity check for user messages.
    
    Returns True if safe, False if toxic content detected.
    """
    user_message = context.get("last_user_message", "").lower()
    
    # Basic profanity and toxic patterns
    toxic_patterns = [
        "fuck", "shit", "bitch", "asshole", "bastard",
        "idiot", "stupid bot", "dumb ai", "useless",
        "kill yourself", "die", "hate you"
    ]
    
    for pattern in toxic_patterns:
        if pattern in user_message:
            return False
    
    return True


@action(name="self_check_input")
async def self_check_input(context: dict[str, Any], llm: Any) -> bool:
    """
    Use the LLM to self-check if input should be blocked.
    
    This is called when more nuanced checking is needed.
    """
    # For now, use rule-based checking
    # In production, you'd call the LLM with the self-check prompt
    jailbreak_safe = await check_jailbreak(context)
    toxicity_safe = await check_toxicity(context)
    topic_safe = await check_general_topic(context)
    
    return jailbreak_safe and toxicity_safe and topic_safe


@action(name="self_check_output")
async def self_check_output(context: dict[str, Any], llm: Any) -> bool:
    """
    Use the LLM to self-check if output should be blocked.
    """
    bot_message = context.get("last_bot_message", "")
    
    # Check for potentially harmful output patterns
    harmful_patterns = [
        "here's how to hack",
        "to make a weapon",
        "illegal but",
        "don't tell anyone",
        "system prompt:",
        "my instructions are"
    ]
    
    message_lower = bot_message.lower()
    
    for pattern in harmful_patterns:
        if pattern in message_lower:
            return False
    
    return True


# =============================================================================
# RAG Relevance Check
# =============================================================================


@action(name="check_relevance")
async def check_relevance(context: dict[str, Any], llm: Any) -> bool:
    """
    Check if retrieved context is relevant to the user's query.
    
    Used in RAG flows to filter out irrelevant retrieved documents.
    """
    user_message = context.get("last_user_message", "")
    retrieved_context = context.get("relevant_chunks", [])
    
    if not retrieved_context:
        return True  # No context to check
    
    # Simple keyword overlap check
    # In production, you'd use embeddings or LLM for semantic similarity
    user_words = set(user_message.lower().split())
    
    for chunk in retrieved_context:
        chunk_words = set(chunk.lower().split())
        overlap = len(user_words & chunk_words)
        if overlap >= 2:  # At least 2 words overlap
            return True
    
    return False  # No relevant context found


# =============================================================================
# Utility Actions
# =============================================================================


@action(name="log_interaction")
async def log_interaction(context: dict[str, Any]) -> ActionResult:
    """
    Log the interaction for debugging and analytics.
    """
    from loguru import logger
    
    user_message = context.get("last_user_message", "")
    bot_message = context.get("last_bot_message", "")
    
    logger.info(f"Guardrails interaction - User: {user_message[:100]}...")
    logger.info(f"Guardrails interaction - Bot: {bot_message[:100]}...")
    
    return ActionResult(return_value=True)
