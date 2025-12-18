"""Use case for processing inbound SMS messages."""

from __future__ import annotations

from loguru import logger

from dacribagents.infrastructure.sqlite.client import SQLiteClient, get_sqlite_client
from dacribagents.infrastructure.sms.provider.telnyx import TelnyxProvider, get_telnyx_provider


class ProcessSMSUseCase:
    """
    Process inbound SMS messages.

    Flow:
    1. Store inbound message in SQLite
    2. Load conversation history
    3. Call agent (TODO: for now, echo back)
    4. Send reply via Telnyx
    5. Store outbound message in SQLite
    """

    def __init__(
        self,
        sqlite_client: SQLiteClient | None = None,
        telnyx_provider: TelnyxProvider | None = None,
    ):
        self.sqlite = sqlite_client or get_sqlite_client()
        self.telnyx = telnyx_provider or get_telnyx_provider()

    async def process(
        self,
        from_number: str,
        to_number: str,
        text: str,
        provider_message_id: str,
        provider: str = "telnyx",
    ) -> dict:
        """
        Process a single inbound SMS message.

        Args:
            from_number: The sender's phone number
            to_number: Your Telnyx number that received the message
            text: The message content
            provider_message_id: Telnyx message ID
            provider: SMS provider name (default: telnyx)

        Returns:
            dict with processing result
        """
        # Normalize phone number to create conversation ID
        # Use from_number as the key (the person texting us)
        normalized = from_number.replace("+", "").replace("-", "").replace(" ", "")
        conversation_id = f"sms_{normalized}"

        logger.info(f"Processing SMS from {from_number}: {text[:50]}...")

        try:
            # 1. Store inbound message
            self.sqlite.add_message(
                conversation_id=conversation_id,
                direction="inbound",
                provider=provider,
                from_number=from_number,
                to_number=to_number,
                text=text,
                provider_message_id=provider_message_id,
                status="received",
            )
            logger.debug(f"Stored inbound message for {conversation_id}")

            # 2. Load conversation history (for future agent context)
            history = self.sqlite.get_recent_messages(conversation_id, limit=20)
            logger.debug(f"Loaded {len(history)} messages for context")

            # 3. Generate response
            # TODO: Call agent with history
            # For now, echo back the message
            response_text = self._generate_response(text, history)

            # 4. Send reply via Telnyx
            result = await self.telnyx.send_sms(
                to=from_number,
                text=response_text,
                from_=to_number,  # Reply from the number they texted
            )

            if not result.success:
                logger.error(f"Failed to send SMS reply: {result.error}")
                return {
                    "status": "error",
                    "error": result.error,
                    "conversation_id": conversation_id,
                }

            # 5. Store outbound message
            self.sqlite.add_message(
                conversation_id=conversation_id,
                direction="outbound",
                provider=provider,
                from_number=to_number,
                to_number=from_number,
                text=response_text,
                provider_message_id=result.message_id,
                status="sent",
            )

            logger.info(f"SMS reply sent to {from_number}, message_id={result.message_id}")

            return {
                "status": "processed",
                "conversation_id": conversation_id,
                "inbound_message_id": provider_message_id,
                "outbound_message_id": result.message_id,
                "reply_sent": True,
            }

        except Exception as e:
            logger.exception(f"Error processing SMS from {from_number}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "conversation_id": conversation_id,
            }

    def _generate_response(self, inbound_text: str, history: list) -> str:
        """
        Generate a response to the inbound message.

        TODO: Replace with agent call.
        For now, echo back the message.
        """
        # Simple echo for testing
        return f"Echo: {inbound_text}"

        # Future implementation:
        # messages = [
        #     {"role": "user" if m.direction == "inbound" else "assistant", "content": m.text}
        #     for m in history
        # ]
        # response = await agent.process(messages)
        # return response.content


# Convenience function for use in background tasks
async def process_sms_event(event: dict) -> dict:
    """
    Process a single SMS event from the queue.

    Args:
        event: InboundSmsEvent dict with keys:
            - from_number
            - to_number
            - text
            - telnyx_message_id
            - provider (default: telnyx)

    Returns:
        Processing result dict
    """
    use_case = ProcessSMSUseCase()
    return await use_case.process(
        from_number=event["from_number"],
        to_number=event["to_number"],
        text=event["text"],
        provider_message_id=event["telnyx_message_id"],
        provider=event.get("provider", "telnyx"),
    )