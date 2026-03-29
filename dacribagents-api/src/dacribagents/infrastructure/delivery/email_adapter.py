"""Email delivery adapter — implements DeliveryChannelAdapter for email.

Uses ``smtplib`` via AWS WorkMail SMTP (or any standard SMTP endpoint).
Aligned with the existing WorkMail IMAP ingest pattern in the repo.
"""

from __future__ import annotations

import smtplib
from email.message import EmailMessage
from uuid import UUID, uuid4

from loguru import logger

from dacribagents.application.ports.delivery_channel import DeliveryResult
from dacribagents.domain.reminders.enums import DeliveryChannel, DeliveryStatus


class SmtpEmailAdapter:
    """DeliveryChannelAdapter implementation backed by SMTP."""

    def __init__(
        self,
        smtp_host: str = "smtp.mail.us-east-1.awsapps.com",
        smtp_port: int = 465,
        smtp_user: str = "",
        smtp_password: str = "",
        from_address: str = "reminders@woodcreek.me",
    ) -> None:
        self._host = smtp_host
        self._port = smtp_port
        self._user = smtp_user
        self._password = smtp_password
        self._from = from_address

    @property
    def channel(self) -> DeliveryChannel:  # noqa: D102
        return DeliveryChannel.EMAIL

    def send(  # noqa: PLR0913
        self,
        *,
        recipient_id: UUID,
        recipient_address: str,
        subject: str,
        body: str,
        reminder_id: UUID,
        urgency: str,
        metadata: dict[str, str] | None = None,
    ) -> DeliveryResult:
        """Send a reminder email via SMTP."""
        message_id = f"<reminder-{reminder_id}-{uuid4().hex[:8]}@woodcreek.me>"

        msg = EmailMessage()
        msg["Subject"] = _format_subject(subject, urgency)
        msg["From"] = self._from
        msg["To"] = recipient_address
        msg["Message-ID"] = message_id
        msg["X-Woodcreek-Reminder-ID"] = str(reminder_id)

        html = _format_email_html(subject, body, urgency, str(reminder_id))
        msg.set_content(_format_email_text(subject, body, urgency))
        msg.add_alternative(html, subtype="html")

        try:
            with smtplib.SMTP_SSL(self._host, self._port, timeout=30) as smtp:
                smtp.set_debuglevel(0)
                if self._user:
                    logger.debug(f"SMTP login as {self._user} to {self._host}:{self._port}")
                    smtp.login(self._user, self._password)
                result = smtp.send_message(msg)
                logger.info(
                    f"Email sent to {recipient_address}, message_id={message_id}, "
                    f"smtp_result={result}, from={self._from}"
                )

            return DeliveryResult(
                status=DeliveryStatus.DELIVERED,
                provider_message_id=message_id,
            )

        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP auth failed: {e}")
            return DeliveryResult(status=DeliveryStatus.FAILED, failure_reason="SMTP authentication failed")
        except smtplib.SMTPRecipientsRefused as e:
            logger.error(f"Recipient refused: {e}")
            return DeliveryResult(status=DeliveryStatus.BOUNCED, failure_reason=str(e))
        except Exception as e:
            logger.error(f"Email delivery failed: {e}")
            return DeliveryResult(status=DeliveryStatus.FAILED, failure_reason=str(e))


def _format_subject(subject: str, urgency: str) -> str:
    if urgency in {"critical", "urgent"}:
        return f"[{urgency.upper()}] {subject}"
    return subject


def _format_email_text(subject: str, body: str, urgency: str) -> str:
    lines = [subject, ""]
    if body:
        lines.append(body)
        lines.append("")
    lines.append("---")
    lines.append("Woodcreek Household Reminders")
    return "\n".join(lines)


def _format_email_html(subject: str, body: str, urgency: str, reminder_id: str) -> str:
    urgency_color = "#dc3545" if urgency in {"critical", "urgent"} else "#333"
    return f"""<html><body style="font-family: sans-serif; max-width: 600px; margin: 0 auto;">
<h2 style="color: {urgency_color};">{subject}</h2>
{f'<p>{body}</p>' if body else ''}
<hr style="border: 1px solid #eee;">
<p style="color: #888; font-size: 12px;">
Woodcreek Household Reminders<br>
Reminder ID: {reminder_id}
</p>
</body></html>"""
