from __future__ import annotations
from email import policy
from email.parser import BytesParser
from datetime import datetime, timezone
from dacribagents.domain.entities.email_message import EmailMessage

def _as_text(msg) -> str:
    # Prefer text/plain; fallback to stripped HTML
    if msg.is_multipart():
        parts = msg.walk()
        for p in parts:
            ctype = p.get_content_type()
            if ctype == "text/plain":
                return p.get_content().strip()
        for p in msg.walk():
            if p.get_content_type() == "text/html":
                html = p.get_content()
                return html  # normalize later (html->text) if you want
    else:
        return msg.get_content().strip()
    return ""

def rfc822_to_email_message(provider: str, account: str, folder: str, rfc822_bytes: bytes) -> EmailMessage:
    em = BytesParser(policy=policy.default).parsebytes(rfc822_bytes)

    message_id = (em.get("Message-Id") or "").strip()
    subject = (em.get("Subject") or "").strip()
    sender = (em.get("From") or "").strip()
    to = [x.strip() for x in (em.get_all("To") or [])]
    cc = [x.strip() for x in (em.get_all("Cc") or [])]

    # Date parsing can be messy; default to now if absent/unparseable
    dt = em.get("Date")
    try:
        date = dt.datetime if dt else datetime.now(timezone.utc)
    except Exception:
        date = datetime.now(timezone.utc)

    text = _as_text(em)

    # Optional thread hints (not guaranteed via IMAP)
    thread_id = (em.get("Thread-Index") or em.get("In-Reply-To") or None)

    return EmailMessage(
        provider=provider,
        account=account,
        folder=folder,
        message_id=message_id,
        thread_id=thread_id,
        subject=subject,
        sender=sender,
        to=to,
        cc=cc,
        date=date,
        text=text,
        metadata={"content_type": em.get_content_type()},
    )
