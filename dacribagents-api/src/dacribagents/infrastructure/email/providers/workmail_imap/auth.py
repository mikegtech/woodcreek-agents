from __future__ import annotations
from dataclasses import dataclass
import imaplib

WORKMAIL_IMAP_HOST = "imap.mail.us-east-1.awsapps.com"
WORKMAIL_IMAP_PORT = 993


@dataclass(frozen=True)
class WorkMailImapCredentials:
    """
    Represents credentials for a single WorkMail mailbox.
    """
    email: str
    password: str


class WorkMailImapAuthenticator:
    """
    Responsible ONLY for establishing an authenticated IMAP connection
    to AWS WorkMail. No folder logic, no fetching, no parsing.
    """

    def __init__(self, creds: WorkMailImapCredentials) -> None:
        self.creds = creds

    def login(self) -> imaplib.IMAP4_SSL:
        """
        Returns an authenticated IMAP4_SSL connection.
        WorkMail requires IMAPS (port 993) and does NOT support STARTTLS.
        """
        conn = imaplib.IMAP4_SSL(
            host=WORKMAIL_IMAP_HOST,
            port=WORKMAIL_IMAP_PORT,
        )
        conn.login(self.creds.email, self.creds.password)
        return conn
