"""Offline, synthetic tests; never contact a real mailbox."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dacribagents.application.ports.email_source import RawEmail
from dacribagents.cli.email_worker_lattice import (
    EmailWorkerLattice,
    MailboxConfig,
    _check_recipient_match_static,
    create_mailboxes_from_env,
)
from dacribagents.infrastructure.email.providers.workmail_imap.client import (
    WorkMailImapConfig,
    WorkMailImapEmailSource,
)


def test_default_workmail_config_unchanged():
    cfg = WorkMailImapConfig(region="us-east-1", email="user@example.invalid", password="synthetic")
    assert cfg.host == "imap.mail.us-east-1.awsapps.com"
    assert cfg.port == 993


def test_configured_zoho_host_uses_verified_tls():
    cfg = WorkMailImapConfig(
        region="us-east-1", email="user@example.invalid", password="synthetic",
        host="imappro.zoho.com", port=993, timeout_seconds=12,
    )
    mock_conn = MagicMock()
    with (
        patch("dacribagents.infrastructure.email.providers.workmail_imap.client.imaplib.IMAP4_SSL",
              return_value=mock_conn) as open_mock,
        patch("dacribagents.infrastructure.email.providers.workmail_imap.client.ssl.create_default_context") as tls_mock,
    ):
        source = WorkMailImapEmailSource(cfg)
        assert source._connect() is mock_conn
        assert open_mock.call_args.args == ("imappro.zoho.com", 993)
        assert open_mock.call_args.kwargs["timeout"] == 12
        assert open_mock.call_args.kwargs["ssl_context"] is tls_mock.return_value
        mock_conn.login.assert_called_once_with("user@example.invalid", "synthetic")


@pytest.mark.parametrize("port", [-1, 0, 65536])
def test_invalid_imap_ports_rejected(port: int):
    with pytest.raises(ValueError):
        WorkMailImapConfig(
            region="us-east-1", email="user@example.invalid", password="synthetic",
            host="imappro.zoho.com", port=port,
        )


def test_provider_neutral_and_legacy_mailboxes(monkeypatch):
    with patch.dict("os.environ", {
        "EMAIL_MAILBOXES": "family,backup",
        "EMAIL_FAMILY_EMAIL": "mailbox@example.invalid",
        "EMAIL_FAMILY_PASSWORD": "synthetic",
        "EMAIL_FAMILY_EMAIL_ALIAS": "family@example.invalid",
        "EMAIL_FAMILY_IMAP_HOST": "imappro.zoho.com",
        "EMAIL_FAMILY_IMAP_PORT": "993",
        "EMAIL_FAMILY_TARGET_FOLDER": "family",
        "WORKMAIL_BACKUP_EMAIL": "legacy@example.invalid",
        "WORKMAIL_BACKUP_PASSWORD": "synthetic",
        "WORKMAIL_BACKUP_ACCOUNT_ID": "existing-identity",
    }, clear=True):
        result = create_mailboxes_from_env()

    assert len(result) == 2
    assert result[0].imap_host == "imappro.zoho.com"
    assert result[0].imap_port == 993
    assert result[0].email_alias == "family@example.invalid"
    assert result[0].target_folder == "family"
    assert result[1].imap_host == "imap.mail.us-east-1.awsapps.com"
    assert result[1].account_id == "existing-identity"


def test_workmail_and_zoho_mailboxes_are_combined():
    with patch.dict("os.environ", {
        "WORKMAIL_MAILBOXES": "agents",
        "WORKMAIL_AGENTS_EMAIL": "agents@example.invalid",
        "WORKMAIL_AGENTS_PASSWORD": "workmail-secret",
        "ZOHO_LOGAN_ROCKWALL_EMAIL": "login@example.invalid",
        "ZOHO_LOGAN_ROCKWALL_PASSWORD": "zoho-secret",
        "ZOHO_LOGAN_ROCKWALL_EMAIL_ALIAS": "logan@woodcreek.me",
    }, clear=True):
        result = create_mailboxes_from_env()

    assert [(mailbox.name, mailbox.imap_host) for mailbox in result] == [
        ("agents", "imap.mail.us-east-1.awsapps.com"),
        ("logan-rockwall", "imappro.zoho.com"),
    ]
    assert result[0].account_id == "workmail-agents"
    assert result[1].account_id == "zoho-logan-rockwall"
    assert result[1].email_alias == "logan@woodcreek.me"


def test_configured_target_folder_is_used():
    cfg = WorkMailImapConfig(
        region="us-east-1",
        email="login@example.invalid",
        password="synthetic",
        destination_folder_override="Processed",
    )

    assert cfg.target_folder == "Processed"


@pytest.mark.parametrize("header", ["To", "Cc"])
def test_recipient_match_handles_multiple_addresses(header):
    raw = (
        f"To: primary@example.invalid\r\n"
        f"{header}: First <first@example.invalid>, Alias <logan@woodcreek.me>\r\n"
        "\r\nbody"
    ).encode()

    assert _check_recipient_match_static(raw, "logan@woodcreek.me")


@pytest.mark.parametrize(
    ("remaining", "delivery_errors"),
    [
        (1, []),
        (0, ["broker unavailable"]),
    ],
)
def test_mailbox_is_not_modified_without_confirmed_delivery(remaining, delivery_errors):
    publisher = MagicMock()
    publisher.publish.return_value = "message-id"
    publisher.flush.return_value = remaining
    publisher.get_errors.return_value = delivery_errors
    source = MagicMock()
    source.fetch_unprocessed.return_value = [
        (
            42,
            RawEmail(
                provider="imap",
                account="login@example.invalid",
                folder="INBOX",
                uid=42,
                rfc822_bytes=b"To: logan@woodcreek.me\r\nSubject: test\r\n\r\nbody",
            ),
        )
    ]

    with patch(
        "dacribagents.cli.email_worker_lattice.WorkMailImapEmailSource",
        return_value=source,
    ):
        worker = EmailWorkerLattice(mailboxes=[], publisher=publisher)
        result = worker._process_mailbox(MailboxConfig(
            name="logan-rockwall",
            email="login@example.invalid",
            password="synthetic",
            email_alias="logan@woodcreek.me",
            account_id="zoho-logan-rockwall",
            target_folder="logan-rockwall",
        ))

    assert result == (0, 0, 1)
    source.mark_processed.assert_not_called()
    source.disconnect.assert_called_once_with()


def test_mailbox_is_modified_after_confirmed_delivery():
    publisher = MagicMock()
    publisher.publish.return_value = "message-id"
    publisher.flush.return_value = 0
    publisher.get_errors.return_value = []
    source = MagicMock()
    source.fetch_unprocessed.return_value = [
        (
            42,
            RawEmail(
                provider="imap",
                account="login@example.invalid",
                folder="INBOX",
                uid=42,
                rfc822_bytes=b"To: logan@woodcreek.me\r\nSubject: test\r\n\r\nbody",
            ),
        )
    ]
    source.mark_processed.return_value = True

    with patch(
        "dacribagents.cli.email_worker_lattice.WorkMailImapEmailSource",
        return_value=source,
    ):
        worker = EmailWorkerLattice(mailboxes=[], publisher=publisher)
        result = worker._process_mailbox(MailboxConfig(
            name="logan-rockwall",
            email="login@example.invalid",
            password="synthetic",
            email_alias="logan@woodcreek.me",
            account_id="zoho-logan-rockwall",
            target_folder="logan-rockwall",
        ))

    assert result == (1, 0, 0)
    source.mark_processed.assert_called_once_with(42)
    source.disconnect.assert_called_once_with()


def test_legacy_mailbox_account_id_retained():
    with patch.dict("os.environ", {
        "WORKMAIL_MAILBOXES": "agents",
        "WORKMAIL_AGENTS_EMAIL": "agents@example.invalid",
        "WORKMAIL_AGENTS_PASSWORD": "synthetic",
    }, clear=True):
        result = create_mailboxes_from_env()

    assert result[0].account_id == "workmail-agents"
