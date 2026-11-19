"""Offline, synthetic tests; never contact a real mailbox."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dacribagents.infrastructure.email.providers.workmail_imap.client import (
    WorkMailImapConfig,
    WorkMailImapEmailSource,
)
from dacribagents.cli.email_worker_lattice import create_mailboxes_from_env


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


def test_legacy_mailbox_account_id_retained():
    with patch.dict("os.environ", {
        "WORKMAIL_MAILBOXES": "agents",
        "WORKMAIL_AGENTS_EMAIL": "agents@example.invalid",
        "WORKMAIL_AGENTS_PASSWORD": "synthetic",
    }, clear=True):
        result = create_mailboxes_from_env()

    assert result[0].account_id == "workmail-agents"
