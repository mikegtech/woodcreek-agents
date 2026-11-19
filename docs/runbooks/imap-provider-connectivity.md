# Provider-neutral IMAP connection check (Exxact)

This runbook verifies access without fetching email bodies, moving messages, publishing Kafka events, or modifying mailbox state. Run it locally on the designated Exxact processing host.

For paid Zoho Mail organization accounts, the published default is `imappro.zoho.com:993` with verified TLS. The authoritative hostname for any specific mailbox is shown under the provider's account server settings; do not assume a generic hostname when it differs. Enable IMAP in the mailbox and organization policies. Use an app-specific password if required by the account's authentication settings.

**Read-only smoke test:**

```bash
python3 dacribagents-api/scripts/check_imap_readonly.py \
  --host imappro.zoho.com --port 993 --user mailbox@example.invalid --folder INBOX
```

Replace the example username with the provider-authorized primary mailbox login. An alias may receive messages but not necessarily be a valid authentication identity. Enter the app password only at the hidden prompt, never on the command line, in a chat, or in a repository file.

The test reports TLS/auth success, a read-only mailbox count, and optional folder-existence status. It does not retrieve email content. Do not include mailbox names, credentials or message content in shared acceptance logs.

**Provider-neutral worker configuration (for future activation):**

```dotenv
ZOHO_MAILBOXES=family
ZOHO_FAMILY_EMAIL=mailbox@example.invalid
ZOHO_FAMILY_PASSWORD=<local-secret>
ZOHO_FAMILY_EMAIL_ALIAS=family@example.invalid
ZOHO_FAMILY_ACCOUNT_ID=imap-family
ZOHO_FAMILY_FOLDER=INBOX
ZOHO_FAMILY_TARGET_FOLDER=family
ZOHO_FAMILY_IMAP_HOST=imappro.zoho.com
ZOHO_FAMILY_IMAP_PORT=993
ZOHO_FAMILY_IMAP_TIMEOUT_SECONDS=15

WORKMAIL_MAILBOXES=backup
WORKMAIL_BACKUP_EMAIL=backup@example.invalid
WORKMAIL_BACKUP_PASSWORD=<local-secret>
```

The `WORKMAIL_MAILBOXES` and `ZOHO_MAILBOXES` lists are combined. Provider-specific host and port are configurable independently for each mailbox; Zoho defaults to `imappro.zoho.com:993` while WorkMail retains its existing endpoint. Do not commit `.env` or credentials.

The worker now waits for Kafka's asynchronous delivery callback before moving messages and honors the configured target folder. The smoke test still does not certify child routing or evidence custody.

Sources: [Zoho IMAP account settings](https://www.zoho.com/mail/help/imap-access.html).
