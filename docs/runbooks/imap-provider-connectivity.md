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
EMAIL_MAILBOXES=family,backup
EMAIL_FAMILY_EMAIL=mailbox@example.invalid
EMAIL_FAMILY_PASSWORD=<local-secret>
EMAIL_FAMILY_EMAIL_ALIAS=family@example.invalid
EMAIL_FAMILY_ACCOUNT_ID=imap-family
EMAIL_FAMILY_FOLDER=INBOX
EMAIL_FAMILY_TARGET_FOLDER=family
EMAIL_FAMILY_IMAP_HOST=imappro.zoho.com
EMAIL_FAMILY_IMAP_PORT=993
EMAIL_FAMILY_IMAP_TIMEOUT_SECONDS=15

EMAIL_BACKUP_EMAIL=backup@example.invalid
EMAIL_BACKUP_PASSWORD=<local-secret>
EMAIL_BACKUP_IMAP_HOST=imap.other-provider.invalid
EMAIL_BACKUP_IMAP_PORT=993
```

The existing `WORKMAIL_MAILBOXES` and `WORKMAIL_<NAME>_*` variables remain supported. In the initial PR, provider-specific host and port are configurable independently for each mailbox; WorkMail remains the fallback when unspecified. Do not commit `.env` or credentials.

**Stop before live processing:** The existing worker moves messages before verifying Kafka's asynchronous delivery outcome, and the destination folder is currently derived from login local-part instead of the configured alias. Both must be corrected and reviewed before enabling live family-record ingestion. The smoke test does not certify child routing or evidence custody.

Sources: [Zoho IMAP account settings](https://www.zoho.com/mail/help/imap-access.html).
