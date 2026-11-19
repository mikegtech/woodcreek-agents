# Adding a provider-neutral IMAP mailbox

The Exxact email processor supports per-mailbox IMAP server, port, and timeout while retaining backward-compatible WorkMail settings.

First follow the [read-only IMAP smoke test](imap-provider-connectivity.md). Do not enable real ingestion until Kafka acknowledgment and post-processing folder checks are accepted.

```dotenv
EMAIL_MAILBOXES=family
EMAIL_FAMILY_EMAIL=mailbox@example.invalid
EMAIL_FAMILY_PASSWORD=<local-secret>
EMAIL_FAMILY_EMAIL_ALIAS=family@example.invalid
EMAIL_FAMILY_ACCOUNT_ID=imap-family
EMAIL_FAMILY_IMAP_HOST=imappro.zoho.com
EMAIL_FAMILY_IMAP_PORT=993
EMAIL_FAMILY_TARGET_FOLDER=family
```

Legacy `WORKMAIL_MAILBOXES` and `WORKMAIL_<NAME>_*` still work. The existing `email-worker` Docker Compose service loads local `.env`; no Compose change is required for these configuration keys. Never place passwords in Git or command-line arguments.

The provider mailbox login may be different from the aliases receiving messages. Recipient headers are not independently authoritative child identity evidence.
