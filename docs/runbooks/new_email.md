# Adding a provider-neutral IMAP mailbox

The Exxact email processor supports per-mailbox IMAP server, port, and timeout while retaining backward-compatible WorkMail settings.

First follow the [read-only IMAP smoke test](imap-provider-connectivity.md). Do not enable real ingestion until Kafka acknowledgment and post-processing folder checks are accepted.

```dotenv
ZOHO_MAILBOXES=family
ZOHO_FAMILY_EMAIL=mailbox@example.invalid
ZOHO_FAMILY_PASSWORD=<local-secret>
ZOHO_FAMILY_EMAIL_ALIAS=family@example.invalid
ZOHO_FAMILY_ACCOUNT_ID=imap-family
ZOHO_FAMILY_IMAP_HOST=imappro.zoho.com
ZOHO_FAMILY_IMAP_PORT=993
ZOHO_FAMILY_TARGET_FOLDER=family
```

`ZOHO_MAILBOXES` and `WORKMAIL_MAILBOXES` are combined, so a Zoho outage does not disable WorkMail processing. Individual `ZOHO_<NAME>_EMAIL` entries also activate a mailbox without a matching `ZOHO_MAILBOXES` entry. `EMAIL_<NAME>_*` remains available for provider-neutral overrides. The existing `email-worker` Docker Compose service loads local `.env`; no Compose change is required for these configuration keys. Never place passwords in Git or command-line arguments.

The provider mailbox login may be different from the aliases receiving messages. Recipient headers are not independently authoritative child identity evidence.
