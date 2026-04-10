How to add a new mailbox now
Add the name to WORKMAIL_MAILBOXES in .env:


WORKMAIL_MAILBOXES=agents,solar,hoa,alarm,energy,gas,mortgage,water,newbox
Add the credentials in .env:


WORKMAIL_NEWBOX_EMAIL=agents@woodcreek.meWORKMAIL_NEWBOX_EMAIL_ALIAS=newbox@woodcreek.meWORKMAIL_NEWBOX_PASSWORD=<new password>
Restart: docker compose up -d email-worker

No docker-compose.yml changes needed! The Python code auto-discovers mailboxes from WORKMAIL_MAILBOXES.