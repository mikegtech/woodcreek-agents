from __future__ import annotations

import os
import time

from dacribagents.cli.ingest_once import main as ingest_once_main


def main() -> int:
    poll = int(os.getenv("EMAIL_POLL_SECONDS", "60"))

    while True:
        try:
            ingest_once_main()
        except Exception as e:
            # replace with structured logging later
            print(f"[worker] ingest failed: {e}")

        time.sleep(poll)


if __name__ == "__main__":
    raise SystemExit(main())
