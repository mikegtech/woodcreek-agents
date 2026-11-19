#!/usr/bin/env python3
"""Safely verify provider IMAP access without fetching, moving, or deleting mail."""

from __future__ import annotations

import argparse
import getpass
import imaplib
import ssl
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="IMAP hostname from provider's account settings")
    parser.add_argument("--port", type=int, default=993)
    parser.add_argument("--user", required=True, help="Primary mailbox login, which may differ from an alias")
    parser.add_argument("--folder", default="INBOX")
    parser.add_argument("--expect-folder", help="Check that a processed folder already exists")
    args = parser.parse_args()

    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")

    password = getpass.getpass("IMAP app password (hidden): ")
    if not password:
        print("FAIL: Missing app password", file=sys.stderr)
        return 1

    try:
        with imaplib.IMAP4_SSL(
            args.host, args.port, ssl_context=ssl.create_default_context(), timeout=15
        ) as imap:
            imap.login(args.user, password)
            print("PASS: Verified TLS and IMAP authentication")
            status, count = imap.select(args.folder, readonly=True)
            if status != "OK":
                print("FAIL: Could not select folder in read-only mode", file=sys.stderr)
                return 1
            print(f"PASS: Read-only folder selection; total messages: {count[0].decode()}")
            if args.expect_folder:
                status, folders = imap.list()
                if status != "OK":
                    print("FAIL: Cannot list folders", file=sys.stderr)
                    return 1
                # Do not log private folder names.
                names = [s.decode("utf-8", errors="replace") for s in (folders or []) if s]
                target = args.expect_folder
                if not any(
                    s.rstrip().endswith('"' + target + '"')
                    or s.rstrip().endswith(' ' + target)
                    for s in names
                ):
                    print("FAIL: Expected folder not found", file=sys.stderr)
                    return 1
                print("PASS: Expected folder present")
        return 0
    except (imaplib.IMAP4.error, ssl.SSLError, TimeoutError, OSError):
        print(
            "FAIL: IMAP login or TLS connection failed. Verify account policy, "
            "provider endpoint, and app password privately.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
