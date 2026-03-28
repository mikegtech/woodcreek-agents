"""Subsystem health/readiness endpoint.

Checks critical dependencies and reports their status.
Does not leak secrets or sensitive configuration.
"""

from __future__ import annotations

import os

from fastapi import APIRouter

router = APIRouter()


@router.get("/health/subsystems")
async def subsystem_health() -> dict:
    """Check readiness of all reminder platform subsystems."""
    checks: dict[str, dict] = {}

    # PostgreSQL
    checks["postgresql"] = _check_postgres()

    # Kafka
    checks["kafka"] = _check_kafka_config()

    # Slack
    checks["slack"] = _check_slack_config()

    # Telnyx SMS
    checks["telnyx"] = _check_telnyx_config()

    # Email/SMTP
    checks["smtp"] = _check_smtp_config()

    # Governance
    checks["governance"] = _check_governance()

    all_healthy = all(c.get("status") == "healthy" for c in checks.values())
    return {
        "status": "healthy" if all_healthy else "degraded",
        "subsystems": checks,
    }


def _check_postgres() -> dict:
    try:
        from dacribagents.infrastructure import get_postgres_client  # noqa: PLC0415

        client = get_postgres_client()
        result = client.health_check()
        return {"status": result.get("status", "unknown")}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def _check_kafka_config() -> dict:
    brokers = os.getenv("KAFKA_BROKERS", "")
    if not brokers:
        return {"status": "not_configured"}
    return {"status": "healthy", "brokers_configured": True}


def _check_slack_config() -> dict:
    token = bool(os.getenv("SLACK_BOT_TOKEN", ""))
    secret = bool(os.getenv("SLACK_SIGNING_SECRET", ""))
    if token and secret:
        return {"status": "healthy"}
    return {"status": "not_configured", "token": token, "signing_secret": secret}


def _check_telnyx_config() -> dict:
    key = bool(os.getenv("TELNYX_API_KEY", ""))
    phone = bool(os.getenv("TELNYX_FROM_NUMBER", ""))
    if key and phone:
        return {"status": "healthy"}
    return {"status": "not_configured", "api_key": key, "from_number": phone}


def _check_smtp_config() -> dict:
    user = bool(os.getenv("SMTP_USER", ""))
    if user:
        return {"status": "healthy"}
    return {"status": "not_configured"}


def _check_governance() -> dict:
    try:
        from dacribagents.application.services.governance import (  # noqa: PLC0415
            get_governance_state,
            is_kill_switch_active,
        )

        state = get_governance_state()
        return {
            "status": "healthy",
            "kill_switch_active": is_kill_switch_active(),
            "households_configured": len(state.household_tiers),
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
