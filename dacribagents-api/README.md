# Woodcreek Home Agents API

Multi-agent orchestration system for home management, built with LangGraph and FastAPI.

## Agents

See [ADR-001](../docs/ADRs/ADR-001.md) for the full agent specification.

## Development
```bash
# Install dependencies
uv sync --group dev

# Run dev server
poe dev

# Run tests
poe test

# Lint
poe lint
```

## Environment Variables

Copy `.env.example` to `.env` and configure your keys.
