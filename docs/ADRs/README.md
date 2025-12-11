woodcreek-agents/
├─ apps/
│  ├─ cli/                      # Local CLI for agent runs & demos
│  └─ web/                      # Optional dashboard (Next.js)
├─ packages/
│  ├─ agents/                   # Agent implementations
│  │  ├─ hoa-compliance/
│  │  ├─ acc/
│  │  ├─ amenities-events/
│  │  ├─ utilities-outage/
│  │  ├─ maintenance-landscape/
│  │  ├─ pets-dogpark/
│  │  ├─ safety-emergency/
│  │  ├─ dues-budget/
│  │  └─ homepro-systems/
│  ├─ shared/
│  │  ├─ prompts/               # Reusable prompt snippets & tools schemas
│  │  ├─ rules/                 # Parsed HOA rules/CC&Rs references
│  │  ├─ data/                  # Lookups (providers, phones, hours)
│  │  ├─ integrations/          # TownSq, email, calendar, drive, etc.
│  │  └─ ui/                    # React components for dashboard
├─ infra/
│  ├─ docker/                   # Images for local dev
│  ├─ bicep-terraform/          # (Optional) deployment as containers/functions
│  └─ github-actions/           # CI workflows (lint, test, build)
├─ docs/
│  ├─ adr/
│  │  └─ ADR-001.md
│  ├─ agents-overview.md
│  └─ integrations.md
├─ tests/
│  └─ e2e/
└─ package.json / pyproject.toml / Taskfile.yml