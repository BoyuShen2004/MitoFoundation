## Summary

- [ ] Describe the change in 1-3 bullets.

## Architecture / Duplication Checklist

- [ ] Existing implementation reused instead of creating parallel copy.
- [ ] Capability owner respected (single owner per capability).
- [ ] No new legacy imports (`registry.*` / `pipeline.*`).
- [ ] If introducing a new module, location matches `docs/architecture/repo-layout.md`.
- [ ] If moving paths, updated `docs/architecture/migration-map.md`.

## Validation

- [ ] Ran relevant tests.
- [ ] Ran smoke checks if pipeline behavior changed.
- [ ] Ran: `python scripts/check_architecture_guardrails.py`.

