---
name: dox-enforce
description: Use BEFORE any file edit, create, or delete. Enforces DOX framework — reads the AGENTS.md chain before editing and runs closeout after.
compatibility: opencode
---

# DOX Enforcement

## Before Editing

1. Read the root `AGENTS.md`
2. Identify every file you expect to touch
3. Walk from repo root to each target path — read every `AGENTS.md` found along the route
4. If a parent `AGENTS.md` lists a child whose scope contains the target, read that child and continue from there
5. Use the nearest `AGENTS.md` as the local contract; parent docs hold repo-wide rules

Do not rely on memory. Re-read the applicable DOX chain in the current session before editing.

## After Editing

Every meaningful change requires a DOX pass:

1. Update the closest owning `AGENTS.md` when a change affects purpose, scope, contracts, workflows, constraints, or artifacts
2. Update parent docs when parent-level structure, ownership, or child index changes
3. Remove stale or contradictory text immediately
4. Small edits that don't change behavior may leave docs unchanged, but the pass still must happen

## Conflict Rule

If docs conflict, the closer doc controls local work details, but no child doc may weaken DOX.
