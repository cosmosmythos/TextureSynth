---
name: beauty-coder
description: Use when writing, editing, refactoring, implementing, or reviewing ANY code across the project. Produce expert-level code that is readable, expressive, and learner-friendly — never cryptic or jargon-heavy. Preserves existing functionality and behavior unless explicitly instructed otherwise.
compatibility: opencode
---


## What I do

- Write code that reads like prose to another human.
- Preserve existing functionality and behavior unless explicitly instructed otherwise.
- Write code that reads like prose to another human.
- Add comments only for non-obvious intent, constraints, or tradeoffs
- Favor explicitness over hidden magic
- Write modern code that is still easy to follow

## How I write

- Avoid jargon-heavy or cryptic code
- Keep logic readable
- Add comments only for non-obvious intent, constraints, or tradeoffs
- Use consistent naming that a code learner can understand
- Break overly-complex logic into helper functions
- **No cryptic abbreviations in variable names.** Spell out what it is: `group_idx` not `gi`, `compiled_group` not `cg`, `validated_node` not `vn`. Single-letter loop variables (`i`, `j`) and short names in tight local scopes are acceptable. Variables with wider scope or appearing in generated output, logs, or errors must use full words.


## When to use me

Before touching any code, proactively explore the codebase to build context relevant to the task, until you have a complete picture of how the affected code works and what depends on it and or what it is dependent on. Do not begin editing until you understand the blast radius.
Use this skill when writing or refactoring code that should be:
- production-ready
- modern
- advanced but approachable
- easy to maintain
- easy to learn from
