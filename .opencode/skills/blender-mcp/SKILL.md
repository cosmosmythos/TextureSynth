name: ai-agent-skill
description: Use when testing, verifying, or troubleshooting an addon inside a live host application process through an agent bridge. It supports two modes: (1) build/sweep/bake style tests against a fresh controlled setup, and (2) introspect + verify whatever the user already has open, asking a quick clarification before mutating anything. Use whenever the user asks to test an addon, verify behavior end-to-end, check what is open, confirm a setup, reproduce a bug that depends on the live host context, or validate that a recent change actually works in the real application.

---

## What I do

Drive the installed addon from inside a running host application through the agent bridge. Many addons are event-driven, so a test is usually not a single function call — it is: change state, let the host process the update, then read back the result.

Use the project’s bundled scripts or helpers as canonical patterns. Read the script that matches the test before writing your own, because addon APIs often have non-obvious contracts such as stable IDs, socket formats, update order, context requirements, or async readback behavior.

## The agent bridge contract

The bridge usually exposes one primary action: execute code in the live host environment. Other helper endpoints may exist, but they are often limited to general scene or object inspection and may not be useful for addon-specific workflows.

Key facts to assume:

| Fact | Implication |
|---|---|
| Code runs in a fresh namespace per call | Every script must self-contain its imports and setup. Nothing should be assumed to persist between calls. |
| Code runs on the main thread | Host operators and UI-sensitive APIs may be valid, but the user’s real editor or application state is being used. |
| Stdout is captured | Use `print()` as the primary reporting channel. Structure output as a test report. |
| Live context reflects the user’s current state | Never assume you can set active UI state by writing to internal collections alone; use the host’s supported context override or direct API when needed. |
| The bridge listens on a local socket | The user may need to enable or connect the bridge first. If the connection fails, instruct them to connect or enable the addon/bridge. |

## How to resolve the addon package

Addon import paths may differ depending on how the addon is installed. Prefer a resolver helper and avoid hardcoding a single import path.

```python
import importlib, sys

def resolve_addon_package():
    # Try the extension/packaged path first, then fall back to the legacy name.
    candidate_names = [
        "addon_package_name",
        "legacy_addon_name",
    ]

    for name in candidate_names:
        if name in sys.modules:
            return importlib.import_module(name)

    # If already loaded under a different registered module, find it dynamically.
    for mod in sys.modules.values():
        if hasattr(mod, "__dict__") and getattr(mod, "__package__", None):
            # Add project-specific detection here.
            pass

    raise RuntimeError("Addon is not loaded in this host application")
