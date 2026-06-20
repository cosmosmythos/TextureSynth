---
name: blender-addon
description: Use when deploying non-C++ edits (Python addon code in ADDON/, or GLSL/JSON shader assets in shader_assets/) to the installed Blender extension folder so the user can manually test changes in Blender fast. Covers copy-only deploy of changed files, never binaries or manifests.
---

## What I do
- Copy changed `.py` files from `ADDON/**` to the install mirror
- Copy changed `.glsl` and `.node.json` from `shader_assets/glsl/` and `shader_assets/nodes/` to the install mirror
- Auto-detect the active Blender version under `%APPDATA%\Blender Foundation\Blender\<ver>\`
- Refuse anything outside the allowed scope or with a protected extension/path

## When to use me
- After editing a `.py` file under `ADDON/` that should be testable in Blender without a full rebuild
- After editing a `.glsl` or `.node.json` shader asset under `shader_assets/`
- When the user asks to "deploy", "push to install", "test in Blender", or "make it live"
- Use the dry-run (`-n`) first when uncertain about which files are in scope

## When NOT to use me
- Editing C++ in `src/` or `core/` -- those need `build_fast.bat`, not a file copy
- Touching `blender_manifest.toml` -- edit in `ADDON/` only; never deploy the manifest
- Producing `.pyd`/`.so` binaries -- GitHub CI ships these, never copy manually
- Editing `wheels/` -- off-limits

## Install location
The script auto-detects the install root by scanning for the first Blender version folder that contains `extensions/user_default/texturesynth`:
`%APPDATA%\Blender Foundation\Blender\<ver>\extensions\user_default\texturesynth\`

On the current machine that resolves to:
`C:\Users\User\AppData\Roaming\Blender Foundation\Blender\5.0\extensions\user_default\texturesynth\`

## Scope rules (enforced by the script)
| Repo path pattern | Deploys? |
|---|---|
| `ADDON/**/*.py` | yes |
| `shader_assets/glsl/**` | yes |
| `shader_assets/nodes/*.glsl` | yes |
| `shader_assets/nodes/*.node.json` | yes |
| anything else | **REFUSE** |
| `blender_manifest.toml`, `wheels/`, `core/` (even if path-matched) | **REFUSE** |
| `.pyd`, `.so`, `.toml` extensions | **REFUSE** |

## Usage
From anywhere, pass repo-relative paths to the files you changed:

```bash
# Dry-run preview (no copies made)
bash .opencode/skills/blender-addon/deploy.sh -n \
  ADDON/nodes/specialized/levels.py \
  shader_assets/nodes/levels.glsl

# Real deploy
bash .opencode/skills/blender-addon/deploy.sh \
  ADDON/nodes/specialized/levels.py \
  shader_assets/nodes/levels.glsl \
  shader_assets/nodes/levels.node.json
```

Output is verbose (`cp -v`), so it doubles as the deploy report. Quote the deploy summary back to the user: which files copied, to where.

## Workflow contract (root AGENTS.md section 4)
This skill is the implementation of the **Standing Deploy Exception** in root `AGENTS.md` section 4. The allow/deny lists in this document and in section 4 are kept in sync intentionally -- do not weaken one without updating the other.

## Verification checklist
Before reporting a deploy as done:
- [ ] Every changed file in `ADDON/`, `shader_assets/glsl/`, or `shader_assets/nodes/` was passed to the script
- [ ] The script reported each file as copied (no REFUSE/SKIP lines)
- [ ] No protected file (`blender_manifest.toml`, `wheels/`, `core/`, `.pyd`, `.so`) was touched
- [ ] The summary states what was copied and the resolved install root
