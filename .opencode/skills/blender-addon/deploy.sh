#!/usr/bin/env bash
# Deploy non-C++ edits to the installed Blender extension for fast manual testing.
# Repo source -> install mirror:
#   ADDON/**                       -> <install>/**
#   shader_assets/glsl/**          -> <install>/shader_assets/glsl/**
#   shader_assets/nodes/*.glsl     -> <install>/shader_assets/nodes/*.glsl
#   shader_assets/nodes/*.node.json-> <install>/shader_assets/nodes/*.node.json
# Never touches: blender_manifest.toml, wheels/, .pyd/.so binaries, core/ C++.

set -euo pipefail

REPO_ROOT="/c/Users/User/Documents/0/TEXTURESYNTH"
APPDATA_WIN="$APPDATA"            # C:\Users\<u>\AppData\Roaming
# Translate to git-bash path
APPDATA_BASH="/c/${APPDATA_WIN#C:\\}"
APPDATA_BASH="${APPDATA_BASH//\\//}"

DRY_RUN=0
if [[ "${1:-}" == "-n" || "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift || true
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 [-n|--dry-run] <repo-relative-file> [more files...]" >&2
  echo "  Files must be under ADDON/, shader_assets/glsl/, or shader_assets/nodes/" >&2
  exit 2
fi

# Locate install root: .../Blender/<ver>/extensions/user_default/texturesynth
INSTALL_ROOT=""
SEARCH_BASE="$APPDATA_BASH/Blender Foundation/Blender"
for ver_dir in "$SEARCH_BASE"/*/; do
  [[ -d "$ver_dir" ]] || continue
  cand="${ver_dir}extensions/user_default/texturesynth"
  if [[ -d "$cand" ]]; then
    INSTALL_ROOT="$cand"
    break
  fi
done

if [[ -z "$INSTALL_ROOT" ]]; then
  echo "ERROR: install folder not found under $SEARCH_BASE/<ver>/extensions/user_default/" >&2
  exit 1
fi
echo "Install root: $INSTALL_ROOT"

deploy_one() {
  local repo_rel="$1"
  local repo_abs="$REPO_ROOT/$repo_rel"

  if [[ ! -f "$repo_abs" ]]; then
    echo "SKIP (missing in repo): $repo_rel" >&2
    return 1
  fi

  # Path-scope guard: must be in one of the allowed source trees.
  local dest_rel=""
  case "$repo_rel" in
    ADDON/*)                                 dest_rel="${repo_rel#ADDON/}" ;;
    shader_assets/glsl/*)                    dest_rel="$repo_rel" ;;
    shader_assets/nodes/*.glsl)              dest_rel="$repo_rel" ;;
    shader_assets/nodes/*.node.json)         dest_rel="$repo_rel" ;;
    *)
      echo "REFUSE (out of scope): $repo_rel" >&2
      echo "  Allowed: ADDON/**, shader_assets/glsl/**, shader_assets/nodes/*.glsl, *.node.json" >&2
      return 1
      ;;
  esac

  # Extension guard.
  case "${repo_rel##*.}" in
    py|glsl|json) ;;
    *)
      echo "REFUSE (bad ext): $repo_rel -- only .py/.glsl/.json" >&2
      return 1
      ;;
  esac

  # Never overwrite protected files even if path-scoped.
  case "$dest_rel" in
    blender_manifest.toml|wheels/*|core/*)
      echo "REFUSE (protected): $dest_rel" >&2
      return 1
      ;;
  esac

  local dest_abs="$INSTALL_ROOT/$dest_rel"
  mkdir -p "$(dirname "$dest_abs")"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "DRY: $repo_rel -> $dest_abs"
  else
    cp -v "$repo_abs" "$dest_abs"
  fi
}

fail=0
for f in "$@"; do
  deploy_one "$f" || fail=1
done

if [[ $fail -ne 0 ]]; then
  echo "" >&2
  echo "One or more files were refused or missing (see above)." >&2
  exit 1
fi
echo "Done."
