#!/usr/bin/env bash
set -euo pipefail

CODE_DIR="${CODE_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
IG_REPO="${IG_REPO:-$CODE_DIR/third_party/invertinggradients}"
if [[ ! -e "$IG_REPO" ]]; then
  mkdir -p "$(dirname -- "$IG_REPO")"
  git clone https://github.com/JonasGeiping/invertinggradients.git "$IG_REPO"
fi
git -C "$IG_REPO" rev-parse --is-inside-work-tree >/dev/null
if [[ -n "${IG_COMMIT:-}" ]]; then
  git -C "$IG_REPO" checkout --detach "$IG_COMMIT"
fi
if [[ ! -s "$IG_REPO/inversefed/reconstruction_algorithms.py" ]]; then
  echo "Missing or empty IG implementation: $IG_REPO" >&2
  exit 1
fi
echo "IG revision: $(git -C "$IG_REPO" rev-parse HEAD)"
