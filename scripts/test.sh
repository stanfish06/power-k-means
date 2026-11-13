#!/usr/bin/env bash
# Rebuild the Python extension from the current sources, then run the tests.
#   scripts/test.sh            all tests
#   scripts/test.sh -k blobs   extra arguments go to pytest
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# On NixOS the first c++ on PATH can be an unwrapped gcc that cannot link;
# use the c++ next to cc unless CXX is set (nix develop sets it).
if [[ -z "${CXX:-}" ]]; then
  CXX="$(dirname "$(command -v cc)")/c++"
  export CXX
fi

cd "$ROOT"
# --reinstall-package rebuilds the extension from the current sources
uv sync --reinstall-package power-k-means
uv run pytest "$@"
