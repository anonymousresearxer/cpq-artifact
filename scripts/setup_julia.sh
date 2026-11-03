#!/usr/bin/env bash
# Installs juliaup, Julia 1.13.2 and Python 3.14 for uv. Used as the setup script of the cloud
# environment "caise" (network: Full, or Custom with the Trusted defaults plus the julialang.org
# domains) and by setup_tower.sh on a local machine. Julia packages are installed and precompiled
# inside the session, because that takes longer than the cloud setup cache window.
set -euo pipefail
if ! command -v juliaup >/dev/null 2>&1; then
  curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel 1.13.2
fi
export PATH="$HOME/.juliaup/bin:$PATH"
grep -q 'juliaup/bin' "$HOME/.bashrc" 2>/dev/null || echo 'export PATH="$HOME/.juliaup/bin:$PATH"' >> "$HOME/.bashrc"
julia --version
if command -v uv >/dev/null 2>&1; then
  uv python install 3.14 || echo "uv could not install Python 3.14 here; the session will retry"
fi
