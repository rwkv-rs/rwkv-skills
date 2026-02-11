#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export NPM_CONFIG_CACHE="${NPM_CONFIG_CACHE:-/tmp/.npm}"
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-/tmp/pw-browsers}"

mkdir -p "$NPM_CONFIG_CACHE" "$PLAYWRIGHT_BROWSERS_PATH"

if [[ "${SPACE_E2E_INSTALL_BROWSER:-1}" == "1" ]]; then
  NPM_CONFIG_CACHE="$NPM_CONFIG_CACHE" PLAYWRIGHT_BROWSERS_PATH="$PLAYWRIGHT_BROWSERS_PATH" \
    npx --yes playwright install chromium
fi

NPM_CONFIG_CACHE="$NPM_CONFIG_CACHE" PLAYWRIGHT_BROWSERS_PATH="$PLAYWRIGHT_BROWSERS_PATH" \
  npx --yes -p playwright node "$ROOT_DIR/scripts/e2e/space_playwright_probe.mjs"
