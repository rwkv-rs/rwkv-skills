#!/usr/bin/env bash
set -euo pipefail

repo=/home/rwkv/chase/rwkv-skills
cd "$repo"
set -a
source .env
set +a
exec .venv/bin/python ops/g1i_strict46/finalize_when_complete.py --interval-s 60
