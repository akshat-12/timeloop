#!/usr/bin/env bash
# Compatibility entry point; evaluation is implemented in Python.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec python3 "$here/run.py" "$@"
