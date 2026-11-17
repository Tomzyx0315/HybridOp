#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
MODE=${1:?Usage: launch.sh full_fo|truncated|hybrid [training arguments]}
shift
case "$MODE" in full_fo|truncated|hybrid) ;; *) echo "Unknown mode: $MODE" >&2; exit 2 ;; esac
cd "$ROOT"
if [[ -z ${PYTHON_BIN:-} ]]; then
    if [[ -x "$ROOT/.venv/bin/python" ]]; then PYTHON_BIN="$ROOT/.venv/bin/python"; else PYTHON_BIN=python3; fi
fi
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" \
    -m hybrid.train --config "$ROOT/hybrid/configs/loop2_219m.json" --mode "$MODE" "$@"
