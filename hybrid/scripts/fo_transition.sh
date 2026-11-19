#!/usr/bin/env bash
set -euo pipefail
TARGET=${1:?Usage: fo_transition.sh truncated|RATIO CHECKPOINT_DIR [training arguments]}
SOURCE=${2:?Provide the full-FO checkpoint directory}
shift 2
OPTIONS=(--fo-checkpoint-dir "$SOURCE" --fo-checkpoint-step 8000 --transition-steps 1000 --prelude-mode none)
if [[ "$TARGET" == truncated ]]; then
    MODE=truncated
else
    MODE=hybrid
    OPTIONS+=(--zo-lambda 1 --zo-norm-ratio "$TARGET")
fi
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/launch.sh" "$MODE" "${OPTIONS[@]}" "$@"
