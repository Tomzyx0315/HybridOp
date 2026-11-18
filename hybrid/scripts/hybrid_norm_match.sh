#!/usr/bin/env bash
set -euo pipefail
RATIO=${1:?Usage: hybrid_norm_match.sh ratio [training arguments]}
shift
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/launch.sh" hybrid \
    --prelude-mode none --zo-lambda 1 --zo-norm-ratio "$RATIO" "$@"
