#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for arg in "$@"; do
    case "$arg" in
        --output-dir|--output-dir=*|--prelude-mode|--prelude-mode=*)
            echo "This sweep selects distinct prelude modes and output directories automatically." >&2
            exit 2 ;;
    esac
done
for prelude in none separate joint joint_recompute; do
    bash "$HERE/launch.sh" hybrid --prelude-mode "$prelude" "$@"
done
