#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for mode in full_fo truncated hybrid; do
    bash "$HERE/launch.sh" "$mode" "$@"
done
