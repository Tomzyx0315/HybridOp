#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/launch.sh" hybrid --prelude-mode separate "$@"
