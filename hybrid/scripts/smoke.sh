#!/usr/bin/env bash
# Run only on the experiment machine. Uses random tokens, not a scientific run.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export NPROC_PER_NODE=${NPROC_PER_NODE:-1}
OUT=${SMOKE_OUTPUT:-/tmp/hybrid-smoke-$(date +%Y%m%d-%H%M%S)}
for variant in full_fo truncated hybrid_none hybrid_separate hybrid_joint hybrid_joint_recompute; do
    mode=$variant
    prelude=none
    if [[ $variant == hybrid_* ]]; then mode=hybrid; prelude=${variant#hybrid_}; fi
    bash "$HERE/launch.sh" "$mode" --prelude-mode "$prelude" \
        --synthetic-data --no-compile --size 1 --aspect-ratio 32 --head-dim 8 \
        --n-prelude 1 --n-recur-block 1 --n-coda 1 --sequence-len 16 --vocab-size 64 \
        --device-batch-size 1 --total-batch-size "$((32 * NPROC_PER_NODE))" \
        --num-iterations 4 --eval-every 2 --eval-rows "$((4 * NPROC_PER_NODE))" \
        --final-eval-rows "$((4 * NPROC_PER_NODE))" --log-every 1 \
        --output-dir "$OUT/$variant" "$@"
done
echo "Smoke outputs: $OUT"
