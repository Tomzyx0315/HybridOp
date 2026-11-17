#!/bin/bash
# Depth robustness sweep with a (1,1) prelude/coda and 18 effective layers.
#
# Architectures:
#   loop2:   1 prelude + 8 recur-block x 2 recurrences + 1 coda
#   loop4:   1 prelude + 4 recur-block x 4 recurrences + 1 coda
#   loop8:   1 prelude + 2 recur-block x 8 recurrences + 1 coda
#   noloop:  18 blocks, no recurrence
#
# Environment variables:
#   ARCH         - loop8 | loop4 | loop2 | noloop (default: loop4)
#   TARGET_FLOPS - FLOPs budget (required)
#   SIZES        - comma-separated model sizes (default: auto from TARGET_FLOPS)
#
# Usage:
#   TARGET_FLOPS=4.64e17 ARCH=noloop ./shells/_submit.sh shells/scaling_laws/scaling_laws_pc11.sh -- --array=0-3
#   TARGET_FLOPS=1e18    ARCH=loop2  ./shells/_submit.sh shells/scaling_laws/scaling_laws_pc11.sh -- --array=0-4
#   TARGET_FLOPS=2.15e18 ARCH=loop4  ./shells/_submit.sh shells/scaling_laws/scaling_laws_pc11.sh -- --array=0-4
#   TARGET_FLOPS=4.64e18 ARCH=loop8  ./shells/_submit.sh shells/scaling_laws/scaling_laws_pc11.sh -- --array=0-5

cd "${REPO_ROOT:-$HOME/looped-lm-scaling}"
uv sync
source .venv/bin/activate

source "${MACHINE_CONFIG:-shells/_machine_config.sh}"
validate_config || exit 1

NPROC_PER_NODE=${NUM_GPUS:-1}

TARGET_FLOPS=${TARGET_FLOPS:?ERROR: Set TARGET_FLOPS env var}
MATRIX_LR=0.014

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if [ -n "$SIZES" ]; then
    IFS=',' read -ra SIZE_ARRAY <<< "$SIZES"
else
    case "$TARGET_FLOPS" in
        4.64e17) SIZE_ARRAY=(6 8 10 12) ;;
        1e18)    SIZE_ARRAY=(6 8 10 12 14) ;;
        2.15e18) SIZE_ARRAY=(8 10 12 14 16) ;;
        4.64e18) SIZE_ARRAY=(10 12 14 16 18 20) ;;
        *)       echo "ERROR: TARGET_FLOPS=$TARGET_FLOPS outside the four case-study budgets (4.64e17, 1e18, 2.15e18, 4.64e18). Set SIZES env var to override."; exit 1 ;;
    esac
fi
SIZE=${SIZE_ARRAY[$TASK_ID]}

if [ -z "${DEVICE_BATCH_SIZE:-}" ]; then
    if [ "$SIZE" -ge 30 ]; then
        DEVICE_BATCH_SIZE=8
    elif [ "$SIZE" -ge 18 ]; then
        DEVICE_BATCH_SIZE=16
    else
        DEVICE_BATCH_SIZE=32
    fi
fi

ARCH=${ARCH:-loop4}
case "$ARCH" in
    loop4)
        N_PRELUDE=1; N_RECUR_BLOCK=4;  N_CODA=1; NUM_RECUR=4
        INJECTION=inject_init_prelude
        ARCH_TAG="loop"  # historical tag, matches main-grid naming
        ;;
    loop2)
        N_PRELUDE=1; N_RECUR_BLOCK=8;  N_CODA=1; NUM_RECUR=2
        INJECTION=inject_init_prelude
        ARCH_TAG="loop2"
        ;;
    loop8)
        N_PRELUDE=1; N_RECUR_BLOCK=2;  N_CODA=1; NUM_RECUR=8
        INJECTION=inject_init_prelude
        ARCH_TAG="loop8"
        ;;
    noloop)
        N_PRELUDE=0; N_RECUR_BLOCK=18; N_CODA=0; NUM_RECUR=1
        INJECTION=passthrough
        ARCH_TAG="noloop"
        ;;
    *)
        echo "ERROR: Unknown ARCH=$ARCH (expected loop8|loop4|loop2|noloop)"; exit 1 ;;
esac

EFFECTIVE_DEPTH=$((N_PRELUDE + NUM_RECUR * N_RECUR_BLOCK + N_CODA))
TAG="isoflops_pc11_${TARGET_FLOPS}_${ARCH_TAG}_s${SIZE}"

echo "=============================================="
echo "Depth-split (1,1)@18 scaling law run: $TAG"
echo "  Arch:        $ARCH (${N_PRELUDE}-${N_RECUR_BLOCK}x${NUM_RECUR}-${N_CODA}, eff. depth $EFFECTIVE_DEPTH)"
echo "  Size:        s$SIZE"
echo "  Budget:      $TARGET_FLOPS FLOPs"
echo "  Device bs:   $DEVICE_BATCH_SIZE"
echo "=============================================="

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train \
    --target-flops $TARGET_FLOPS \
    --model-tag $TAG \
    --run $TAG \
    --device-batch-size $DEVICE_BATCH_SIZE \
    --eval-every=-1 \
    --core-metric-every=-1 \
    --sample-every=-1 \
    --save-every=${SAVE_EVERY:--1} \
    --num-recur=$NUM_RECUR \
    --bptt-k=$NUM_RECUR \
    --input-injection=$INJECTION \
    --n-prelude=$N_PRELUDE \
    --n-recur-block=$N_RECUR_BLOCK \
    --n-coda=$N_CODA \
    --matrix-lr=$MATRIX_LR \
    --size $SIZE
