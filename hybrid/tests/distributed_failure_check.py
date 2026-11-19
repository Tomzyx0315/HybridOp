"""Verify a rank-local error exits promptly instead of hanging in NCCL cleanup."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def worker(directory):
    from hybrid import steps
    from nanochat import common
    from hybrid.train import main

    original_backward = steps.backward_microbatch
    original_cleanup = common.compute_cleanup

    def fail_rank_zero(*args, **kwargs):
        result = original_backward(*args, **kwargs)
        if int(os.environ["RANK"]) == 0:
            raise RuntimeError("injected_rank_zero_training_failure")
        return result

    def observe_cleanup():
        if int(os.environ["RANK"]) == 0:
            (directory / "rank_zero_entered_cleanup").write_text("unexpected\n")
        return original_cleanup()

    steps.backward_microbatch = fail_rank_zero
    common.compute_cleanup = observe_cleanup
    main(["--device", "cuda", "--synthetic-data", "--no-compile", "--size", "1",
          "--aspect-ratio", "32", "--head-dim", "8", "--sequence-len", "16",
          "--vocab-size", "64", "--n-prelude", "1", "--n-recur-block", "1", "--n-coda", "1",
          "--device-batch-size", "1", "--total-batch-size", "64", "--num-iterations", "2",
          "--eval-every", "-1", "--output-dir", str(directory / "run")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    if args.worker:
        worker(directory)
        return
    directory.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
               "-m", "hybrid.tests.distributed_failure_check", "--worker", "--output-dir", str(directory)]
    env = {k: v for k, v in os.environ.items() if k not in {"RANK", "LOCAL_RANK", "WORLD_SIZE"}}
    env["HYBRID_DIAGNOSTICS_DIR"] = str(directory / "stacks")
    started = time.monotonic()
    result = subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=120)
    elapsed = time.monotonic() - started
    (directory / "console.log").write_text(result.stdout)
    assert result.returncode != 0, "The injected exception must fail the worker"
    assert "injected_rank_zero_training_failure" in result.stdout
    assert "Watchdog caught collective operation timeout" not in result.stdout
    assert not (directory / "rank_zero_entered_cleanup").exists()
    summary = dict(passed=True, elapsed_seconds=elapsed, worker_exit_code=result.returncode,
                   original_exception_visible=True, failed_rank_called_collective_cleanup=False)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
