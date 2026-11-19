"""Launch one command after all requested GPUs pass consecutive idle checks."""

import argparse
import csv
import fcntl
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def inspect_gpus(indices, max_memory_mib=256, max_utilization=5):
    def query(fields, kind):
        result = subprocess.run(
            ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=20)
        return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))

    try:
        cards = {}
        for index, uuid, memory, utilization in query("index,uuid,memory.used,utilization.gpu", "gpu"):
            cards[int(index)] = dict(index=int(index), uuid=uuid.strip(), memory_mib=int(memory),
                                     utilization_percent=int(utilization))
        missing = sorted(set(indices) - cards.keys())
        if missing:
            raise ValueError(f"Missing GPU indices: {missing}")
        selected = [cards[i] for i in indices]
        uuids = {card["uuid"] for card in selected}
        processes = [dict(uuid=uuid.strip(), pid=int(pid))
                     for uuid, pid in query("gpu_uuid,pid", "compute-apps") if uuid.strip() in uuids]
        busy = [card["index"] for card in selected
                if card["memory_mib"] > max_memory_mib or card["utilization_percent"] > max_utilization]
        return dict(idle=not processes and not busy, gpus=selected, processes=processes,
                    busy_by_memory_or_utilization=busy, query_error=None)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        # A failed query or N/A counter is not evidence that the GPUs are idle.
        return dict(idle=False, gpus=[], processes=[], busy_by_memory_or_utilization=[],
                    query_error=str(exc))


def wait_until_idle(indices, *, poll_seconds, idle_checks, max_memory_mib, max_utilization,
                    emit, inspect=inspect_gpus, sleep=time.sleep):
    consecutive = 0
    while True:
        snapshot = inspect(indices, max_memory_mib, max_utilization)
        consecutive = consecutive + 1 if snapshot["idle"] else 0
        ready = consecutive >= idle_checks
        emit(dict(event="gpu_ready" if ready else "gpu_wait", checked_at=datetime.now(timezone.utc).isoformat(),
                  consecutive_idle_checks=consecutive, required_idle_checks=idle_checks,
                  poll_seconds=poll_seconds, max_memory_mib=max_memory_mib,
                  max_utilization_percent=max_utilization, **snapshot))
        if ready:
            return snapshot
        sleep(poll_seconds)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-indices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--poll-seconds", type=float, default=180)
    parser.add_argument("--idle-checks", type=int, default=2)
    parser.add_argument("--max-memory-mib", type=int, default=256)
    parser.add_argument("--max-utilization", type=int, default=5)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        indices = [int(item) for item in args.gpu_indices.split(",")]
    except ValueError:
        parser.error("gpu-indices must be comma-separated integers")
    if not indices or len(indices) != len(set(indices)) or min(indices) < 0:
        parser.error("GPU indices must be unique and nonnegative")
    if args.poll_seconds <= 0 or args.idle_checks < 2 or args.max_memory_mib < 0 or not 0 <= args.max_utilization <= 100:
        parser.error("Require positive polling interval, at least two checks, and valid thresholds")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Provide a command after --")
    args.state_file.parent.mkdir(parents=True, exist_ok=True)
    args.lock_file.parent.mkdir(parents=True, exist_ok=True)

    def emit(record):
        temp = args.state_file.with_name(f"{args.state_file.name}.{os.getpid()}.tmp")
        temp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temp.replace(args.state_file)
        print(json.dumps(record, allow_nan=False), flush=True)

    with args.lock_file.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another instance already holds this queue lock")
        snapshot = wait_until_idle(indices, poll_seconds=args.poll_seconds, idle_checks=args.idle_checks,
                                   max_memory_mib=args.max_memory_mib, max_utilization=args.max_utilization,
                                   emit=emit)
        # Bind CUDA to exactly the physical GPUs whose UUIDs were checked.
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(card["uuid"] for card in snapshot["gpus"]))
        child = subprocess.Popen(command, env=env)

        def forward_signal(signum, _frame):
            if child.poll() is None:
                child.send_signal(signum)

        previous = {sig: signal.signal(sig, forward_signal) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            code = child.wait()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        emit(dict(event="gpu_job_exit", checked_at=datetime.now(timezone.utc).isoformat(), returncode=code))
        return code if code >= 0 else 128 - code


if __name__ == "__main__":
    raise SystemExit(main())
