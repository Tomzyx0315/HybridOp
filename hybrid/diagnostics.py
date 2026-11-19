"""Opt-in stack capture when a training process stops making progress."""

import faulthandler
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


@contextmanager
def stall_tracebacks():
    seconds = float(os.environ.get("HYBRID_STALL_SECONDS", "0"))
    if seconds <= 0:
        yield lambda: None
        return
    directory = Path(os.environ["HYBRID_DIAGNOSTICS_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    rank = os.environ.get("RANK", "0")
    path = directory / f"stacks_rank{rank}_pid{os.getpid()}.log"
    with path.open("w", buffering=1) as handle:
        handle.write(f"Started {datetime.now(timezone.utc).isoformat()}, timeout={seconds}s\n")

        def heartbeat():
            faulthandler.cancel_dump_traceback_later()
            faulthandler.dump_traceback_later(seconds, repeat=True, file=handle)

        heartbeat()
        try:
            yield heartbeat
        finally:
            faulthandler.cancel_dump_traceback_later()
