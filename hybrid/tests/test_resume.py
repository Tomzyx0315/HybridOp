"""Tiny end-to-end checkpoint/resume acceptance check; no dataset download."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from hybrid.config import PRELUDE_MODES


@pytest.mark.integration
@pytest.mark.parametrize("prelude_mode", PRELUDE_MODES)
def test_resumed_hybrid_matches_uninterrupted(tmp_path, prelude_mode):
    root = Path(__file__).resolve().parents[2]
    common = [sys.executable, "-m", "hybrid.train", "--mode", "hybrid", "--device", "cpu",
              "--synthetic-data", "--no-compile", "--size", "1", "--aspect-ratio", "16",
              "--head-dim", "8", "--sequence-len", "8", "--vocab-size", "32",
              "--n-prelude", "1", "--n-recur-block", "1", "--n-coda", "1",
              "--device-batch-size", "1", "--total-batch-size", "16", "--num-iterations", "4",
              "--eval-every", "-1", "--log-every", "1", "--prelude-mode", prelude_mode]
    env = {k: v for k, v in os.environ.items() if k not in {"RANK", "LOCAL_RANK", "WORLD_SIZE"}}
    env["OMP_NUM_THREADS"] = "1"
    def run(directory, extra=()):
        subprocess.run(common + ["--output-dir", str(directory)] + list(extra), cwd=root, env=env,
                       check=True, capture_output=True, text=True, timeout=180)
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    run(continuous)
    run(resumed, ["--stop-after", "2"])
    run(resumed, ["--resume-step", "2"])
    left = torch.load(continuous / "model_000004.pt", weights_only=True)
    right = torch.load(resumed / "model_000004.pt", weights_only=True)
    for name in left:
        torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
