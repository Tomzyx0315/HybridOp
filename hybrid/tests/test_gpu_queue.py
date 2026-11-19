"""The shared-server queue must never interpret a busy or unknown GPU as free."""

import subprocess
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hybrid.wait_for_gpus import inspect_gpus, wait_until_idle, main


class GPUQueueTests(unittest.TestCase):
    def inspect(self, cards="0, GPU-a, 0, 0\n1, GPU-b, 12, 1\n", processes=""):
        with patch("hybrid.wait_for_gpus.subprocess.run", side_effect=[
            subprocess.CompletedProcess([], 0, stdout=cards),
            subprocess.CompletedProcess([], 0, stdout=processes),
        ]):
            return inspect_gpus([0, 1])

    def test_idle(self):
        self.assertTrue(self.inspect()["idle"])

    def test_existing_process_blocks_even_at_zero_utilization(self):
        self.assertFalse(self.inspect(processes="GPU-a, 123\n")["idle"])

    def test_memory_or_utilization_alone_blocks(self):
        for cards in ["0, GPU-a, 257, 0\n1, GPU-b, 0, 0\n",
                      "0, GPU-a, 0, 6\n1, GPU-b, 0, 0\n"]:
            with self.subTest(cards=cards):
                self.assertFalse(self.inspect(cards=cards)["idle"])

    def test_missing_gpu_and_unknown_counter_fail_closed(self):
        for cards in ["0, GPU-a, 0, 0\n", "0, GPU-a, N/A, 0\n1, GPU-b, 0, 0\n"]:
            with self.subTest(cards=cards):
                result = self.inspect(cards=cards)
                self.assertFalse(result["idle"])
                self.assertIsNotNone(result["query_error"])

    def test_query_failure_and_timeout_fail_closed(self):
        for error in [subprocess.CalledProcessError(1, "nvidia-smi"),
                      subprocess.TimeoutExpired("nvidia-smi", 20), FileNotFoundError("nvidia-smi")]:
            with self.subTest(error=error), patch("hybrid.wait_for_gpus.subprocess.run", side_effect=error):
                result = inspect_gpus([0, 1])
                self.assertFalse(result["idle"])
                self.assertIsNotNone(result["query_error"])

    def test_busy_sample_resets_consecutive_checks(self):
        samples = iter([True, False, True, True])
        events, sleeps = [], []
        wait_until_idle([0, 1], poll_seconds=180, idle_checks=2, max_memory_mib=256,
                        max_utilization=5, emit=events.append,
                        inspect=lambda *args: dict(idle=next(samples)), sleep=sleeps.append)
        self.assertEqual([e["consecutive_idle_checks"] for e in events], [1, 0, 1, 2])
        self.assertEqual(sleeps, [180, 180, 180])
        self.assertEqual(events[-1]["event"], "gpu_ready")

    def test_launches_after_two_checks_with_verified_gpu_uuids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "child.json"
            cards = "0, GPU-a, 0, 0\n1, GPU-b, 0, 0\n"
            responses = [subprocess.CompletedProcess([], 0, stdout=value)
                         for value in [cards, "", cards, ""]]
            child = ("import json,os,pathlib; pathlib.Path(" + repr(str(marker)) +
                     ").write_text(json.dumps(os.environ['CUDA_VISIBLE_DEVICES']))")
            with patch("hybrid.wait_for_gpus.subprocess.run", side_effect=responses) as query, contextlib.redirect_stdout(io.StringIO()):
                result = main(["--gpu-indices", "0,1", "--poll-seconds", "0.01",
                               "--state-file", str(root/"state.json"), "--lock-file", str(root/"lock"),
                               "--", sys.executable, "-c", child])
            self.assertEqual(result, 0)
            self.assertEqual(query.call_count, 4)
            self.assertEqual(json.loads(marker.read_text()), "GPU-a,GPU-b")
            self.assertEqual(json.loads((root/"state.json").read_text())["event"], "gpu_job_exit")


if __name__ == "__main__":
    unittest.main()
