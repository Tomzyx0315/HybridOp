"""FO-to-target interpolation, including early credit to prelude and resume state."""

import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from hybrid.norm_matching import mix_shared_gradients, separated_backward
from hybrid.steps import backward_microbatch
from hybrid.tests.test_gradients import cpu_test_settings, tiny
from hybrid.transition import mix_transition_gradients, transition_backward, transition_flops_per_token


@pytest.mark.parametrize("ratio", [None, 0.3, 1.0])
@pytest.mark.parametrize("fraction", [0.0, 0.35, 1.0])
def test_gradient_is_convex_combination_of_full_fo_and_target(ratio, fraction):
    cfg, model, x, y = tiny()
    cfg = replace(cfg, mode="truncated" if ratio is None else "hybrid", zo_norm_ratio=ratio,
                  fo_checkpoint_dir="/source", fo_checkpoint_step=6, transition_steps=5)
    full, target = copy.deepcopy(model), copy.deepcopy(model)
    exact = {n: torch.zeros_like(p) for n, p in model.shared_named_parameters()}
    noise = {n: torch.zeros_like(p) for n, p in model.shared_named_parameters()} if ratio is not None else None
    target_noise = {n: torch.zeros_like(p) for n, p in model.shared_named_parameters()} if ratio is not None else None
    for micro in range(2):
        batch = x.roll(micro, dims=1)
        backward_microbatch(full, batch, y, replace(cfg, mode="full_fo"), step=7,
                            micro_step=micro, rank=0, accumulation_steps=2)
        if ratio is None:
            backward_microbatch(target, batch, y, cfg, step=7, micro_step=micro, rank=0, accumulation_steps=2)
        else:
            separated_backward(target, batch, y, cfg, target_noise, step=7, micro=micro, rank=0, accumulation=2)
        transition_backward(model, batch, y, cfg, noise, exact, fraction,
                            step=7, micro=micro, rank=0, accumulation=2)
    if ratio is not None:
        mix_shared_gradients(target, target_noise, ratio)
    mix_transition_gradients(model, noise, exact, ratio, fraction)
    for name, p in model.named_parameters():
        expected = ((1-fraction)*dict(full.named_parameters())[name].grad +
                    fraction*dict(target.named_parameters())[name].grad)
        torch.testing.assert_close(p.grad, expected, rtol=5e-4, atol=2e-7, msg=name)
    # The temporary implementation replays exact early and prelude computations.
    assert transition_flops_per_token(model, cfg, 0.5) > transition_flops_per_token(model, cfg, 1)


@pytest.mark.integration
@pytest.mark.parametrize("ratio", [None, 0.3])
def test_fork_resume_preserves_original_cursor_optimizer_and_transition(tmp_path, ratio):
    root = Path(__file__).resolve().parents[2]
    common = [sys.executable, "-m", "hybrid.train", "--device", "cpu", "--synthetic-data", "--no-compile",
              "--size", "1", "--aspect-ratio", "16", "--head-dim", "8", "--sequence-len", "8",
              "--vocab-size", "32", "--n-prelude", "1", "--n-recur-block", "1", "--n-coda", "1",
              "--device-batch-size", "1", "--total-batch-size", "16", "--num-iterations", "7",
              "--eval-every", "-1", "--log-every", "1"]
    env = {k: v for k, v in os.environ.items() if k not in {"RANK", "LOCAL_RANK", "WORLD_SIZE"}}
    env["OMP_NUM_THREADS"] = "1"

    def run(directory, args):
        subprocess.run(common + ["--output-dir", str(directory), *args], cwd=root, env=env,
                       check=True, capture_output=True, text=True, timeout=180)

    source, continuous, resumed = tmp_path/'source', tmp_path/'continuous', tmp_path/'resumed'
    run(source, ["--mode", "full_fo", "--stop-after", "2"])
    parent_bytes = (source/'model_000002.pt').read_bytes()
    fork = ["--fo-checkpoint-dir", str(source), "--fo-checkpoint-step", "2", "--transition-steps", "4",
            "--mode", "truncated" if ratio is None else "hybrid"]
    if ratio is not None:
        fork += ["--zo-norm-ratio", str(ratio)]
    run(continuous, fork)
    run(resumed, [*fork, "--stop-after", "4"])
    run(resumed, [*fork, "--resume-step", "4"])
    for kind in ("model_000007.pt", "optim_000007_rank0.pt", "training_000007_rank0.pt"):
        a = torch.load(continuous/kind, weights_only=True)
        b = torch.load(resumed/kind, weights_only=True)

        def equal(left, right):
            if isinstance(left, torch.Tensor):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            elif isinstance(left, dict):
                assert left.keys() == right.keys()
                for key in left:
                    equal(left[key], right[key])
            elif isinstance(left, (list, tuple)):
                assert len(left) == len(right)
                for l, r in zip(left, right):
                    equal(l, r)
            else:
                assert left == right
        equal(a, b)
    assert (source/'model_000002.pt').read_bytes() == parent_bytes
    metrics=[json.loads(l) for l in (continuous/'metrics.jsonl').read_text().splitlines()]
    assert [d['transition_fraction'] for d in metrics] == [0, 1/3, 2/3, 1, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA transition acceptance")
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("ratio", [None, 0.3])
def test_cuda_transition_replay_is_close_to_full_gradient_interpolation(compiled, ratio):
    cfg, original, x, y = tiny()
    cfg = replace(cfg, mode="truncated" if ratio is None else "hybrid", zo_norm_ratio=ratio)
    model = original.cuda()
    full, target = copy.deepcopy(model), copy.deepcopy(model)
    x, y = x.cuda(), y.cuda()
    if compiled:
        torch._dynamo.config.disable = False
        for m in (model, full, target):
            m.compile_blocks()
    fraction = 0.35
    exact = {n: torch.zeros_like(p) for n,p in model.shared_named_parameters()}
    noise = {n: torch.zeros_like(p) for n,p in model.shared_named_parameters()} if ratio is not None else None
    target_noise = {n: torch.zeros_like(p) for n,p in model.shared_named_parameters()} if ratio is not None else None
    backward_microbatch(full, x, y, replace(cfg, mode="full_fo"), step=7,
                        micro_step=0, rank=0, accumulation_steps=1)
    if ratio is None:
        backward_microbatch(target, x, y, cfg, step=7, micro_step=0, rank=0, accumulation_steps=1)
    else:
        separated_backward(target, x, y, cfg, target_noise, step=7, micro=0, rank=0, accumulation=1)
        mix_shared_gradients(target, target_noise, ratio)
    transition_backward(model, x, y, cfg, noise, exact, fraction, step=7, micro=0, rank=0, accumulation=1)
    mix_transition_gradients(model, noise, exact, ratio, fraction)
    # Separate BF16 VJPs have different accumulation rounding from full BPTT.
    # Check every parameter's relative L2 error rather than tiny near-zero entries.
    for n,p in model.named_parameters():
        expected=(1-fraction)*dict(full.named_parameters())[n].grad+fraction*dict(target.named_parameters())[n].grad
        error=(p.grad-expected).float().norm()
        assert error <= 0.03*expected.float().norm()+2e-7, (n,error.item(),expected.float().norm().item())
