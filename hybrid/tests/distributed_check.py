"""Two-GPU acceptance: distributed mixed gradients equal average-then-update.

Run: torchrun --standalone --nproc_per_node=2 -m hybrid.tests.distributed_check
"""

import argparse
import copy

import torch
import torch.distributed as dist

from hybrid.config import Config, PRELUDE_MODES, recipe
from hybrid.model import ExperimentGPT
from hybrid.steps import backward_microbatch
from hybrid.train import build_optimizer, model_config
from nanochat.common import compute_cleanup, compute_init
from nanochat.optim import MuonAdamW


def check_variant(prelude_mode, rank, world, device):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    cfg = Config(mode="hybrid", prelude_mode=prelude_mode, size=1, aspect_ratio=32,
                 head_dim=8, sequence_len=8, vocab_size=64, n_prelude=1, n_recur_block=1, n_coda=1,
                 device_batch_size=1, total_batch_size=8 * world, num_iterations=4, compile=False)
    model = ExperimentGPT(model_config(cfg)).to(device)
    model.init_weights()
    with torch.no_grad():
        model.inject.weight[:, cfg.width:].copy_(torch.eye(cfg.width, device=device) * 0.2)
    optimizer = build_optimizer(model, cfg, recipe(cfg, world))
    reference = copy.deepcopy(model)
    lookup = {id(p): name for name, p in model.named_parameters()}
    reference_params = dict(reference.named_parameters())
    groups = []
    for group in optimizer.param_groups:
        groups.append({**{key: value for key, value in group.items() if key != "params"},
                       "params": [reference_params[lookup[id(p)]] for p in group["params"]]})
    expected_optimizer = MuonAdamW(groups)
    for step in range(3):
        gen = torch.Generator(device=device).manual_seed(200 + rank + step * world)
        x = torch.randint(cfg.vocab_size, (1, 8), device=device, generator=gen)
        y = torch.randint(cfg.vocab_size, (1, 8), device=device, generator=gen)
        backward_microbatch(model, x, y, cfg, step=step, micro_step=0, rank=rank, accumulation_steps=1)
        for name, p in model.named_parameters():
            averaged = p.grad.clone()
            dist.all_reduce(averaged, op=dist.ReduceOp.SUM)
            averaged /= world
            reference_params[name].grad = averaged
        expected_optimizer.step()
        optimizer.step()
        for name, p in model.named_parameters():
            torch.testing.assert_close(p, reference_params[name], rtol=1e-5, atol=1e-6)
            from_rank_zero = p.detach().clone()
            dist.broadcast(from_rank_zero, src=0)
            assert torch.equal(p, from_rank_zero), f"Rank parameters drifted: {name} ({prelude_mode})"
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
    if rank == 0:
        print(f"PASS {prelude_mode}: distributed FO+ZO averaging, single update, identical rank parameters")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prelude-mode", choices=["all", *PRELUDE_MODES], default="all")
    args = parser.parse_args()
    _, rank, _, world, device = compute_init("cuda")
    try:
        if world < 2:
            raise ValueError("Launch with at least two GPU ranks")
        torch._dynamo.config.disable = True
        for mode in PRELUDE_MODES if args.prelude_mode == "all" else [args.prelude_mode]:
            check_variant(mode, rank, world, device)
    finally:
        compute_cleanup()


if __name__ == "__main__":
    main()
