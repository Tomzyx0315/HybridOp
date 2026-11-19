"""Two-GPU acceptance: distributed mixed gradients equal average-then-update.

Run: torchrun --standalone --nproc_per_node=2 -m hybrid.tests.distributed_check
"""

import argparse
import copy

import torch
import torch.distributed as dist

from hybrid.config import Config, PRELUDE_MODES, recipe
from hybrid.model import ExperimentGPT
from hybrid.norm_matching import mix_shared_gradients, separated_backward
from hybrid.steps import backward_microbatch
from hybrid.train import build_optimizer, model_config
from hybrid.transition import mix_transition_gradients, transition_backward
from nanochat.common import compute_cleanup, compute_init
from nanochat.optim import MuonAdamW


def check_variant(prelude_mode, rank, world, device, norm_ratio=None, fraction=None):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    cfg = Config(mode="truncated" if fraction is not None and norm_ratio is None else "hybrid",
                 prelude_mode=prelude_mode, size=1, aspect_ratio=32,
                 head_dim=8, sequence_len=8, vocab_size=64, n_prelude=1, n_recur_block=1, n_coda=1,
                 device_batch_size=1, total_batch_size=8 * world, num_iterations=4, compile=False,
                 zo_norm_ratio=norm_ratio)
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
        options = {key: value for key, value in group.items() if key != "params"}
        params = [reference_params[lookup[id(p)]] for p in group["params"]]
        # Muon uses BF16 batched matmuls. Match the owners' batch sizes so this
        # checks distributed averaging/update, rather than batch-kernel roundoff.
        chunk = (len(params) + world - 1) // world if group["kind"] == "muon" else len(params)
        for start in range(0, len(params), chunk):
            groups.append({**options, "params": params[start:start + chunk]})
    expected_optimizer = MuonAdamW(groups)
    for step in range(3):
        gen = torch.Generator(device=device).manual_seed(200 + rank + step * world)
        x = torch.randint(cfg.vocab_size, (1, 8), device=device, generator=gen)
        y = torch.randint(cfg.vocab_size, (1, 8), device=device, generator=gen)
        if fraction is not None:
            shared = dict(model.shared_named_parameters())
            exact = {n: torch.zeros_like(p) for n, p in shared.items()}
            noise = {n: torch.zeros_like(p) for n, p in shared.items()} if norm_ratio is not None else None
            transition_backward(model, x, y, cfg, noise, exact, fraction,
                                step=step, micro=0, rank=rank, accumulation=1)
            averaged_fo = {n: p.grad.clone() for n, p in shared.items()}
            averaged_exact = {n: g.clone() for n, g in exact.items()}
            averaged_noise = {n: g.clone() for n, g in noise.items()} if noise is not None else {}
            for pack in (averaged_fo, averaged_exact, averaged_noise):
                for tensor in pack.values():
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                    tensor.div_(world)
            fo_norm = sum(g.double().square().sum() for g in averaged_fo.values()).sqrt()
            noise_norm = sum(g.double().square().sum() for g in averaged_noise.values()) ** 0.5
            scale = fraction * norm_ratio * fo_norm.item() / noise_norm if noise_norm else 0.0
            mix_transition_gradients(model, noise, exact, norm_ratio, fraction)
            for n, p in shared.items():
                actual = p.grad.clone()
                dist.all_reduce(actual, op=dist.ReduceOp.SUM)
                actual.div_(world)
                expected = averaged_fo[n] + (1-fraction)*averaged_exact[n]
                if noise is not None:
                    expected = expected + scale*averaged_noise[n]
                torch.testing.assert_close(actual, expected)
        elif norm_ratio is None:
            backward_microbatch(model, x, y, cfg, step=step, micro_step=0, rank=rank, accumulation_steps=1)
        else:
            buffers = {n: torch.zeros_like(p) for n, p in model.shared_named_parameters()}
            separated_backward(model, x, y, cfg, buffers, step=step, micro=0, rank=rank, accumulation=1)
            # Build an independent average-then-scale reference before the helper
            # mutates either buffer. The optimizer must not divide by world twice.
            averaged_fo, averaged_zo = {}, {}
            for n, p in model.shared_named_parameters():
                averaged_fo[n], averaged_zo[n] = p.grad.clone(), buffers[n].clone()
                for tensor in (averaged_fo[n], averaged_zo[n]):
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                    tensor.div_(world)
            fo_norm = sum(g.double().square().sum() for g in averaged_fo.values()).sqrt()
            zo_norm = sum(g.double().square().sum() for g in averaged_zo.values()).sqrt()
            scale = (norm_ratio * fo_norm / zo_norm).item() if zo_norm else 0.0
            mix_shared_gradients(model, buffers, norm_ratio)
            for n, p in model.shared_named_parameters():
                torch.testing.assert_close(p.grad, averaged_fo[n] + scale * averaged_zo[n])
        for name, p in model.named_parameters():
            averaged = p.grad.clone()
            dist.all_reduce(averaged, op=dist.ReduceOp.SUM)
            averaged /= world
            reference_params[name].grad = averaged
        expected_optimizer.step()
        optimizer.step()
        for name, p in model.named_parameters():
            torch.testing.assert_close(p, reference_params[name], rtol=1e-5, atol=1e-6,
                                       msg=f"{prelude_mode}, step {step}, {name}")
            from_rank_zero = p.detach().clone()
            dist.broadcast(from_rank_zero, src=0)
            assert torch.equal(p, from_rank_zero), f"Rank parameters drifted: {name} ({prelude_mode})"
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
    if rank == 0:
        print(f"PASS {prelude_mode}, norm_ratio={norm_ratio}, transition={fraction}: averaging, single update, identical rank parameters")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prelude-mode", choices=["all", *PRELUDE_MODES], default="all")
    parser.add_argument("--norm-ratio", type=float)
    parser.add_argument("--transition-fraction", type=float)
    args = parser.parse_args()
    _, rank, _, world, device = compute_init("cuda")
    try:
        if world < 2:
            raise ValueError("Launch with at least two GPU ranks")
        torch._dynamo.config.disable = True
        modes = (["none"] if args.norm_ratio is not None or args.transition_fraction is not None else
                 (PRELUDE_MODES if args.prelude_mode == "all" else [args.prelude_mode]))
        for mode in modes:
            check_variant(mode, rank, world, device, args.norm_ratio, args.transition_fraction)
    finally:
        compute_cleanup()


if __name__ == "__main__":
    main()
