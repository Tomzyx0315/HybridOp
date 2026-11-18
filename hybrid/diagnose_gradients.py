"""Inspect a saved looped-model checkpoint without saving model updates.

Run with the checkpoint's original world size. Replays one optimizer batch,
separates suffix FO, exact early-loop VJP, and random-direction estimates, then
optionally compares in-memory optimizer updates from the same saved state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--data-step", type=int,
                        help="Load data cursors from this step in the same checkpoint directory")
    parser.add_argument("--direction-step", type=int,
                        help="Use this step to seed probe directions for matched comparisons")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--updates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--match-ratios", nargs="+", type=float, default=[],
                        help="Also compare ZO rescaled to these multiples of the shared suffix FO norm")
    args = parser.parse_args()
    if any(not math.isfinite(r) or r < 0 for r in args.match_ratios):
        parser.error("match ratios must be finite and nonnegative")
    for name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        if name in os.environ:
            os.environ[name] = str(Path(os.environ[name]) / f"rank{os.environ.get('RANK', '0')}")

    import torch
    import torch.distributed as dist
    from hybrid.config import Config, lr_multiplier
    from hybrid.model import ExperimentGPT, GaussianDirection, derive_seed, probe_precision, training_precision
    from hybrid.steps import local_directional_derivative, suffix_backward
    from hybrid.train import build_optimizer, load_rank_state, model_config
    from nanochat.common import compute_cleanup, compute_init, get_base_dir
    from nanochat.dataloader import prepacked_data_loader

    _, rank, _, world, device = compute_init("cuda")
    try:
        if not args.compile:
            torch._dynamo.config.disable = True
        args.output_dir.mkdir(parents=True, exist_ok=True)
        meta = json.loads((args.checkpoint_dir / f"meta_{args.step:06d}.json").read_text())
        cfg = Config(**meta["user_config"])
        assert world == meta["world_size"], "Use the checkpoint's original world size"
        assert cfg.prelude_mode == "none" and cfg.zo_directions == 1
        assert 1 <= cfg.fo_loops < cfg.num_recur, "Diagnostic requires an early loop"
        data_step = args.step if args.data_step is None else args.data_step
        direction_step = args.step if args.direction_step is None else args.direction_step
        data_meta = json.loads((args.checkpoint_dir / f"meta_{data_step:06d}.json").read_text())
        assert data_meta["world_size"] == world
        for key in ("sequence_len", "device_batch_size", "total_batch_size", "seed"):
            assert data_meta["user_config"][key] == meta["user_config"][key], key
        weights = torch.load(args.checkpoint_dir / f"model_{args.step:06d}.pt", map_location="cpu", weights_only=True)
        with torch.device("meta"):
            model = ExperimentGPT(model_config(cfg))
        model.to_empty(device=device)
        model.init_weights()
        model.load_state_dict(weights, strict=True)
        model.train()
        if args.compile:
            model.compile_blocks()
        named = dict(model.named_parameters())
        shared = dict(model.shared_named_parameters())
        dimension = sum(p.numel() for p in shared.values())
        accumulation = meta["resolved"]["grad_accum_steps"]
        runtime = load_rank_state(args.checkpoint_dir, data_step, rank)
        loader = prepacked_data_loader(
            cfg.data_dir or str(Path(get_base_dir()) / f"prepacked_T{cfg.sequence_len}_llama"),
            cfg.device_batch_size, cfg.sequence_len, device="cuda", resume_state=runtime["dataloader_state"])
        packs = {"fo": {n: torch.zeros_like(p) for n, p in named.items()}}
        for key in ("early_exact", "zo_bf16", "zo_ideal", "zo_fp32"):
            packs[key] = {n: torch.zeros_like(p) for n, p in shared.items()}

        def emit(kind, **data):
            if rank == 0:
                print(json.dumps({"event": kind, **data}, allow_nan=False), flush=True)

        def norm(pack):
            return math.sqrt(sum(x.double().square().sum().item() for x in pack.values()))

        def dot(a, b):
            return sum((a[n].double() * b[n].double()).sum().item() for n in a.keys() & b.keys())

        def cosine(a, b):
            return dot(a, b) / max(norm(a) * norm(b), 1e-30)

        def averaged_scalar(value):
            v = torch.tensor(value, device=device, dtype=torch.float64)
            dist.all_reduce(v, op=dist.ReduceOp.AVG)
            return v.item()

        def projection(direction):
            return sum((p.grad.float() * direction.noise(n, p)).sum() for n, p in shared.items())

        micro_records, scans = [], []
        batch_digest = hashlib.sha256()
        micro_exact_squared = 0.0
        example = None
        emit("begin", checkpoint=str(args.checkpoint_dir), step=args.step, world_size=world,
             accumulation=accumulation, shared_dimension=dimension, compile=args.compile,
             checkpoint_mode=cfg.mode, data_step=data_step, direction_step=direction_step)
        for micro in range(accumulation):
            x, y, _ = next(loader)
            batch_digest.update(x.cpu().contiguous().numpy().tobytes())
            batch_digest.update(y.cpu().contiguous().numpy().tobytes())
            if example is None:
                example = (x.clone(), y.clone())
            model.zero_grad(set_to_none=True)
            loss, e, v = suffix_backward(model, x, y, cfg.fo_loops, 1 / accumulation, True)
            with torch.no_grad():
                for n, p in named.items():
                    assert p.grad is not None, n
                    packs["fo"][n].add_(p.grad)
            model.zero_grad(set_to_none=True)
            early = cfg.num_recur - cfg.fo_loops
            with training_precision(device):
                h = model.early_loops(e, early)
            (h.float() * v.float()).sum().backward()
            direction = GaussianDirection(derive_seed(cfg.zo_seed, direction_step, micro, rank, 0))
            exact_projection = projection(direction).item()
            current_exact_squared = sum(p.grad.float().square().sum().item() for p in shared.values())
            micro_exact_squared += current_exact_squared
            with torch.no_grad():
                for n, p in shared.items():
                    packs["early_exact"][n].add_(p.grad)
            model.zero_grad(set_to_none=True)
            del h
            bf16 = local_directional_derivative(model, e, v, early, direction, cfg.zo_epsilon, "bf16").item()
            fp32 = local_directional_derivative(model, e, v, early, direction, cfg.zo_epsilon, "fp32").item()
            with torch.no_grad():
                for n, p in shared.items():
                    z = direction.noise(n, p)
                    for key, coefficient in (("zo_bf16", bf16), ("zo_ideal", exact_projection), ("zo_fp32", fp32)):
                        packs[key][n].add_(z, alpha=coefficient)
            micro_records.append(dict(rank=rank, micro=micro, loss=loss.item(), exact_projection_bf16=exact_projection,
                                      bf16=bf16, fp32=fp32, early_exact_norm=math.sqrt(current_exact_squared)))
            if micro == 0 and args.scan:
                with probe_precision(device, "fp32"):
                    h = model.early_loops(e.float(), early)
                    (h * v.float()).sum().backward()
                reference_fp32 = projection(direction).item()
                model.zero_grad(set_to_none=True)
                del h
                for precision, reference in (("bf16", exact_projection), ("fp32", reference_fp32)):
                    for epsilon in (1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
                        estimate = local_directional_derivative(model, e, v, early, direction, epsilon, precision).item()
                        scans.append(dict(rank=rank, precision=precision, epsilon=epsilon,
                                          reference=reference, estimate=estimate))
            emit("micro_done", micro=micro, rank_zero=micro_records[-1])

        # These are exactly the pre-reduction quantities reported by training.
        combined = {n: packs["fo"][n] + packs["zo_bf16"].get(n, 0) for n in named}
        local = {"fo": norm(packs["fo"]), "early_exact": norm(packs["early_exact"]),
                 "zo_bf16": norm(packs["zo_bf16"]), "mixed": norm(combined)}
        del combined
        all_local, all_micro, all_scans = [None] * world, [None] * world, [None] * world
        dist.all_gather_object(all_local, local)
        dist.all_gather_object(all_micro, micro_records)
        dist.all_gather_object(all_scans, scans)
        all_digests = [None] * world
        dist.all_gather_object(all_digests, batch_digest.hexdigest())
        summed_micro_squared = averaged_scalar(micro_exact_squared) * world
        for pack in packs.values():
            for tensor in pack.values():
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
        fo_shared = {n: packs["fo"][n] for n in shared}
        exact = packs["early_exact"]
        oracle_shared = {n: fo_shared[n] + exact[n] for n in shared}
        result = {
            "checkpoint_directory": str(args.checkpoint_dir), "checkpoint_mode": cfg.mode,
            "checkpoint_step": args.step, "world_size": world, "accumulation": accumulation,
            "data_step": data_step, "direction_step": direction_step, "batch_sha256_by_rank": all_digests,
            "global_tokens": world * accumulation * cfg.device_batch_size * cfg.sequence_len,
            "compile": args.compile, "shared_dimension": dimension,
            "parameter_relative_perturbation_rms": cfg.zo_epsilon * math.sqrt(dimension) / norm(shared),
            "local_rank_norms": all_local, "global_norms": {k: norm(p) for k, p in packs.items()},
            "fo_shared_norm": norm(fo_shared), "oracle_shared_norm": norm(oracle_shared),
            "exact_early_cosine_to_suffix": cosine(exact, fo_shared),
            "ideal_gaussian_estimator_predicted_rms": math.sqrt(norm(exact)**2 + (dimension + 1) * summed_micro_squared / world**2),
            "projections": [item for row in all_micro for item in row],
            "precision_scan": [item for row in all_scans for item in row],
            "comparisons": {},
        }
        for key in ("zo_bf16", "zo_ideal", "zo_fp32"):
            pack = packs[key]
            mixed = {n: fo_shared[n] + pack[n] for n in shared}
            result["comparisons"][key] = dict(cosine_to_exact_early=cosine(pack, exact),
                                                mixed_shared_cosine_to_oracle=cosine(mixed, oracle_shared),
                                                norm_ratio_to_exact_early=norm(pack) / norm(exact))
        result["bf16_vs_ideal"] = dict(cosine=cosine(packs["zo_bf16"], packs["zo_ideal"]),
                                        relative_l2_error=math.sqrt(sum((packs["zo_bf16"][n]-packs["zo_ideal"][n]).double().square().sum().item() for n in shared)) / norm(packs["zo_ideal"]))
        norm_match_scale = norm(fo_shared) / max(norm(packs["zo_bf16"]), 1e-30)
        result["norm_matched"] = []
        for ratio in [0.0, *args.match_ratios]:
            mixed = {n: fo_shared[n] + ratio * norm_match_scale * packs["zo_bf16"][n] for n in shared}
            result["norm_matched"].append(dict(ratio=ratio, effective_lambda=ratio * norm_match_scale,
                                               mixed_norm=norm(mixed), cosine_to_oracle=cosine(mixed, oracle_shared)))
        emit("gradient_summary", **{k: v for k, v in result.items() if k not in {"projections", "precision_scan"}})
        if rank == 0:
            (args.output_dir / "gradients.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

        if args.updates:
            # Every counterfactual restores BOTH weights and the saved optimizer.
            # Already-averaged gradients are identical on all ranks, so the actual
            # optimizer's AVG reduction leaves them unchanged.
            optimizer = build_optimizer(model, cfg, meta["resolved"])
            saved_opt = torch.load(args.checkpoint_dir / f"optim_{args.step:06d}_rank{rank}.pt", map_location="cpu", weights_only=True)
            update_results = []
            for history in ("checkpoint", "zero_moments"):
                oracle_delta = None
                variants = [("oracle_early_parameters", exact, 1.0),
                                                   ("truncated", exact, 0.0),
                                                   ("hybrid_lambda_1", packs["zo_bf16"], 1.0),
                                                   ("hybrid_lambda_0.01", packs["zo_bf16"], 0.01),
                                                   ("hybrid_lambda_0.001", packs["zo_bf16"], 0.001)]
                variants += [(f"norm_match_{ratio:g}", packs["zo_bf16"], ratio * norm_match_scale)
                             for ratio in args.match_ratios]
                for label, correction, weight in variants:
                    model.load_state_dict(weights, strict=True)
                    optimizer.load_state_dict(saved_opt)
                    if history == "zero_moments":
                        for state in optimizer.state.values():
                            for key in ("momentum_buffer", "exp_avg", "exp_avg_sq"):
                                if key in state:
                                    state[key].zero_()
                            if "step" in state:
                                state["step"] = 0
                    for group in optimizer.param_groups:
                        group["lr"] = group["initial_lr"] * lr_multiplier(args.step, meta["resolved"]["num_iterations"], cfg)
                    for n, p in named.items():
                        p.grad = packs["fo"][n].clone()
                        if n in correction:
                            p.grad.add_(correction[n], alpha=weight)
                    optimizer.step()
                    delta = {n: p.detach() - weights[n].to(device) for n, p in shared.items()}
                    if oracle_delta is None:
                        oracle_delta = {n: p.clone() for n, p in delta.items()}
                    with torch.no_grad(), training_precision(device):
                        updated_loss = model(*example).item()
                    item = dict(history=history, variant=label, shared_update_norm=norm(delta),
                                effective_lambda=weight,
                                shared_update_cosine_to_oracle=cosine(delta, oracle_delta),
                                shared_gradient_dot_update=dot(oracle_shared, delta),
                                loss_first_micro_after_update=averaged_scalar(updated_loss))
                    update_results.append(item)
                    emit("update_counterfactual", **item)
                    model.zero_grad(set_to_none=True)
            result["optimizer_counterfactuals"] = update_results
        model.load_state_dict(weights, strict=True)
        emit("complete", output=str(args.output_dir), saved_model_updates=False)
        if rank == 0:
            (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    finally:
        compute_cleanup()


if __name__ == "__main__":
    main()
