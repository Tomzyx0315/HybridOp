"""Bounded checkpoint forks comparing norm-matched local ZO with FO baselines.

This research runner keeps FO and ZO buffers separately and averages each before
norm matching. It deliberately spends extra memory/communication on measurement.
It never changes the source checkpoint or the ordinary training entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, replace
from pathlib import Path


from hybrid.norm_matching import mix_shared_gradients, separated_backward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=8000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ratios", nargs="+", type=float, default=[1.0, 0.3, 0.1])
    parser.add_argument("--eval-at", nargs="+", type=int, default=[50, 100, 200])
    args = parser.parse_args()
    if args.steps < 1 or any(not math.isfinite(x) or x < 0 for x in args.ratios):
        parser.error("Require positive steps and finite nonnegative ratios")
    for name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        if name in os.environ:
            os.environ[name] = str(Path(os.environ[name]) / f"rank{os.environ.get('RANK', '0')}")

    import torch
    import torch.distributed as dist
    from hybrid.config import Config, lr_multiplier
    from hybrid.model import ExperimentGPT, training_precision
    from hybrid.steps import backward_microbatch
    from hybrid.train import build_optimizer, load_rank_state, model_config
    from nanochat.checkpoint_manager import save_checkpoint
    from nanochat.common import compute_cleanup, compute_init, get_base_dir
    from nanochat.dataloader import prepacked_data_loader, prepacked_eval_loader
    from nanochat.loss_eval import evaluate_loss

    _, rank, _, world, device = compute_init("cuda")
    try:
        root = args.output_dir
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(root)
        dist.barrier()
        root.mkdir(parents=True, exist_ok=True)
        source = args.checkpoint_dir
        meta = json.loads((source / f"meta_{args.step:06d}.json").read_text())
        cfg = Config(**meta["user_config"])
        assert cfg.mode == "full_fo" and meta["world_size"] == world
        assert cfg.compile and cfg.num_recur == 2 and cfg.fo_loops == 1
        assert args.step + args.steps <= meta["resolved"]["num_iterations"]
        weights = torch.load(source / f"model_{args.step:06d}.pt", map_location="cpu", weights_only=True)
        opt_state = torch.load(source / f"optim_{args.step:06d}_rank{rank}.pt", map_location="cpu", weights_only=True)
        runtime = load_rank_state(source, args.step, rank)
        with torch.device("meta"):
            model = ExperimentGPT(model_config(cfg))
        model.to_empty(device=device)
        model.init_weights()
        model.compile_blocks()
        shared = dict(model.shared_named_parameters())
        data_dir = cfg.data_dir or str(Path(get_base_dir()) / f"prepacked_T{cfg.sequence_len}_llama")
        accumulation = meta["resolved"]["grad_accum_steps"]
        variants = [("full_fo", "full_fo", None), ("truncated", "truncated", None)]
        variants += [(f"norm_match_{ratio:g}", "hybrid", ratio) for ratio in args.ratios]
        variants.append(("hybrid_raw", "hybrid", None))
        assert len({v[0] for v in variants}) == len(variants)

        def emit(record, case=None):
            if rank == 0:
                line = json.dumps(record, allow_nan=False)
                print(line, flush=True)
                if case is not None:
                    with (case / "metrics.jsonl").open("a") as f:
                        f.write(line + "\n")

        def evaluate(label, case, offset):
            model.eval()
            batches = prepacked_eval_loader(data_dir, cfg.device_batch_size, cfg.sequence_len, device="cuda")
            with training_precision(device):
                val = evaluate_loss(model, batches, cfg.eval_rows // (world * cfg.device_batch_size))
            model.train()
            emit(dict(event="validation", variant=label, offset=offset, step=args.step + offset,
                      val_loss=val.loss, val_ppl=val.ppl, evaluated_rows=cfg.eval_rows), case)
            return val.loss

        completed = []
        initial_loss = None
        optimizer = None
        if rank == 0:
            (root / "manifest.json").write_text(json.dumps(dict(source=str(source), source_step=args.step,
                steps=args.steps, variants=variants, base_config=asdict(cfg), world_size=world,
                matching="global shared-parameter L2 norms, after rank averaging",
                optimizer_history="restored identically from full FO checkpoint"), indent=2) + "\n")
        for label, mode, ratio in variants:
            case = root / label
            case.mkdir(exist_ok=True)
            run_cfg = replace(cfg, mode=mode, prelude_mode="none", zo_lambda=1.0)
            model.load_state_dict(weights, strict=True)
            model.zero_grad(set_to_none=True)
            if optimizer is not None:
                del optimizer
            optimizer = build_optimizer(model, run_cfg, meta["resolved"])
            optimizer.load_state_dict(opt_state)
            torch.set_rng_state(runtime["cpu_rng"])
            torch.cuda.set_rng_state(runtime["cuda_rng"], device)
            loader = prepacked_data_loader(data_dir, cfg.device_batch_size, cfg.sequence_len,
                                          device="cuda", resume_state=runtime["dataloader_state"])
            zo_buffers = {n: torch.zeros_like(p) for n, p in shared.items()} if mode == "hybrid" else None
            emit(dict(event="start", variant=label, ratio=ratio, steps=args.steps))
            start_loss = evaluate(label, case, 0)
            if initial_loss is None:
                initial_loss = start_loss
            elif abs(start_loss - initial_loss) > 1e-6:
                raise RuntimeError(f"Fork initial loss mismatch: {start_loss} != {initial_loss}")
            run_seconds = 0.0
            for offset in range(1, args.steps + 1):
                step = args.step + offset - 1
                torch.cuda.synchronize()
                started = time.perf_counter()
                loss_sum = torch.zeros((), device=device)
                if zo_buffers is not None:
                    for z in zo_buffers.values():
                        z.zero_()
                digest = hashlib.sha256() if offset == 1 else None
                for micro in range(accumulation):
                    x, y, loader_state = next(loader)
                    if digest is not None:
                        digest.update(x.cpu().contiguous().numpy().tobytes())
                        digest.update(y.cpu().contiguous().numpy().tobytes())
                    if mode == "hybrid":
                        loss = separated_backward(model, x, y, run_cfg, zo_buffers, step=step,
                                                   micro=micro, rank=rank, accumulation=accumulation)
                    else:
                        loss, _ = backward_microbatch(model, x, y, run_cfg, step=step,
                            micro_step=micro, rank=rank, accumulation_steps=accumulation)
                    loss_sum += loss / accumulation
                extra = mix_shared_gradients(model, zo_buffers, ratio) if zo_buffers is not None else {}
                finite = torch.isfinite(loss_sum)
                for p in model.parameters():
                    assert p.grad is not None
                    finite &= torch.isfinite(p.grad).all()
                flag = finite.to(torch.int32)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                if not flag.item():
                    raise FloatingPointError(f"Non-finite loss or gradient: {label}, step {step}")
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"] * lr_multiplier(step, meta["resolved"]["num_iterations"], cfg)
                # Shared gradients are already identical across ranks after mixing;
                # the optimizer's AVG preserves them. Other params average here.
                optimizer.step()
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                duration = time.perf_counter() - started
                run_seconds += duration
                dist.all_reduce(loss_sum, op=dist.ReduceOp.AVG)
                if digest is not None:
                    digests = [None] * world
                    dist.all_gather_object(digests, digest.hexdigest())
                    if rank == 0:
                        (case / "first_batch_sha256.json").write_text(json.dumps(digests) + "\n")
                if offset == 1 or offset % 10 == 0 or offset == args.steps:
                    emit(dict(event="train", variant=label, offset=offset, step=step + 1,
                              train_loss=loss_sum.item(), step_seconds=duration, **extra), case)
                if offset in args.eval_at or offset == args.steps:
                    final_loss = evaluate(label, case, offset)
            checkpoint = case / "checkpoint"
            checkpoint.mkdir(exist_ok=True)
            torch.save(dict(dataloader_state=loader_state, cpu_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state(device)),
                       checkpoint / f"training_{args.step + args.steps:06d}_rank{rank}.pt")
            save_checkpoint(str(checkpoint), args.step + args.steps, model.state_dict(), optimizer.state_dict(),
                dict(step=args.step + args.steps, model_config=asdict(model.config), user_config=asdict(run_cfg),
                     resolved=meta["resolved"], world_size=world,
                     total_train_seconds=meta["total_train_seconds"] + run_seconds,
                     norm_match_experiment=dict(ratio=ratio, variant=label, source=str(source), source_step=args.step)), rank=rank)
            dist.barrier()
            completed.append(dict(variant=label, val_loss_start=start_loss, val_loss_end=final_loss,
                                  delta_val_loss=final_loss-start_loss, train_seconds=run_seconds))
            emit(dict(event="complete", **completed[-1]))
            if rank == 0:
                (root / "summary.json").write_text(json.dumps(completed, indent=2) + "\n")
            del zo_buffers
        emit(dict(event="all_complete", variants=completed))
    finally:
        compute_cleanup()


if __name__ == "__main__":
    main()
