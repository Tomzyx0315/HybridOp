"""Train three gradient variants with the upstream model/data/optimizer recipe."""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

from hybrid.config import (default_run_name, execution_plan, lr_multiplier, normalize_zo_settings,
                           parse_args, recipe, transition_fraction)


def main(argv=None):
    cfg = parse_args(argv)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    resolved = recipe(cfg, world_size)
    if cfg.dry_run:
        print(json.dumps({"config": asdict(cfg), "world_size": world_size, "resolved": resolved,
                          "execution_plan": execution_plan(cfg)}, indent=2))
        return
    # A dry run works without torch, CUDA, a tokenizer download, or a dataset.
    from hybrid.diagnostics import stall_tracebacks
    with stall_tracebacks() as heartbeat:
        run_training(cfg, resolved, heartbeat=heartbeat)


def model_config(cfg):
    from nanochat.gpt import GPTConfig
    return GPTConfig(sequence_len=cfg.sequence_len, vocab_size=cfg.vocab_size, size=cfg.size,
                     n_head=cfg.width // cfg.head_dim, n_kv_head=cfg.width // cfg.head_dim,
                     n_embd=cfg.width, n_prelude=cfg.n_prelude, n_recur_block=cfg.n_recur_block,
                     n_coda=cfg.n_coda, num_recur=cfg.num_recur, bptt_k=cfg.num_recur,
                     input_injection="inject_init_prelude")


def build_optimizer(model, cfg, resolved):
    return model.setup_optimizer(
        matrix_lr=cfg.matrix_lr * resolved["batch_lr_scale"] * resolved["data_lr_scale"],
        embedding_lr=cfg.embedding_lr * resolved["batch_lr_scale"],
        unembedding_lr=cfg.unembedding_lr * resolved["batch_lr_scale"],
        d_ref=10 * cfg.aspect_ratio,
    )


def validate_data(cfg, path):
    from nanochat.tokenizer import get_tokenizer
    tokenizer = get_tokenizer()
    if tokenizer.get_vocab_size() != cfg.vocab_size:
        raise ValueError("Configured vocabulary disagrees with the upstream tokenizer")
    for split in ("train", "val"):
        if not list(path.glob(f"{split}-*.parquet")):
            raise FileNotFoundError(f"No {split} shards in {path}; prepare upstream prepacked data first")
        meta = path / f"meta_{split}.json"
        if meta.exists():
            info = json.loads(meta.read_text())
            if info["row_capacity"] != cfg.sequence_len + 1:
                raise ValueError(f"Context length mismatch in {meta}")
            if "vocab_size" in info and info["vocab_size"] != cfg.vocab_size:
                raise ValueError(f"Vocabulary mismatch in {meta}; verify the tokenizer/token IDs before training")


def synthetic_loader(cfg, device, rank, state=None, validation=False):
    """Deterministic smoke-test data; completely separate from real experiments."""
    import torch
    from hybrid.model import derive_seed
    batch_index = (state or {}).get("batch_index", 0)
    while True:
        generator = torch.Generator().manual_seed(derive_seed("data", cfg.seed, rank, batch_index, validation))
        row = torch.randint(cfg.vocab_size, (cfg.device_batch_size, cfg.sequence_len + 1), generator=generator)
        row = row.to(device)
        batch_index += 1
        yield row[:, :-1].contiguous(), row[:, 1:].contiguous(), {"batch_index": batch_index}


def load_rank_state(directory, step, rank):
    import torch
    # Tensor-only and primitive metadata, so keep torch.load's safe loader enabled.
    return torch.load(directory / f"training_{step:06d}_rank{rank}.pt", map_location="cpu", weights_only=True)


def validate_resume(cfg, meta, world_size):
    if meta.get("norm_match_experiment", {}).get("ratio") is not None:
        raise ValueError("This short-fork checkpoint stores norm matching outside user_config; resume from a hybrid.train checkpoint instead")
    if meta["world_size"] != world_size:
        raise ValueError("Resuming a sharded optimizer requires the original world size")
    allowed = {"output_dir", "data_dir", "resume_step", "stop_after", "eval_every", "eval_rows",
               "final_eval_rows", "save_every", "log_every", "run", "dry_run", "compile"}
    saved_config = dict(meta["user_config"])
    saved_config.setdefault("zo_norm_ratio", None)
    saved_config.setdefault("fo_checkpoint_dir", "")
    saved_config.setdefault("fo_checkpoint_step", -1)
    saved_config.setdefault("transition_steps", 0)
    # Checkpoints from the original Hybrid implementation predate these fields.
    saved_config.setdefault("prelude_mode", "none")
    if "zo_rho" not in saved_config:
        saved_config.setdefault("zo_input_epsilon", 1e-3)
    saved_config = normalize_zo_settings(saved_config)
    for key, value in asdict(cfg).items():
        if key not in allowed and saved_config[key] != value:
            raise ValueError(f"Resume changes experimental setting {key}; use a new run instead")


def validate_fo_fork(cfg, meta, world_size, resolved):
    if meta["user_config"]["mode"] != "full_fo" or meta["step"] != cfg.fo_checkpoint_step:
        raise ValueError("Fork source must be the requested full-FO checkpoint")
    source_cfg = replace(cfg, mode="full_fo", zo_norm_ratio=None, fo_checkpoint_dir="",
                         fo_checkpoint_step=-1, transition_steps=0)
    validate_resume(source_cfg, meta, world_size)
    for key in ("num_iterations", "grad_accum_steps", "initial_lrs", "total_parameters", "actual_tokens"):
        if meta["resolved"][key] != resolved[key]:
            raise ValueError(f"FO fork changes original recipe: {key}")


def run_training(cfg, resolved, heartbeat=lambda: None):
    # Triton/Inductor artifacts can race when ranks compile simultaneously on a
    # shared filesystem. Keep each rank's artifacts under the configured root.
    # This must happen before importing torch, which reads cache environment vars.
    if cfg.compile and "RANK" in os.environ:
        for name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
            if os.environ.get(name):
                os.environ[name] = str(Path(os.environ[name]) / f"rank{int(os.environ['RANK'])}")
    import torch
    import torch.distributed as dist
    from hybrid.model import ExperimentGPT, training_precision
    from hybrid.steps import backward_microbatch, estimated_flops_per_token
    from hybrid.norm_matching import mix_shared_gradients, separated_backward
    from hybrid.transition import (mix_transition_gradients, transition_backward,
                                   transition_flops_per_token)
    from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
    from nanochat.common import compute_cleanup, compute_init, get_base_dir, print0
    from nanochat.dataloader import prepacked_data_loader, prepacked_eval_loader
    from nanochat.loss_eval import evaluate_loss

    if cfg.device != "cuda" and all(key in os.environ for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        raise ValueError("CPU tests must use python -m hybrid.train directly, without torchrun; distributed training uses CUDA/NCCL")
    _, rank, _, world_size, device = compute_init(cfg.device)
    failed = False
    try:
        torch.manual_seed(cfg.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(cfg.seed)
        if not cfg.compile:
            # Also disable upstream @torch.compile optimizer kernels for tiny tests.
            torch._dynamo.config.disable = True
        config = model_config(cfg)
        with torch.device("meta"):
            model = ExperimentGPT(config)
        model.to_empty(device=device)
        model.init_weights()
        actual_count = sum(p.numel() for p in model.parameters())
        if actual_count != resolved["total_parameters"]:
            raise RuntimeError("Model parameter count no longer matches the recorded recipe")
        output = (Path(cfg.output_dir).expanduser() if cfg.output_dir else
                  Path(get_base_dir()) / "hybrid_runs" / default_run_name(cfg))
        # All ranks check before rank 0 creates the run. Never silently overwrite.
        if cfg.resume_step < 0 and output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Nonempty output directory: {output}; choose another or resume")
        if dist.is_initialized():
            dist.barrier()
        output.mkdir(parents=True, exist_ok=True)
        optimizer = build_optimizer(model, cfg, resolved)
        start_step = 0
        loader_state = None
        total_train_seconds = 0.0
        total_estimated_flops = 0
        if cfg.resume_step >= 0 or cfg.fo_checkpoint_dir:
            loading_resume = cfg.resume_step >= 0
            source = output if loading_resume else Path(cfg.fo_checkpoint_dir).expanduser()
            source_step = cfg.resume_step if loading_resume else cfg.fo_checkpoint_step
            weights, opt_state, meta = load_checkpoint(str(source), source_step, device,
                                                        load_optimizer=True, rank=rank)
            if loading_resume:
                validate_resume(cfg, meta, world_size)
            else:
                validate_fo_fork(cfg, meta, world_size, resolved)
            model.load_state_dict(weights, strict=True)
            optimizer.load_state_dict(opt_state)
            runtime = load_rank_state(source, source_step, rank)
            loader_state = runtime["dataloader_state"]
            start_step = meta["step"]
            total_train_seconds = meta["total_train_seconds"]
            total_estimated_flops = meta.get("estimated_total_flops",
                meta["resolved"]["estimated_flops_per_token"] * start_step * cfg.total_batch_size)
            torch.set_rng_state(runtime["cpu_rng"])
            if device.type == "cuda":
                torch.cuda.set_rng_state(runtime["cuda_rng"], device)
            del weights, opt_state, runtime
        if cfg.compile:
            # The repository already provides this basic optimization. It keeps
            # state_dict names intact and lets early loops run under no_grad.
            model.compile_blocks()
        model.train()
        if cfg.synthetic_data:
            train_loader = synthetic_loader(cfg, device, rank, loader_state)
        else:
            data_dir = (Path(cfg.data_dir).expanduser() if cfg.data_dir else
                        Path(get_base_dir()) / f"prepacked_T{cfg.sequence_len}_llama")
            validate_data(cfg, data_dir)
            train_loader = prepacked_data_loader(str(data_dir), cfg.device_batch_size, cfg.sequence_len,
                                                  device=cfg.device, resume_state=loader_state)
        steps = resolved["num_iterations"]
        end_step = min(steps, cfg.stop_after) if cfg.stop_after >= 0 else steps
        if end_step <= start_step:
            raise ValueError("No remaining steps; stop_after is an absolute optimizer-step limit")
        accumulation = resolved["grad_accum_steps"]
        zo_buffers = ({name: torch.zeros_like(p) for name, p in model.shared_named_parameters()}
                      if cfg.zo_norm_ratio is not None else None)
        exact_buffers = ({name: torch.zeros_like(p) for name, p in model.shared_named_parameters()}
                         if cfg.fo_checkpoint_dir and transition_fraction(start_step, cfg) < 1 else None)
        reports_input_zo = cfg.mode == "hybrid" and cfg.prelude_mode != "none" and cfg.zo_lambda != 0
        flops_per_token = estimated_flops_per_token(model, cfg)
        resolved = {**resolved, "estimated_flops_per_token": flops_per_token,
                    "execution_plan": execution_plan(cfg)}
        if rank == 0:
            (output / f"config_start_{start_step:06d}.json").write_text(json.dumps(
                {"config": asdict(cfg), "resolved": resolved, "world_size": world_size}, indent=2) + "\n")
            print0(json.dumps({"mode": cfg.mode, "output": str(output), **resolved}, indent=2))
        heartbeat()
        wandb_run = None
        if rank == 0 and cfg.run != "dummy":
            import wandb
            wandb_run = wandb.init(project="hybrid-loop", name=cfg.run,
                                    config={**asdict(cfg), **resolved})

        def emit(record):
            if rank == 0:
                with (output / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
                print0(json.dumps(record, allow_nan=False))
                if wandb_run is not None:
                    wandb_run.log(record, step=record["step"])

        def evaluate(step, final=False):
            rows = cfg.final_eval_rows if final else cfg.eval_rows
            stride = cfg.device_batch_size * world_size
            eval_steps = rows // stride
            if eval_steps < 1:
                raise ValueError("eval_rows is smaller than one distributed microbatch")
            if cfg.synthetic_data:
                batches = ((x, y) for x, y, _ in synthetic_loader(cfg, device, rank, validation=True))
            else:
                batches = prepacked_eval_loader(str(data_dir), cfg.device_batch_size, cfg.sequence_len,
                                                 device=cfg.device)
            model.eval()
            with training_precision(device):
                metrics = evaluate_loss(model, batches, eval_steps)
            model.train()
            emit(dict(event="validation", step=step, tokens=step * cfg.total_batch_size,
                      val_loss=metrics.loss, val_ppl=metrics.ppl, evaluated_rows=eval_steps * stride))
            heartbeat()

        def save(step):
            runtime = dict(dataloader_state=loader_state, cpu_rng=torch.get_rng_state())
            if device.type == "cuda":
                runtime["cuda_rng"] = torch.cuda.get_rng_state(device)
            # The upstream optimizer shards state; the data cursor is also rank-local.
            torch.save(runtime, output / f"training_{step:06d}_rank{rank}.pt")
            save_checkpoint(str(output), step, model.state_dict(), optimizer.state_dict(),
                            dict(step=step, model_config=asdict(config), user_config=asdict(cfg),
                                 resolved=resolved, world_size=world_size,
                                 total_train_seconds=total_train_seconds,
                                 estimated_total_flops=total_estimated_flops), rank=rank)
            if dist.is_initialized():
                dist.barrier()
            heartbeat()

        if cfg.eval_every > 0:
            evaluate(start_step)
        for step in range(start_step, end_step):
            fraction = transition_fraction(step, cfg)
            if fraction == 1:
                exact_buffers = None
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            loss_sum = torch.zeros((), device=device)
            zo_abs = torch.zeros((), device=device)
            boundary_norm = torch.zeros((), device=device)
            input_abs = torch.zeros((), device=device)
            input_norm = torch.zeros((), device=device)
            direct_norm = torch.zeros((), device=device)
            if zo_buffers is not None:
                for buffer in zo_buffers.values():
                    buffer.zero_()
            if exact_buffers is not None:
                for buffer in exact_buffers.values():
                    buffer.zero_()
            for micro in range(accumulation):
                x, y, loader_state = next(train_loader)
                if cfg.fo_checkpoint_dir:
                    loss, extra = transition_backward(model, x, y, cfg, zo_buffers, exact_buffers,
                        fraction, step=step, micro=micro, rank=rank, accumulation=accumulation)
                elif zo_buffers is None:
                    loss, extra = backward_microbatch(model, x, y, cfg, step=step, micro_step=micro,
                                                       rank=rank, accumulation_steps=accumulation)
                else:
                    loss, extra = separated_backward(model, x, y, cfg, zo_buffers, step=step,
                        micro=micro, rank=rank, accumulation=accumulation, return_metrics=True)
                loss_sum += loss / accumulation
                zo_abs += extra.get("zo_directional_abs", 0) / accumulation
                boundary_norm += extra.get("boundary_grad_norm", 0) / accumulation
                input_abs += extra.get("zo_input_directional_abs", 0) / accumulation
                input_norm += extra.get("zo_input_grad_norm", 0) / accumulation
                direct_norm += extra.get("prelude_direct_grad_norm", 0) / accumulation
            if cfg.fo_checkpoint_dir:
                match_metrics = mix_transition_gradients(model, zo_buffers, exact_buffers,
                                                         cfg.zo_norm_ratio, fraction)
                match_metrics.update(transition_fraction=fraction, early_exact_weight=1-fraction)
            else:
                match_metrics = (mix_shared_gradients(model, zo_buffers, cfg.zo_norm_ratio)
                                 if zo_buffers is not None else {})
            # Ordinary modes average the complete gradient inside the optimizer.
            # Norm matching first averages the shared FO/ZO buffers separately;
            # their identical mixed replicas are preserved by the optimizer AVG.
            squared_norm = torch.zeros((), device=device)
            for parameter in model.parameters():
                if parameter.grad is None:
                    raise RuntimeError("A trainable parameter has no gradient")
                squared_norm += parameter.grad.float().square().sum()
            finite = torch.isfinite(squared_norm) & torch.isfinite(loss_sum)
            finite_flag = finite.to(torch.int32)
            if dist.is_initialized():
                dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
            if not finite_flag.item():
                raise FloatingPointError(f"Non-finite loss/gradient at step {step}; optimizer was not updated")
            multiplier = lr_multiplier(step, steps, cfg)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * multiplier
            optimizer.step()
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            # Report the slowest rank's elapsed time / largest rank's memory.
            maxima = torch.tensor([elapsed, squared_norm.sqrt().item(),
                                   torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                                   torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0],
                                  device=device, dtype=torch.float64)
            averages = torch.stack((loss_sum, zo_abs, boundary_norm, input_abs, input_norm, direct_norm))
            if dist.is_initialized():
                dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
                dist.all_reduce(averages, op=dist.ReduceOp.SUM)
                averages /= world_size
            seconds = maxima[0].item()
            total_train_seconds += seconds
            heartbeat()
            step_flops = (transition_flops_per_token(model, cfg, fraction)
                          if cfg.fo_checkpoint_dir else flops_per_token)
            total_estimated_flops += step_flops * cfg.total_batch_size
            completed = step + 1
            if completed % cfg.log_every == 0 or completed == start_step + 1 or completed == end_step:
                emit(dict(event="train", step=completed, mode=cfg.mode, prelude_mode=cfg.prelude_mode,
                          tokens=completed * cfg.total_batch_size, train_loss=averages[0].item(),
                          step_seconds=seconds, tokens_per_second=cfg.total_batch_size / seconds,
                          total_train_seconds=total_train_seconds, grad_norm_local_max=maxima[1].item(),
                          peak_allocated_bytes=maxima[2].item(), peak_reserved_bytes=maxima[3].item(),
                          zo_directional_abs=averages[1].item(), boundary_grad_norm=averages[2].item(),
                          zo_input_directional_abs=averages[3].item() if reports_input_zo else None,
                          zo_input_grad_norm=averages[4].item() if reports_input_zo else None,
                          prelude_direct_grad_norm=averages[5].item() if reports_input_zo else None,
                          estimated_total_flops=total_estimated_flops,
                          lr_matrix=optimizer.param_groups[3]["lr"],
                          lr_embedding=optimizer.param_groups[1]["lr"],
                          zo_epsilon=cfg.zo_epsilon, zo_input_epsilon=cfg.zo_input_epsilon, zo_rho=cfg.zo_rho,
                          zo_lambda=cfg.zo_lambda, zo_norm_ratio=cfg.zo_norm_ratio,
                          **match_metrics))
            final = completed == end_step
            if cfg.eval_every > 0 and (completed % cfg.eval_every == 0 or final):
                evaluate(completed, final=completed == steps)
            if final or (cfg.save_every > 0 and completed % cfg.save_every == 0):
                save(completed)
        if wandb_run is not None:
            wandb_run.finish()
    except BaseException:
        failed = True
        # Cleanup can itself wait for other NCCL ranks. Preserve the original
        # Python exception before entering it, rather than losing it in a timeout.
        traceback.print_exc()
        raise
    finally:
        # A failed rank must exit so torchrun can terminate its peers. Calling
        # collective NCCL shutdown here can hang while peers wait in all_reduce,
        # concealing the original exception behind a later watchdog timeout.
        if not failed:
            compute_cleanup()


if __name__ == "__main__":
    main()
