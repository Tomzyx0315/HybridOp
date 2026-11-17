"""Compare local finite differences to an exact local VJP at a saved checkpoint.

This diagnostic deliberately records an early-loop backward graph. It is not
used by the memory-saving training path. Run on the experiment machine.
"""

import argparse
import json
import math
from dataclasses import fields
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--epsilons", nargs="+", type=float, default=[3e-4, 1e-3, 3e-3])
    parser.add_argument("--directions", type=int, default=3)
    parser.add_argument("--probe-kind", choices=["parameter", "input", "joint"], default="parameter")
    scale = parser.add_mutually_exclusive_group()
    scale.add_argument("--input-epsilon", type=float,
                       help="Fixed joint input scale; otherwise use rho * each scanned parameter epsilon")
    scale.add_argument("--zo-rho", type=float, help="Joint input/parameter scale ratio; default: checkpoint ratio")
    args = parser.parse_args()
    if any(not math.isfinite(epsilon) or epsilon <= 0 for epsilon in args.epsilons) or args.directions < 1:
        parser.error("epsilons must be finite and positive; directions must be positive")
    if args.input_epsilon is not None:
        if args.probe_kind != "joint":
            parser.error("--input-epsilon is only used by --probe-kind joint; input-only probes use --epsilons")
        if not math.isfinite(args.input_epsilon) or args.input_epsilon <= 0:
            parser.error("input-epsilon must be finite and positive")
    if args.zo_rho is not None:
        if args.probe_kind != "joint" or not math.isfinite(args.zo_rho) or args.zo_rho <= 0:
            parser.error("zo-rho must be finite and positive and is only used by joint probes")

    import torch
    from hybrid.config import Config, normalize_zo_settings
    from hybrid.model import ExperimentGPT, GaussianDirection, derive_seed, training_precision
    from hybrid.steps import local_probe_difference, suffix_backward
    from hybrid.train import model_config, synthetic_loader, validate_data
    from nanochat.checkpoint_manager import load_checkpoint
    from nanochat.common import compute_cleanup, compute_init, get_base_dir
    from nanochat.dataloader import prepacked_data_loader

    _, rank, _, world, device = compute_init(args.device)
    try:
        if world != 1:
            raise ValueError("This diagnostic runs on a single GPU; do not launch multiple ranks")
        torch._dynamo.config.disable = True
        weights, _, meta = load_checkpoint(str(args.checkpoint_dir), args.step, device)
        defaults = Config()
        saved_config = dict(meta["user_config"])
        if "zo_rho" not in saved_config:
            saved_config.setdefault("zo_input_epsilon", 1e-3)
        saved_config = normalize_zo_settings(saved_config)
        cfg = Config(**{field.name: saved_config.get(field.name, getattr(defaults, field.name))
                        for field in fields(Config)})
        with torch.device("meta"):
            model = ExperimentGPT(model_config(cfg))
        model.to_empty(device=device)
        model.init_weights()
        model.load_state_dict(weights)
        del weights
        if cfg.synthetic_data:
            loader = synthetic_loader(cfg, device, rank)
        else:
            directory = Path(args.data_dir or cfg.data_dir or (Path(get_base_dir()) / f"prepacked_T{cfg.sequence_len}_llama"))
            validate_data(cfg, directory)
            loader = prepacked_data_loader(str(directory), cfg.device_batch_size, cfg.sequence_len,
                                             device=args.device)
        x, y, _ = next(loader)
        model.train()
        loss, e, v = suffix_backward(model, x, y, cfg.fo_loops, need_boundary=True)
        norm = v.float().norm().item()
        print(json.dumps({"loss": loss.item(), "boundary_grad_norm": norm, "checkpoint_step": args.step}))
        if norm == 0:
            raise ValueError("Boundary gradient is zero; use a checkpoint after the initial [I,0] state")
        model.zero_grad(set_to_none=True)
        early = cfg.num_recur - cfg.fo_loops
        input_leaf = e.detach().requires_grad_(True)
        with training_precision(device):
            h = model.early_loops(input_leaf, early)
        (h.float() * v.float()).sum().backward()
        seeds = [derive_seed(cfg.zo_seed, "diagnostic", i) for i in range(args.directions)]
        exact = []
        with torch.no_grad():
            for seed in seeds:
                direction = GaussianDirection(seed)
                projection = sum((p.grad.float() * direction.noise(name, p).float()).sum()
                                  for name, p in model.shared_named_parameters())
                input_projection = (input_leaf.grad.float() * direction.noise("prelude_output", e.float())).sum()
                exact.append((projection.item(), input_projection.item()))
        model.zero_grad(set_to_none=True)
        del h, input_leaf
        for seed, (parameter_projection, input_projection) in zip(seeds, exact):
            for precision in (["bf16", "fp32"] if device.type == "cuda" else ["fp32"]):
                for epsilon in args.epsilons:
                    direction = GaussianDirection(seed)
                    rho = args.zo_rho if args.zo_rho is not None else cfg.zo_rho
                    input_epsilon = (args.input_epsilon if args.input_epsilon is not None else rho * epsilon
                                     ) if args.probe_kind == "joint" else epsilon
                    use_parameter = args.probe_kind != "input"
                    use_input = args.probe_kind != "parameter"
                    difference = local_probe_difference(
                        model, e, v, early,
                        parameter_direction=direction if use_parameter else None,
                        parameter_epsilon=epsilon if use_parameter else 0.0,
                        input_direction=direction if use_input else None,
                        input_epsilon=input_epsilon if use_input else 0.0,
                        probe_dtype=precision).item()
                    if args.probe_kind == "joint":
                        # Compare the joint half-difference, including BOTH projections.
                        reference = epsilon * parameter_projection + input_epsilon * input_projection
                        estimate = difference / 2
                    else:
                        reference = parameter_projection if use_parameter else input_projection
                        estimate = difference / (2 * epsilon)
                    print(json.dumps(dict(seed=seed, probe_kind=args.probe_kind, probe_dtype=precision,
                                          epsilon=epsilon, input_epsilon=input_epsilon if use_input else None,
                                          exact_local_projection=reference, estimated_projection=estimate,
                                          absolute_error=abs(estimate - reference),
                                          relative_error=abs(estimate - reference) / max(abs(reference), 1e-12))))
    finally:
        compute_cleanup()


if __name__ == "__main__":
    main()
