"""Experiment configuration and the upstream HyperP recipe (no torch import)."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path


PRELUDE_MODES = ("none", "separate", "joint", "joint_recompute")


@dataclass
class Config:
    mode: str = "full_fo"
    size: int = 16
    aspect_ratio: int = 64
    head_dim: int = 128
    sequence_len: int = 2048
    vocab_size: int = 32001
    n_prelude: int = 2
    n_recur_block: int = 8
    n_coda: int = 2
    num_recur: int = 2
    fo_loops: int = 1
    total_batch_size: int = 262144
    device_batch_size: int = 2
    num_iterations: int = -1
    target_param_data_ratio: float = 20.0
    embedding_lr: float = 0.3
    unembedding_lr: float = 0.004
    matrix_lr: float = 0.014
    warmup_ratio: float = 0.0
    warmdown_ratio: float = 1.0
    final_lr_frac: float = 0.1
    zo_epsilon: float = 1e-3
    zo_lambda: float = 1.0
    zo_norm_ratio: float | None = None
    zo_directions: int = 1
    zo_seed: int = 12345
    zo_probe_dtype: str = "bf16"
    prelude_mode: str = "none"
    zo_rho: float = 1.0
    seed: int = 42
    device: str = "cuda"
    compile: bool = True
    data_dir: str = ""
    output_dir: str = ""
    run: str = "dummy"
    eval_every: int = 250
    eval_rows: int = 4992
    final_eval_rows: int = 34560
    save_every: int = -1
    log_every: int = 10
    resume_step: int = -1
    stop_after: int = -1
    synthetic_data: bool = False
    dry_run: bool = False

    @property
    def width(self):
        return math.ceil(self.size * self.aspect_ratio / self.head_dim) * self.head_dim

    @property
    def zo_input_epsilon(self):
        return self.zo_rho * self.zo_epsilon

    def validate(self):
        if self.mode not in {"full_fo", "truncated", "hybrid"}:
            raise ValueError("Unknown training mode")
        if self.prelude_mode not in PRELUDE_MODES:
            raise ValueError("Unknown prelude mode")
        if self.mode != "hybrid" and self.prelude_mode != "none":
            raise ValueError("prelude_mode is only applicable to hybrid training")
        if self.device not in {"cuda", "cpu"} or self.zo_probe_dtype not in {"bf16", "fp32"}:
            raise ValueError("Unsupported device or probe dtype")
        for key in ("size", "aspect_ratio", "head_dim", "sequence_len", "vocab_size",
                    "n_recur_block", "num_recur", "total_batch_size", "device_batch_size",
                    "zo_directions", "eval_rows", "final_eval_rows", "log_every"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.n_prelude < 0 or self.n_coda < 0 or self.head_dim % 2:
            raise ValueError("Layer counts must be nonnegative; head_dim must be even")
        if not 1 <= self.fo_loops <= self.num_recur:
            raise ValueError("fo_loops must be in [1, num_recur]")
        if self.mode != "full_fo" and self.fo_loops == self.num_recur:
            raise ValueError("truncated/hybrid require at least one early loop")
        for key in ("embedding_lr", "unembedding_lr", "matrix_lr", "zo_epsilon", "zo_rho", "zo_input_epsilon",
                    "target_param_data_ratio"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not math.isfinite(self.zo_lambda) or self.zo_lambda < 0:
            raise ValueError("zo_lambda must be finite and nonnegative")
        if self.zo_norm_ratio is not None:
            if not math.isfinite(self.zo_norm_ratio) or self.zo_norm_ratio < 0:
                raise ValueError("zo_norm_ratio must be finite and nonnegative")
            if self.mode != "hybrid" or self.prelude_mode != "none" or self.zo_lambda != 1:
                raise ValueError("Norm matching requires hybrid, prelude_mode=none, zo_lambda=1")
        if not (0 <= self.warmup_ratio <= 1 and 0 <= self.warmdown_ratio <= 1
                and self.warmup_ratio + self.warmdown_ratio <= 1):
            raise ValueError("Invalid warmup/warmdown fractions")
        if not 0 <= self.final_lr_frac <= 1:
            raise ValueError("final_lr_frac must be in [0, 1]")
        if self.num_iterations != -1 and self.num_iterations <= 0:
            raise ValueError("num_iterations must be -1 or positive")
        if self.resume_step < -1 or self.stop_after < -1:
            raise ValueError("resume_step/stop_after must be -1 or nonnegative")
        if self.num_recur < 2:
            raise ValueError("These comparisons require at least two recurrences")


def default_run_name(cfg: Config):
    # Keep existing directories for the original three runs. New prelude
    # experiments get distinct directories so they cannot overwrite each other.
    name = f"loop{cfg.num_recur}_s{cfg.size}_{cfg.mode}"
    if cfg.mode == "hybrid" and cfg.prelude_mode != "none":
        name += f"_prelude_{cfg.prelude_mode}"
    if cfg.zo_norm_ratio is not None:
        name += f"_norm{cfg.zo_norm_ratio:g}"
    return f"{name}_seed{cfg.seed}"


def execution_plan(cfg: Config):
    active = cfg.mode == "hybrid" and cfg.zo_lambda != 0
    probes = 2 * cfg.zo_directions if active else 0
    return dict(parameter_or_joint_probe_pairs=probes // 2,
                input_only_probe_pairs=probes // 2 if cfg.prelude_mode == "separate" else 0,
                extra_prelude_forwards=int(active and cfg.prelude_mode == "joint_recompute"),
                prelude_backwards=1,
                zo_rho=cfg.zo_rho, zo_input_epsilon=cfg.zo_input_epsilon,
                zo_norm_ratio=cfg.zo_norm_ratio,
                output_name=default_run_name(cfg))


def normalize_zo_settings(settings):
    """Convert an older absolute input epsilon into the canonical ratio setting."""
    result = dict(settings)
    if "zo_input_epsilon" in result:
        epsilon = result.get("zo_epsilon", Config().zo_epsilon)
        input_epsilon = result.pop("zo_input_epsilon")
        if not math.isfinite(epsilon) or epsilon <= 0 or not math.isfinite(input_epsilon) or input_epsilon <= 0:
            raise ValueError("ZO epsilons must be finite and positive")
        rho = input_epsilon / epsilon
        if "zo_rho" in result and not math.isclose(result["zo_rho"], rho, rel_tol=1e-12):
            raise ValueError("Conflicting zo_rho and zo_input_epsilon")
        result["zo_rho"] = rho
    return result


def parameter_counts(width, prelude, recur, coda, loops, vocab_size):
    """Exact for the unchanged MHA/ReLU²/learned-injection upstream model."""
    padded_vocab = math.ceil(vocab_size / 64) * 64
    block = 12 * width**2 + 2 * width
    injection = 2 * width**2
    outer_norms = 3 * width
    total = (prelude + recur + coda) * block + injection + outer_norms + 2 * padded_vocab * width
    # Match GPT.effective_params(), including its convention for outer norms.
    effective = (prelude + loops * recur + coda) * block + loops * injection + outer_norms
    return total, effective


def recipe(cfg: Config, world_size: int):
    cfg.validate()
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    micro_tokens = world_size * cfg.device_batch_size * cfg.sequence_len
    if cfg.total_batch_size % micro_tokens:
        raise ValueError("total_batch_size must be divisible by world_size × microbatch × context")
    total, effective = parameter_counts(cfg.width, cfg.n_prelude, cfg.n_recur_block,
                                         cfg.n_coda, cfg.num_recur, cfg.vocab_size)
    # Preserve base_train.py exactly: reference physical layout is 2/4/2, and
    # get_scaling_params(ref_model) passes the TRAINING model's num_recur.
    ref_width = math.ceil(10 * cfg.aspect_ratio / cfg.head_dim) * cfg.head_dim
    _, ref_effective = parameter_counts(ref_width, 2, 4, 2, cfg.num_recur, cfg.vocab_size)
    d_ref = 10 * ref_effective
    target_tokens = (cfg.num_iterations * cfg.total_batch_size if cfg.num_iterations > 0
                     else int(cfg.target_param_data_ratio * effective))
    steps = cfg.num_iterations if cfg.num_iterations > 0 else target_tokens // cfg.total_batch_size
    if steps < 1:
        raise ValueError("Training horizon is shorter than one global batch")
    batch_scale = (cfg.total_batch_size / 262144) ** 0.558
    data_scale = (target_tokens / d_ref) ** -0.32
    width_scale = (cfg.width / (10 * cfg.aspect_ratio)) ** -0.5
    return dict(
        total_parameters=total, effective_parameters=effective, width=cfg.width,
        n_heads=cfg.width // cfg.head_dim, effective_depth=cfg.n_prelude + cfg.num_recur * cfg.n_recur_block + cfg.n_coda,
        num_iterations=steps, target_tokens=target_tokens, actual_tokens=steps * cfg.total_batch_size,
        grad_accum_steps=cfg.total_batch_size // micro_tokens, reference_tokens=d_ref,
        batch_lr_scale=batch_scale, data_lr_scale=data_scale,
        initial_lrs=dict(matrix=cfg.matrix_lr * batch_scale * data_scale,
                         embedding=cfg.embedding_lr * batch_scale * width_scale,
                         unembedding=cfg.unembedding_lr * batch_scale * width_scale,
                         norm=0.005 * width_scale),
    )


def lr_multiplier(step: int, steps: int, cfg: Config):
    warmup, warmdown = round(cfg.warmup_ratio * steps), round(cfg.warmdown_ratio * steps)
    if step < warmup:
        return (step + 1) / warmup
    if step <= steps - warmdown:
        return 1.0
    progress = (steps - step) / warmdown
    return progress + (1 - progress) * cfg.final_lr_frac


def parse_args(argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/loop2_219m.json")
    known, _ = preliminary.parse_known_args(argv)
    defaults = asdict(Config())
    supplied = normalize_zo_settings(json.loads(known.config.read_text()))
    unknown = supplied.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    defaults.update(supplied)
    parser = argparse.ArgumentParser(parents=[preliminary], description=__doc__)
    input_scale = parser.add_mutually_exclusive_group()
    input_scale.add_argument("--zo-input-epsilon", type=float,
                             help="Absolute input perturbation; converted to rho = input epsilon / zo epsilon")
    choices = dict(mode=["full_fo", "truncated", "hybrid"], device=["cuda", "cpu"],
                   zo_probe_dtype=["bf16", "fp32"], prelude_mode=PRELUDE_MODES)
    for field in fields(Config):
        default = getattr(Config(), field.name)
        kwargs = dict(default=defaults[field.name])
        if isinstance(default, bool):
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = float if field.name == "zo_norm_ratio" else type(default)
        if field.name in choices:
            kwargs["choices"] = choices[field.name]
        target = input_scale if field.name == "zo_rho" else parser
        if field.name == "zo_rho":
            kwargs["help"] = "Input/parameter perturbation ratio; input epsilon = rho * zo epsilon (default: 1)"
        if field.name == "zo_norm_ratio":
            kwargs["help"] = "Match global ZO norm to this multiple of shared suffix FO norm (default: disabled)"
        target.add_argument("--" + field.name.replace("_", "-"), **kwargs)
    namespace = vars(parser.parse_args(argv))
    namespace.pop("config")
    input_epsilon = namespace.pop("zo_input_epsilon")
    if input_epsilon is not None:
        namespace.pop("zo_rho")
        namespace = normalize_zo_settings({**namespace, "zo_input_epsilon": input_epsilon})
    cfg = Config(**namespace)
    cfg.validate()
    return cfg
