"""Fade full first-loop credit into truncation or globally norm-matched ZO."""

from dataclasses import replace


def transition_backward(model, x, y, cfg, zo_buffers, exact_buffers, fraction, *,
                        step, micro, rank, accumulation):
    import torch
    from hybrid.model import derive_seed, training_precision
    from hybrid.norm_matching import separated_backward
    from hybrid.steps import (add_local_zo, backward_microbatch, deferred_suffix_backward,
                              finish_prelude_backward)

    if fraction == 0:
        return backward_microbatch(model, x, y, replace(cfg, mode="full_fo", zo_norm_ratio=None),
                                   step=step, micro_step=micro, rank=rank, accumulation_steps=accumulation)
    if fraction == 1:
        if zo_buffers is not None:
            return separated_backward(model, x, y, cfg, zo_buffers, step=step, micro=micro,
                                      rank=rank, accumulation=accumulation, return_metrics=True)
        return backward_microbatch(model, x, y, cfg, step=step, micro_step=micro,
                                   rank=rank, accumulation_steps=accumulation)
    if not 0 < fraction < 1 or exact_buffers is None:
        raise ValueError("Partial transition needs exact-gradient buffers and fraction in (0, 1)")

    # Keep suffix shared gradients separate: they define the normalization norm.
    # Prelude is replayed once after direct and indirect input cotangents combine.
    loss, e, v, direct, _ = deferred_suffix_backward(
        model, x, y, cfg.fo_loops, 1 / accumulation, recompute=True)
    shared = dict(model.shared_named_parameters())
    input_leaf = e.detach().requires_grad_(True)
    with training_precision(x.device):
        h = model.early_loops(input_leaf, cfg.num_recur - cfg.fo_loops)
    grads = torch.autograd.grad((h.float() * v.float()).sum(), [input_leaf, *shared.values()])
    with torch.no_grad():
        for (name, _), grad in zip(shared.items(), grads[1:]):
            exact_buffers[name].add_(grad)
    indirect = grads[0].detach()
    del grads, h, input_leaf
    finish_prelude_backward(model, x, None, direct, indirect, 1 - fraction)
    extra = dict(boundary_grad_norm=v.float().norm().detach())

    if zo_buffers is not None:
        fo = {name: p.grad for name, p in shared.items()}
        try:
            for name, p in shared.items():
                p.grad = zo_buffers[name]
            seeds = [derive_seed(cfg.zo_seed, step, micro, rank, i) for i in range(cfg.zo_directions)]
            derivatives = add_local_zo(model, e, v, cfg.num_recur - cfg.fo_loops,
                                       seeds, cfg.zo_epsilon, 1.0, cfg.zo_probe_dtype)
        finally:
            for name, p in shared.items():
                p.grad = fo[name]
        extra["zo_directional_abs"] = derivatives.abs().mean().detach()
    return loss, extra


def mix_transition_gradients(model, zo_buffers, exact_buffers, ratio, fraction):
    import torch
    import torch.distributed as dist
    from hybrid.norm_matching import mix_shared_gradients

    if fraction == 0:
        return {}  # Already full FO, including the prelude's indirect gradient.
    metrics = (mix_shared_gradients(model, zo_buffers, fraction * ratio)
               if zo_buffers is not None else {})
    if fraction < 1:
        with torch.no_grad():
            for name, p in model.shared_named_parameters():
                correction = exact_buffers[name]
                if zo_buffers is not None and dist.is_initialized():
                    # FO/ZO were already averaged. Add an averaged exact term.
                    dist.all_reduce(correction, op=dist.ReduceOp.AVG)
                p.grad.add_(correction, alpha=1 - fraction)
    return metrics


def transition_flops_per_token(model, cfg, fraction):
    """Same leading-matmul convention as steps.py; count exact replay in the ramp."""
    from hybrid.steps import estimated_flops_per_token
    full = estimated_flops_per_token(model, replace(cfg, mode="full_fo"))
    if fraction == 0:
        return full
    target = estimated_flops_per_token(model, cfg)
    if fraction == 1:
        return target
    truncated = estimated_flops_per_token(model, replace(cfg, mode="truncated"))
    prelude_forward = (2 * sum(p.numel() for p in model.transformer.prelude.parameters()) +
                       4 * cfg.n_prelude * model.config.n_embd * cfg.sequence_len)
    return target + 3 * (full - truncated) // 2 + prelude_forward
