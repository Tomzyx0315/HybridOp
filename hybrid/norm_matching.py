"""Accumulate and globally norm-match first-loop ZO to suffix FO gradients."""

import math


def separated_backward(model, x, y, cfg, zo_buffers, *, step, micro, rank, accumulation,
                       return_metrics=False):
    """Accumulate suffix FO in .grad and the early parameter estimate separately."""
    from hybrid.model import derive_seed
    from hybrid.steps import add_local_zo, suffix_backward
    loss, e, v = suffix_backward(model, x, y, cfg.fo_loops, 1 / accumulation, True)
    shared = dict(model.shared_named_parameters())
    fo = {name: p.grad for name, p in shared.items()}
    try:
        for name, p in shared.items():
            p.grad = zo_buffers[name]
        seeds = [derive_seed(cfg.zo_seed, step, micro, rank, i) for i in range(cfg.zo_directions)]
        derivatives = add_local_zo(model, e, v, cfg.num_recur - cfg.fo_loops, seeds,
                                   cfg.zo_epsilon, 1.0, cfg.zo_probe_dtype)
    finally:
        for name, p in shared.items():
            p.grad = fo[name]
    if return_metrics:
        return loss, dict(zo_directional_abs=derivatives.abs().mean().detach(),
                          boundary_grad_norm=v.float().norm().detach())
    return loss


def mix_shared_gradients(model, zo_buffers, ratio=None):
    """ratio=None applies raw ZO; a number sets ||scaled ZO||/||suffix FO||."""
    import torch
    import torch.distributed as dist
    shared = dict(model.shared_named_parameters())
    with torch.no_grad():
        if dist.is_initialized():
            for name, p in shared.items():
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                dist.all_reduce(zo_buffers[name], op=dist.ReduceOp.AVG)
        fo2 = sum(p.grad.float().square().sum() for p in shared.values())
        zo2 = sum(z.float().square().sum() for z in zo_buffers.values())
        dot = sum((p.grad.float() * zo_buffers[n].float()).sum() for n, p in shared.items())
        fo_norm, zo_norm = fo2.sqrt().item(), zo2.sqrt().item()
        if not math.isfinite(fo_norm) or not math.isfinite(zo_norm):
            raise FloatingPointError("Non-finite norm before mixing")
        scale = 1.0 if ratio is None else (ratio * fo_norm / zo_norm if zo_norm else 0.0)
        for name, p in shared.items():
            p.grad.add_(zo_buffers[name], alpha=scale)
        return dict(fo_shared_norm=fo_norm, raw_zo_norm=zo_norm, effective_lambda=scale,
                    scaled_zo_norm=scale * zo_norm,
                    fo_zo_cosine=dot.item() / max(fo_norm * zo_norm, 1e-30))
