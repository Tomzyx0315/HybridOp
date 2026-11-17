"""Gradient construction only; the caller performs exactly one optimizer step."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hybrid.model import GaussianDirection, derive_seed, probe_precision, training_precision


def _suffix_loss(model, e, targets, fo_loops, need_boundary):
    early = model.config.num_recur - fo_loops
    # Do not let the early no_grad visits populate autocast's weight cache:
    # the same parameters need differentiable casts in the later FO visits.
    with torch.no_grad(), training_precision(e.device, cache_enabled=False):
        state = model.early_loops(e.detach(), early)
    boundary = state.detach().requires_grad_(need_boundary)
    state = boundary
    for _ in range(fo_loops):
        state = model.recur(e, state)
    logits = model._predict(state, model.rotary(e.shape[1]))
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-1)
    return loss, boundary


def suffix_backward(model, tokens, targets, fo_loops, loss_scale=1.0, need_boundary=False):
    """Return a detached local objective after freeing all saved FO activations."""
    with training_precision(tokens.device):
        e = model.encode(tokens)
        # Keep the prelude graph for the direct injection into the FO suffix,
        # but never construct autograd graphs for early recurrences.
        loss, boundary = _suffix_loss(model, e, targets, fo_loops, need_boundary)
    (loss * loss_scale).backward()
    result = loss.detach()
    if need_boundary:
        # The cotangent ALREADY contains the gradient-accumulation normalization.
        return result, e.detach(), boundary.grad.detach()
    return result, None, None


def deferred_suffix_backward(model, tokens, targets, fo_loops, loss_scale=1.0, recompute=False):
    """Backprop the suffix only, returning a and v; prelude is differentiated later.

    e_graph is the sole live prefix graph in the retained variant. No retain_graph
    is needed: the leaf supplied to the suffix disconnects the two graphs.
    """
    with training_precision(tokens.device):
        if recompute:
            with torch.no_grad():
                initial_e = model.encode(tokens)
            e_graph = None
        else:
            initial_e = model.encode(tokens)
            e_graph = initial_e
        e = initial_e.detach().requires_grad_(True)
        loss, boundary = _suffix_loss(model, e, targets, fo_loops, True)
    (loss * loss_scale).backward()
    return loss.detach(), e.detach(), boundary.grad.detach(), e.grad.detach(), e_graph


def finish_prelude_backward(model, tokens, e_graph, direct, estimated_input, weight):
    # Both direct and estimated_input already include 1 / accumulation_steps.
    if e_graph is None:
        # Parameters have not changed. The current model has no dropout or other
        # stochastic forward ops, so this replays the same prelude function.
        with training_precision(tokens.device):
            e_graph = model.encode(tokens)
    upstream = direct.float() + weight * estimated_input
    e_graph.backward(upstream.to(e_graph.dtype))


@torch.no_grad()
def local_probe_difference(model, e, cotangent, early, parameter_direction=None,
                           parameter_epsilon=0.0, input_direction=None,
                           input_epsilon=0.0, probe_dtype="bf16"):
    """Return <v, A_plus - A_minus>, before dividing by either perturbation scale."""
    if e.requires_grad or cotangent.requires_grad:
        raise ValueError("The local objective must hold e and the boundary cotangent fixed")
    probe_e = e.float() if probe_dtype == "fp32" or input_direction is not None else e
    input_noise = (input_direction.noise("prelude_output", e.float())
                   if input_direction is not None else None)
    with probe_precision(e.device, probe_dtype):
        plus = model.early_loops(probe_e if input_noise is None else probe_e + input_epsilon * input_noise,
                                 early, parameter_direction, parameter_epsilon)
        minus = model.early_loops(probe_e if input_noise is None else probe_e - input_epsilon * input_noise,
                                  early, parameter_direction, -parameter_epsilon)
    # Shift e in ALL early injections and in the initial state. The later FO
    # suffix is not replayed, and its a/v remain fixed throughout both probes.
    difference = plus.float() - minus.float()
    difference.mul_(cotangent.float())
    result = difference.sum()
    if not torch.isfinite(result):
        raise FloatingPointError("Non-finite local ZO probe difference")
    return result


@torch.no_grad()
def local_directional_derivative(model, e, cotangent, early, direction, epsilon, probe_dtype="bf16"):
    difference = local_probe_difference(model, e, cotangent, early, parameter_direction=direction,
                                         parameter_epsilon=epsilon, probe_dtype=probe_dtype)
    coefficient = difference / (2 * epsilon)
    if not torch.isfinite(coefficient):
        raise FloatingPointError("Non-finite ZO directional derivative")
    return coefficient


@torch.no_grad()
def _accumulate_parameter_direction(model, direction, scale):
    for name, parameter in model.shared_named_parameters():
        noise = direction.noise(name, parameter)
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        parameter.grad.add_(noise, alpha=scale)


@torch.no_grad()
def add_local_zo(model, e, cotangent, early, seeds, epsilon=1e-3, weight=1.0, probe_dtype="bf16"):
    if not seeds:
        raise ValueError("At least one ZO direction is required")
    derivatives = []
    for seed in seeds:
        direction = GaussianDirection(seed)
        coefficient = local_directional_derivative(model, e, cotangent, early, direction, epsilon, probe_dtype)
        derivatives.append(coefficient)
        scale = weight * coefficient.item() / len(seeds)
        _accumulate_parameter_direction(model, direction, scale)
    return torch.stack(derivatives)


@torch.no_grad()
def estimate_input_zo(model, e, cotangent, early, seeds, epsilon=1e-3, probe_dtype="bf16"):
    """Input-only probes: do not alter the shared-parameter estimate or .grad."""
    if not seeds:
        raise ValueError("At least one ZO direction is required")
    estimate = torch.zeros_like(e, dtype=torch.float32)
    derivatives = []
    for seed in seeds:
        direction = GaussianDirection(seed)
        difference = local_probe_difference(model, e, cotangent, early, input_direction=direction,
                                             input_epsilon=epsilon, probe_dtype=probe_dtype)
        coefficient = difference / (2 * epsilon)
        derivatives.append(coefficient)
        estimate.add_(direction.noise("prelude_output", e.float()), alpha=coefficient.item() / len(seeds))
    return estimate, torch.stack(derivatives)


@torch.no_grad()
def add_joint_zo(model, e, cotangent, early, seeds, parameter_epsilon=1e-3,
                 input_epsilon=1e-3, weight=1.0, probe_dtype="bf16"):
    """Same two probes for theta and e, with their respective denominators.

    Return the UNWEIGHTED input estimate; weight is applied by prelude.backward.
    Input noise uses a distinct name in GaussianDirection, independent of each
    parameter tensor. No execution-mode label enters the RNG seed.
    """
    if not seeds:
        raise ValueError("At least one ZO direction is required")
    estimate = torch.zeros_like(e, dtype=torch.float32)
    parameter_derivatives, input_derivatives = [], []
    for seed in seeds:
        direction = GaussianDirection(seed)
        difference = local_probe_difference(model, e, cotangent, early, direction, parameter_epsilon,
                                             direction, input_epsilon, probe_dtype)
        parameter_coefficient = difference / (2 * parameter_epsilon)
        input_coefficient = difference / (2 * input_epsilon)
        parameter_derivatives.append(parameter_coefficient)
        input_derivatives.append(input_coefficient)
        _accumulate_parameter_direction(model, direction, weight * parameter_coefficient.item() / len(seeds))
        estimate.add_(direction.noise("prelude_output", e.float()), alpha=input_coefficient.item() / len(seeds))
    return estimate, torch.stack(parameter_derivatives), torch.stack(input_derivatives)


def backward_microbatch(model, tokens, targets, cfg, *, step, micro_step, rank, accumulation_steps):
    loss_scale = 1.0 / accumulation_steps
    if cfg.mode == "full_fo":
        with training_precision(tokens.device):
            loss = model(tokens, targets)
        (loss * loss_scale).backward()
        return loss.detach(), {}
    hybrid = cfg.mode == "hybrid" and cfg.zo_lambda != 0
    if not hybrid or cfg.prelude_mode == "none":
        loss, e, cotangent = suffix_backward(model, tokens, targets, cfg.fo_loops, loss_scale, hybrid)
    else:
        loss, e, cotangent, direct, e_graph = deferred_suffix_backward(
            model, tokens, targets, cfg.fo_loops, loss_scale, cfg.prelude_mode == "joint_recompute")
    if not hybrid:  # lambda=0 follows exactly the original truncated path in all variants.
        return loss, {}
    seeds = [derive_seed(cfg.zo_seed, step, micro_step, rank, index)
             for index in range(cfg.zo_directions)]
    early = model.config.num_recur - cfg.fo_loops
    # The suffix graph is freed. Only retained-prelude variants keep a prefix graph.
    if cfg.prelude_mode in {"joint", "joint_recompute"}:
        input_estimate, derivatives, input_derivatives = add_joint_zo(
            model, e, cotangent, early, seeds, cfg.zo_epsilon, cfg.zo_input_epsilon,
            cfg.zo_lambda, cfg.zo_probe_dtype)
    else:
        derivatives = add_local_zo(model, e, cotangent, early, seeds,
                                   cfg.zo_epsilon, cfg.zo_lambda, cfg.zo_probe_dtype)
        if cfg.prelude_mode == "separate":
            input_estimate, input_derivatives = estimate_input_zo(
                model, e, cotangent, early, seeds, cfg.zo_input_epsilon, cfg.zo_probe_dtype)
    metrics = {"zo_directional_abs": derivatives.abs().mean().detach(),
               "boundary_grad_norm": cotangent.float().norm().detach()}
    if cfg.prelude_mode != "none":
        finish_prelude_backward(model, tokens, e_graph, direct, input_estimate, cfg.zo_lambda)
        metrics.update(zo_input_directional_abs=input_derivatives.abs().mean().detach(),
                       zo_input_grad_norm=input_estimate.norm().detach(),
                       prelude_direct_grad_norm=direct.float().norm().detach())
    return loss, metrics


def estimated_flops_per_token(model, cfg):
    """Leading matmul/attention FLOPs; excludes optimizer, probes' RNG/copies."""
    # Upstream estimate_flops uses bptt_k for its backward cost estimate only.
    previous = model.config.bptt_k
    try:
        model.config.bptt_k = model.config.num_recur if cfg.mode == "full_fo" else cfg.fo_loops
        total = model.estimate_flops()
    finally:
        model.config.bptt_k = previous
    if cfg.mode == "hybrid" and cfg.zo_lambda != 0:
        early = model.config.num_recur - cfg.fo_loops
        matrices = sum(p.numel() for p in model.transformer.recur.parameters())
        matrices += sum(p.numel() for p in model.inject.parameters())
        attention = model.config.n_embd * model.config.sequence_len
        forward = 2 * matrices + 4 * cfg.n_recur_block * attention
        total += 2 * cfg.zo_directions * early * forward
        if cfg.prelude_mode == "separate":
            total += 2 * cfg.zo_directions * early * forward
        elif cfg.prelude_mode == "joint_recompute":
            # One extra prelude forward, not an extra backward. Embedding lookup
            # and pointwise norms remain excluded by the upstream FLOPs convention.
            prelude_params = sum(p.numel() for p in model.transformer.prelude.parameters())
            total += 2 * prelude_params + 4 * cfg.n_prelude * attention
    return total
