"""Input VJP, joint-probe scaling, and retained/recomputed prelude equivalence."""

import copy
from dataclasses import replace

import pytest
import torch

from hybrid.config import Config, PRELUDE_MODES, recipe
from hybrid.model import ExperimentGPT, GaussianDirection
from hybrid.steps import (add_joint_zo, backward_microbatch, deferred_suffix_backward,
                          estimated_flops_per_token, estimate_input_zo, finish_prelude_backward,
                          local_probe_difference)
from hybrid.train import build_optimizer, model_config
from hybrid.tests.test_gradients import backward, cpu_test_settings, tiny


@pytest.mark.parametrize("loops,fo_loops", [(2, 1), (4, 2)])
@pytest.mark.parametrize("recompute", [False, True])
def test_exact_input_and_parameter_vjps_recover_full_model_gradient(loops, fo_loops, recompute):
    cfg, model, x, y = tiny(loops=loops)
    full = copy.deepcopy(model)
    cfg = replace(cfg, fo_loops=fo_loops)
    backward(full, x, y, replace(cfg, mode="full_fo"))
    _, e, v, a, e_graph = deferred_suffix_backward(model, x, y, fo_loops, recompute=recompute)
    input_leaf = e.detach().requires_grad_(True)
    h = model.early_loops(input_leaf, loops - fo_loops)
    shared = [p for _, p in model.shared_named_parameters()]
    local_grads = torch.autograd.grad((h * v).sum(), [input_leaf] + shared)
    with torch.no_grad():
        for parameter, local_grad in zip(shared, local_grads[1:]):
            parameter.grad.add_(local_grad)
    finish_prelude_backward(model, x, e_graph, a, local_grads[0], 1.0)
    for name, p in model.named_parameters():
        torch.testing.assert_close(p.grad, dict(full.named_parameters())[name].grad, rtol=5e-4, atol=2e-7)


@pytest.mark.parametrize("loops", [2, 3])
def test_input_and_joint_finite_differences_match_local_vjp(loops):
    _, model, x, y = tiny(loops=loops)
    _, e, v, _, _ = deferred_suffix_backward(model, x, y, 1, recompute=True)
    input_leaf = e.detach().requires_grad_(True)
    shared = list(model.shared_named_parameters())
    h = model.early_loops(input_leaf, loops - 1)
    grads = torch.autograd.grad((h * v).sum(), [input_leaf] + [p for _, p in shared])
    direction = GaussianDirection(508)
    exact_input = (grads[0] * direction.noise("prelude_output", e.float())).sum()
    exact_parameter = sum((g * direction.noise(name, p)).sum() for (name, p), g in zip(shared, grads[1:]))
    parameter_epsilon, input_epsilon = 1e-4, 3e-4
    input_difference = local_probe_difference(model, e, v, loops - 1, input_direction=direction,
                                               input_epsilon=input_epsilon, probe_dtype="fp32")
    joint_difference = local_probe_difference(model, e, v, loops - 1, direction, parameter_epsilon,
                                               direction, input_epsilon, "fp32")
    torch.testing.assert_close(input_difference / (2 * input_epsilon), exact_input, rtol=0.02, atol=2e-6)
    expected_parameter_coefficient = exact_parameter + input_epsilon / parameter_epsilon * exact_input
    torch.testing.assert_close(joint_difference / (2 * parameter_epsilon), expected_parameter_coefficient,
                               rtol=0.02, atol=2e-6)


def test_joint_reconstruction_uses_separate_scales_and_applies_lambda_once():
    _, model, x, y = tiny()
    _, e, v, _, _ = deferred_suffix_backward(model, x, y, 1, recompute=True)
    before = {name: p.grad.clone() for name, p in model.shared_named_parameters()}
    parameter_epsilon, input_epsilon, weight = 1e-3, 4e-3, 0.25
    seeds = [55, 56]
    differences = [local_probe_difference(model, e, v, 1, GaussianDirection(seed), parameter_epsilon,
                                           GaussianDirection(seed), input_epsilon) for seed in seeds]
    estimate, parameter_cs, input_cs = add_joint_zo(model, e, v, 1, seeds, parameter_epsilon,
                                                    input_epsilon, weight)
    expected_input = torch.zeros_like(e, dtype=torch.float32)
    for seed, difference, pc, ic in zip(seeds, differences, parameter_cs, input_cs):
        torch.testing.assert_close(pc, difference / (2 * parameter_epsilon))
        torch.testing.assert_close(ic, difference / (2 * input_epsilon))
        expected_input.add_(GaussianDirection(seed).noise("prelude_output", e.float()),
                            alpha=ic.item() / len(seeds))
    torch.testing.assert_close(estimate, expected_input)
    for name, p in model.shared_named_parameters():
        expected = before[name]
        for seed, pc in zip(seeds, parameter_cs):
            expected = expected + weight / len(seeds) * pc * GaussianDirection(seed).noise(name, p)
        torch.testing.assert_close(p.grad, expected, rtol=1e-5, atol=1e-7)


def test_separate_probes_only_change_prelude_gradients():
    cfg, model, x, y = tiny()
    base = copy.deepcopy(model)
    backward(base, x, y, cfg)
    backward(model, x, y, replace(cfg, prelude_mode="separate"))
    changed = False
    for name, p in model.named_parameters():
        reference = dict(base.named_parameters())[name].grad
        if name.startswith(("transformer.prelude.", "transformer.wte.", "norm_emb.")):
            changed |= not torch.allclose(p.grad, reference)
        else:
            torch.testing.assert_close(p.grad, reference, rtol=1e-5, atol=1e-7)
    assert changed, "Nonzero early input credit must reach prelude/embedding"


@pytest.mark.parametrize("directions", [1, 2])
def test_retained_and_recomputed_joint_match_gradients_and_optimizer_updates(directions):
    cfg, retained, x, y = tiny()
    recomputed = copy.deepcopy(retained)
    cfg = replace(cfg, prelude_mode="joint", zo_directions=directions,
                  zo_epsilon=1e-3, zo_rho=2, zo_lambda=0.7)
    recomputed_cfg = replace(cfg, prelude_mode="joint_recompute")
    opt_a = build_optimizer(retained, cfg, recipe(cfg, 1))
    opt_b = build_optimizer(recomputed, recomputed_cfg, recipe(recomputed_cfg, 1))
    for step in range(2):
        for micro in range(2):
            loss_a, _ = backward_microbatch(retained, x, y, cfg, step=step, micro_step=micro,
                                             rank=0, accumulation_steps=2)
            loss_b, _ = backward_microbatch(recomputed, x, y, recomputed_cfg, step=step, micro_step=micro,
                                             rank=0, accumulation_steps=2)
            torch.testing.assert_close(loss_a, loss_b, rtol=1e-6, atol=1e-8)
        for name, p in retained.named_parameters():
            torch.testing.assert_close(p.grad, dict(recomputed.named_parameters())[name].grad,
                                       rtol=1e-5, atol=1e-7)
        opt_a.step()
        opt_b.step()
        for name, p in retained.named_parameters():
            torch.testing.assert_close(p, dict(recomputed.named_parameters())[name], rtol=1e-5, atol=1e-7)
        retained.zero_grad(set_to_none=True)
        recomputed.zero_grad(set_to_none=True)


@pytest.mark.parametrize("mode", PRELUDE_MODES)
def test_prelude_executes_one_backward_and_probes_follow_suffix(mode):
    cfg, model, x, y = tiny()
    events = []
    handles = [
        model.transformer.prelude[0].register_forward_pre_hook(
            lambda module, inputs: events.append(("prelude", torch.is_grad_enabled()))),
        model.transformer.prelude[0].register_full_backward_hook(
            lambda module, inputs, outputs: events.append(("prelude_backward",))),
        model.transformer.recur[0].register_forward_pre_hook(
            lambda module, inputs: events.append(("recur", torch.is_grad_enabled()))),
    ]
    backward(model, x, y, replace(cfg, prelude_mode=mode))
    for handle in handles:
        handle.remove()
    assert events.count(("prelude_backward",)) == 1
    assert events.count(("recur", False)) == (5 if mode == "separate" else 3)
    assert events.count(("recur", True)) == 1
    if mode == "joint_recompute":
        assert events[0] == ("prelude", False)
        assert events[-2:] == [("prelude", True), ("prelude_backward",)]
    elif mode != "none":
        assert events[0] == ("prelude", True)
        assert events[-1] == ("prelude_backward",)


@pytest.mark.parametrize("mode", PRELUDE_MODES)
def test_no_parameter_or_rng_mutation(mode):
    cfg, model, x, y = tiny()
    values = {name: p.detach().clone() for name, p in model.named_parameters()}
    identities = {name: id(p) for name, p in model.named_parameters()}
    rng = torch.get_rng_state().clone()
    backward(model, x, y, replace(cfg, prelude_mode=mode))
    for name, p in model.named_parameters():
        assert id(p) == identities[name]
        assert torch.equal(p, values[name])
    assert torch.equal(rng, torch.get_rng_state())


def test_input_probe_preserves_input_values_and_parameter_gradients():
    _, model, x, y = tiny()
    _, e, v, _, _ = deferred_suffix_backward(model, x, y, 1, recompute=True)
    original = e.clone()
    gradients = {name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()}
    estimate_input_zo(model, e, v, 1, [80, 81])
    assert torch.equal(e, original)
    for name, p in model.named_parameters():
        if gradients[name] is None:
            assert p.grad is None
        else:
            assert torch.equal(p.grad, gradients[name])


def test_agreed_flops_for_all_variants():
    cfg = Config(mode="hybrid")
    with torch.device("meta"):
        model = ExperimentGPT(model_config(cfg))
    expected = dict(none=2235678720, separate=2781003776, joint=2235678720, joint_recompute=2302795776)
    for mode, flops in expected.items():
        assert estimated_flops_per_token(model, replace(cfg, prelude_mode=mode)) == flops
        assert estimated_flops_per_token(model, replace(cfg, prelude_mode=mode, zo_lambda=0)) == 1690353664
