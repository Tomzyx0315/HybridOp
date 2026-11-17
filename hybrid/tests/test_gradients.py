"""Numerical acceptance tests. Run on the experiment machine with PyTorch installed."""

import copy
from dataclasses import replace

import pytest
import torch

from hybrid.config import Config, PRELUDE_MODES
from hybrid.model import ExperimentGPT, GaussianDirection, derive_seed
from hybrid.steps import (add_local_zo, backward_microbatch, estimated_flops_per_token,
                          local_directional_derivative, suffix_backward)
from hybrid.train import model_config
from nanochat.gpt import GPT


@pytest.fixture(autouse=True)
def cpu_test_settings():
    old = torch._dynamo.config.disable
    threads = torch.get_num_threads()
    torch._dynamo.config.disable = True
    torch.set_num_threads(1)
    yield
    torch._dynamo.config.disable = old
    torch.set_num_threads(threads)


def tiny(mode="hybrid", loops=2):
    cfg = Config(mode=mode, device="cpu", size=1, aspect_ratio=16, head_dim=8,
                 sequence_len=5, vocab_size=32, n_prelude=1, n_recur_block=1, n_coda=1,
                 num_recur=loops, device_batch_size=2, total_batch_size=20, compile=False)
    torch.manual_seed(19)
    model = ExperimentGPT(model_config(cfg))
    model.init_weights()
    with torch.no_grad():
        # Leave the degenerate [I,0] initial point to test genuine early credit.
        model.inject.weight[:, cfg.width:].copy_(0.3 * torch.eye(cfg.width))
        model.lm_head.weight.mul_(10)
    tokens = torch.randint(cfg.vocab_size, (2, cfg.sequence_len))
    targets = torch.randint(cfg.vocab_size, tokens.shape)
    return cfg, model, tokens, targets


def gradients(model):
    return {name: p.grad.clone() for name, p in model.named_parameters()}


def backward(model, x, y, cfg, micro=0, accum=1):
    return backward_microbatch(model, x, y, cfg, step=7, micro_step=micro, rank=0,
                               accumulation_steps=accum)


@pytest.mark.parametrize("mode,k", [("full_fo", 2), ("truncated", 1)])
def test_matches_upstream_forward_and_gradients(mode, k):
    cfg, model, x, y = tiny(mode)
    upstream = GPT(replace(model.config, bptt_k=k))
    upstream.init_weights()
    upstream.load_state_dict(model.state_dict())
    reference_loss = upstream(x, y)
    reference_loss.backward()
    loss, _ = backward(model, x, y, cfg)
    torch.testing.assert_close(loss, reference_loss.detach(), rtol=0, atol=0)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(upstream.named_parameters())[name].grad)


@pytest.mark.parametrize("prelude_mode", PRELUDE_MODES)
def test_zero_lambda_is_exactly_truncated(prelude_mode):
    cfg, model, x, y = tiny()
    trunc = copy.deepcopy(model)
    backward(model, x, y, replace(cfg, zo_lambda=0, prelude_mode=prelude_mode))
    backward(trunc, x, y, replace(cfg, mode="truncated"))
    for name, p in model.named_parameters():
        torch.testing.assert_close(p.grad, dict(trunc.named_parameters())[name].grad, rtol=0, atol=0)


@pytest.mark.parametrize("loops", [2, 4])
def test_local_vjp_equals_missing_shared_gradient(loops):
    cfg, model, x, y = tiny(loops=loops)
    full = copy.deepcopy(model)
    backward(full, x, y, replace(cfg, mode="full_fo"))
    _, e, v = suffix_backward(model, x, y, cfg.fo_loops, need_boundary=True)
    truncated = gradients(model)
    model.zero_grad(set_to_none=True)
    early = model.config.num_recur - cfg.fo_loops
    h = model.early_loops(e, early)
    (h * v).sum().backward()
    for name, p in model.shared_named_parameters():
        missing = dict(full.named_parameters())[name].grad - truncated[name]
        torch.testing.assert_close(p.grad, missing, rtol=5e-4, atol=2e-7)


@pytest.mark.parametrize("loops", [2, 3])
def test_finite_difference_matches_exact_directional_derivative(loops):
    cfg, model, x, y = tiny(loops=loops)
    _, e, v = suffix_backward(model, x, y, 1, need_boundary=True)
    model.zero_grad(set_to_none=True)
    early = loops - 1
    direction = GaussianDirection(101)
    (model.early_loops(e, early) * v).sum().backward()
    exact = sum((p.grad * direction.noise(name, p)).sum() for name, p in model.shared_named_parameters())
    estimate = local_directional_derivative(model, e, v, early, direction, 1e-4, "fp32")
    torch.testing.assert_close(estimate, exact, rtol=0.015, atol=2e-6)


def test_hybrid_combines_gradients_before_optimizer():
    cfg, model, x, y = tiny()
    reference = copy.deepcopy(model)
    _, e, v = suffix_backward(reference, x, y, 1, need_boundary=True)
    seeds = [derive_seed(cfg.zo_seed, 7, 0, 0, i) for i in range(2)]
    coeffs = [local_directional_derivative(reference, e, v, 1, GaussianDirection(s), cfg.zo_epsilon)
              for s in seeds]
    with torch.no_grad():
        for name, p in reference.shared_named_parameters():
            for seed, coefficient in zip(seeds, coeffs):
                p.grad.add_(GaussianDirection(seed).noise(name, p), alpha=0.5 * coefficient.item() / 2)
    backward(model, x, y, replace(cfg, zo_lambda=0.5, zo_directions=2))
    for name, p in model.named_parameters():
        torch.testing.assert_close(p.grad, dict(reference.named_parameters())[name].grad)


@pytest.mark.parametrize("prelude_mode", PRELUDE_MODES)
def test_accumulation_is_normalized_once(prelude_mode):
    cfg, model, x, y = tiny()
    cfg = replace(cfg, prelude_mode=prelude_mode)
    expected = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    for micro in range(2):
        single = copy.deepcopy(model)
        backward(single, x, y, cfg, micro=micro)
        for name, p in single.named_parameters():
            expected[name].add_(p.grad / 2)
    for micro in range(2):
        backward(model, x, y, cfg, micro=micro, accum=2)
    for name, p in model.named_parameters():
        torch.testing.assert_close(p.grad, expected[name], rtol=1e-5, atol=1e-7)


def test_probes_do_not_modify_parameters_or_rng():
    _, model, x, y = tiny()
    _, e, v = suffix_backward(model, x, y, 1, need_boundary=True)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    identities = {name: id(p) for name, p in model.named_parameters()}
    rng = torch.get_rng_state().clone()
    add_local_zo(model, e, v, 1, [81, 82])
    for name, p in model.named_parameters():
        assert id(p) == identities[name]
        assert torch.equal(p, before[name])
    assert torch.equal(torch.get_rng_state(), rng)


def test_probe_exception_restores_parameter_bindings(monkeypatch):
    _, model, x, _ = tiny()
    with torch.no_grad():
        e = model.encode(x).detach()
    originals = dict(model.named_parameters())
    before = {name: p.detach().clone() for name, p in originals.items()}
    def fail(*args, **kwargs):
        raise RuntimeError("probe failure")
    monkeypatch.setattr(model.transformer.recur[0], "forward", fail)
    with pytest.raises(RuntimeError, match="probe failure"), torch.no_grad():
        model.early_loops(e, 1, GaussianDirection(6), 1e-3)
    for name, p in model.named_parameters():
        assert p is originals[name]
        assert torch.equal(p, before[name])


def test_probe_order_and_no_grad():
    cfg, model, x, y = tiny()
    events = []
    def observe(module, args):
        events.append((torch.is_grad_enabled(), model.lm_head.weight.grad is not None))
    handle = model.transformer.recur[0].register_forward_pre_hook(observe)
    backward(model, x, y, cfg)
    handle.remove()
    assert events == [(False, False), (True, False), (False, True), (False, True)]


def test_initial_zero_boundary_is_well_defined():
    cfg, model, x, y = tiny()
    with torch.no_grad():
        model.inject.weight[:, cfg.width:].zero_()
    _, extra = backward(model, x, y, cfg)
    assert extra["boundary_grad_norm"].item() == 0
    assert extra["zo_directional_abs"].item() == 0


def test_one_direction_flop_estimate_matches_full_fo():
    cfg, model, _, _ = tiny()
    full = estimated_flops_per_token(model, replace(cfg, mode="full_fo"))
    hybrid = estimated_flops_per_token(model, cfg)
    truncated = estimated_flops_per_token(model, replace(cfg, mode="truncated"))
    assert full == hybrid
    assert truncated < full


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA acceptance check")
@pytest.mark.parametrize("prelude_mode", PRELUDE_MODES)
def test_cuda_bf16_compiled_hybrid(prelude_mode):
    cfg, model, x, y = tiny()
    cfg = replace(cfg, prelude_mode=prelude_mode)
    torch._dynamo.config.disable = False
    model = model.cuda()
    model.transformer.wte.to(torch.bfloat16)
    eager = copy.deepcopy(model)
    model.compile_blocks()
    for micro in range(2):
        backward(model, x.cuda(), y.cuda(), cfg, micro=micro)
        backward(eager, x.cuda(), y.cuda(), cfg, micro=micro)
    for name, p in model.named_parameters():
        assert torch.isfinite(p.grad).all()
        # Compilation may fuse BF16 pointwise operations differently.
        torch.testing.assert_close(p.grad, dict(eager.named_parameters())[name].grad, rtol=0.15, atol=1e-3)
