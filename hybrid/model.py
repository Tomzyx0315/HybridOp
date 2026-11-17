"""Expose loop boundaries without changing upstream parameters or forward math."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager, nullcontext

import torch
from torch.func import functional_call

from nanochat.gpt import GPT


def derive_seed(*parts):
    data = ":".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little") % (2**63 - 1)


class GaussianDirection:
    """Replay one global N(0,I) direction tensor by tensor, without changing RNG state."""

    def __init__(self, seed):
        self.seed = seed

    def noise(self, name, parameter):
        generator = torch.Generator(device=parameter.device)
        generator.manual_seed(derive_seed(self.seed, name))
        return torch.randn(parameter.shape, dtype=parameter.dtype, device=parameter.device,
                           generator=generator)

    def call(self, name, module, args, scale):
        # Only one block's perturbed weights exist at a time. Original parameter
        # storage is never modified, so there is no +/- roundoff drift between ranks.
        perturbed = {}
        for local_name, parameter in module.named_parameters():
            noise = self.noise(f"{name}.{local_name}", parameter)
            perturbed[local_name] = noise.mul_(scale).add_(parameter.detach())
        return functional_call(module, perturbed, args, strict=True)


class ExperimentGPT(GPT):
    def encode(self, tokens):
        cos_sin = self.rotary(tokens.shape[1])
        e = self.norm_emb(self.transformer.wte(tokens))
        for block in self.transformer.prelude:
            e = block(e, cos_sin, None)
        return e

    def rotary(self, length):
        if length > self.cos.shape[1]:
            raise ValueError("Sequence exceeds the rotary cache")
        return self.cos[:, :length], self.sin[:, :length]

    def recur(self, e, state=None, direction=None, scale=0.0):
        if self.config.input_injection != "inject_init_prelude":
            raise ValueError("This experiment uses the upstream learned concat injection")
        cos_sin = self.rotary(e.shape[1])
        state = e if state is None else state
        joined = torch.cat((e, state), dim=-1)
        if direction is None:
            u = self.inject(joined)
            for block in self.transformer.recur:
                u = block(u, cos_sin, None)
            return self.norm_recur(u)
        u = direction.call("inject", self.inject, (joined,), scale)
        for index, block in enumerate(self.transformer.recur):
            u = direction.call(f"transformer.recur.{index}", block, (u, cos_sin, None), scale)
        return direction.call("norm_recur", self.norm_recur, (u,), scale)

    def early_loops(self, e, count, direction=None, scale=0.0):
        state = None
        for _ in range(count):
            # Every occurrence uses the SAME perturbation of the shared weights.
            state = self.recur(e, state, direction, scale)
        return state

    def shared_named_parameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith(("transformer.recur.", "inject.", "norm_recur.")):
                yield name, parameter


def training_precision(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


@contextmanager
def probe_precision(device, dtype):
    if device.type != "cuda" or dtype == "bf16":
        with training_precision(device):
            yield
        return
    # FA2/FA3 require half precision. FP32 is an optional numerical diagnostic;
    # use the existing SDPA fallback and full FP32 matmul for that scope only.
    import nanochat.flash_attention as attention
    previous_backend = attention._override_impl
    previous_precision = torch.backends.cuda.matmul.fp32_precision
    try:
        attention._override_impl = "sdpa"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        with torch.autocast("cuda", enabled=False):
            yield
    finally:
        attention._override_impl = previous_backend
        torch.backends.cuda.matmul.fp32_precision = previous_precision
