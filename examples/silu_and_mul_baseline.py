"""Packed BF16 gate and values for the activation used in a SwiGLU layer."""

import torch
from _hotkernel_contract import Target
from torch.nn.functional import silu


def silu_and_mul_baseline(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    return silu(gate) * up


def reference(x: torch.Tensor) -> torch.Tensor:
    return silu_and_mul_baseline(x.float()).to(x.dtype)


TARGET = Target(
    name="silu_and_mul",
    dtype="bf16",
    shapes={"x": (4096, 8192)},
    rotate=("x",),
    reference=reference,
    reference_name="SiLU and multiplication in fp32",
)


def run(fn, impl: str, extra: dict | None = None) -> None:
    TARGET.run(fn, impl, extra)


if __name__ == "__main__":
    run(silu_and_mul_baseline, impl="eager PyTorch SiLU and multiply")
