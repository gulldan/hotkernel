"""Pinned Hub adapters. Torch and kernels are imported only in the target interpreter."""

from __future__ import annotations

import os
import runpy
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


def _check_packed_gate(x) -> None:
    import torch

    if not (
        x.is_cuda and x.is_contiguous() and x.ndim and x.shape[-1] > 0
        and x.shape[-1] % 2 == 0
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        raise ValueError("silu_and_mul expects a contiguous CUDA float tensor with an even width")


def _activation_silu(kernel, _baseline):
    import torch

    def run(x):
        _check_packed_gate(x)
        out = torch.empty((*x.shape[:-1], x.shape[-1] // 2), device=x.device, dtype=x.dtype)
        return kernel.silu_and_mul(out, x)

    return run


def _liger_silu(kernel, _baseline):
    apply = kernel.layers.LigerSiLUMulFunction.apply

    def run(x):
        _check_packed_gate(x)
        return apply(*x.chunk(2, dim=-1))

    return run


def _liger_rmsnorm(kernel, baseline):
    if "EPS" not in baseline:
        raise ValueError("The RMSNorm baseline must define EPS so both variants use the same epsilon")
    eps = float(baseline["EPS"])
    apply = kernel.layers.LigerRMSNormFunction.apply

    def run(x, weight):
        return apply(x, weight, eps, 0.0, "gemma", False, None)

    return run


@dataclass(frozen=True)
class HubKernel:
    repo_id: str
    revision: str
    entry: str
    adapter: Callable

    def provenance(self) -> dict:
        return {"repo_id": self.repo_id, "revision": self.revision, "entry": self.entry}


_ACTIVATION = "281eca0bef25176754f9e55cee447a7e99264e01"
_LIGER = "d38583f406e1ac2b7dfc0215da0135dca2243bea"

HUB_KERNELS = {
    "silu_and_mul": {
        "activation": HubKernel(
            "kernels-community/activation", _ACTIVATION, "silu_and_mul", _activation_silu
        ),
        "liger": HubKernel(
            "kernels-community/liger-kernels", _LIGER, "LigerSiLUMulFunction", _liger_silu
        ),
    },
    "rmsnorm": {
        "liger": HubKernel(
            "kernels-community/liger-kernels", _LIGER, "LigerRMSNormFunction", _liger_rmsnorm
        )
    },
}


def prepare_candidate(baseline_path: str, operation: str, name: str):
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from kernels import get_kernel

    sys.path.insert(0, str(Path(baseline_path).parent))
    baseline = runpy.run_path(baseline_path)
    if not callable(baseline.get("run")) or not callable(
        getattr(baseline.get("TARGET"), "make_inputs", None)
    ):
        raise TypeError("The baseline must expose TARGET and run(fn, impl, extra)")

    spec = HUB_KERNELS[operation][name]
    kernel = get_kernel(spec.repo_id, revision=spec.revision)
    return baseline["TARGET"], spec.adapter(kernel, baseline), baseline["run"]


def candidate_script(baseline: Path, operation: str, name: str) -> str:
    """A normal HotKernel target which can import its adapter from the other environment."""
    spec = HUB_KERNELS[operation][name]
    return (
        "import sys\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})\n"
        "from hotkernel.hub import prepare_candidate\n"
        f"TARGET, profile_fn, run = prepare_candidate({str(baseline)!r}, {operation!r}, {name!r})\n"
        "if __name__ == '__main__':\n"
        f"    run(profile_fn, impl={spec.repo_id + ':' + spec.entry!r}, "
        f"extra={{'hub_kernel': {spec.provenance()!r}}})\n"
    )


def compare_traces(baseline: dict, candidate: dict) -> dict:
    if "error" in baseline or "error" in candidate or not (
        baseline.get("kernels") and candidate.get("kernels")
    ):
        return {"next_check": "Collect a CUDA trace for both variants before attributing the result."}

    copies = [trace["tensor_copy_calls"] for trace in (baseline, candidate)]
    hotspot = candidate["kernels"][0]
    if copies[1] > copies[0]:
        next_check = "Inspect input strides and aten::copy_ calls for extra layout conversions."
    else:
        next_check = (
            f"Profile {hotspot['short_name']} with Nsight Compute, if available, "
            "to check memory traffic and compute utilization."
        )
    return {
        "cuda_events": {
            "baseline": baseline["cuda_events"], "candidate": candidate["cuda_events"]
        },
        "tensor_copy_calls": {"baseline": copies[0], "candidate": copies[1]},
        "candidate_hotspot": hotspot,
        "next_check": next_check,
    }
