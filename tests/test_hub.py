from pathlib import Path

import pytest

from hotkernel import server


@pytest.mark.parametrize(
    ("verdicts", "recommended", "traced"),
    [
        (("faster", "incorrect"), "activation", ["silu_and_mul_baseline", "activation"]),
        (("error", "slower"), None, ["silu_and_mul_baseline", "liger"]),
        (("error", "error"), None, ["silu_and_mul_baseline"]),
    ],
)
def test_compare_hub_candidates(tmp_path: Path, monkeypatch, verdicts, recommended, traced):
    baseline = tmp_path / "silu_and_mul_baseline.py"
    baseline.touch()
    checked = []
    profiles = []
    monkeypatch.setattr(server, "_python_or_error", lambda: ("python", None))
    monkeypatch.setattr(server, "_run", lambda *args, **kwargs: {"returncode": 0})

    def verify(candidate, reference, *, n, warmup, **kwargs):
        script = Path(candidate)
        checked.append(script)
        assert Path(reference) == baseline.resolve()
        assert (n, warmup) == (5, 1)
        assert str(baseline) in script.read_text()
        result = verdicts[len(checked) - 1]
        if result == "error":
            return {"error": "no compatible build"}
        return {"verdict": result, "candidate": {"mean_ms": 0.1 if result == "faster" else 0.01}}

    def trace(script, *, warm, **kwargs):
        assert warm
        name = Path(script).stem
        profiles.append(name)
        return {
            "kernels": [{"short_name": name}],
            "cuda_events": 8,
            "tensor_copy_calls": 16 if name == "liger" else 0,
        }

    monkeypatch.setattr(server, "verify", verify)
    monkeypatch.setattr(server, "torch_trace", trace)
    result = server.compare_hub(baseline.name, cwd=str(tmp_path))

    assert result["operation"] == "silu_and_mul"
    assert result["recommended"] == recommended
    assert result["status"] == ("unavailable" if verdicts == ("error", "error") else "compared")
    assert profiles == traced
    assert not any(p.exists() for p in checked)
    assert [a["name"] for a in result["candidates"]] == ["activation", "liger"]
    assert all(len(a["revision"]) == 40 for a in result["candidates"])
    if "liger" in profiles:
        assert "input strides" in result["candidates"][1]["diagnosis"]["next_check"]


def test_compare_hub_unknown_operation_reports_catalog():
    result = server.compare_hub("not_in_catalog_baseline.py")
    assert "error" in result
    assert result["supported_operations"] == ["rmsnorm", "silu_and_mul"]


def test_compare_hub_missing_optional_dependency(tmp_path: Path, monkeypatch):
    baseline = tmp_path / "rmsnorm_baseline.py"
    baseline.touch()
    monkeypatch.setattr(server, "_python_or_error", lambda: ("python", None))
    monkeypatch.setattr(server, "_run", lambda *args, **kwargs: {"returncode": 1})
    assert "HOTKERNEL_PYTHON" in server.compare_hub(str(baseline))["error"]
