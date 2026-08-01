import os
import pytest
from types import SimpleNamespace

from csubst import structural_alphabet


def _fake_torch_module(cuda_available=False, mps_available=False, mps_built=True):
    cuda_ns = SimpleNamespace(is_available=lambda: bool(cuda_available))
    mps_ns = SimpleNamespace(
        is_available=lambda: bool(mps_available),
        is_built=lambda: bool(mps_built),
    )
    return SimpleNamespace(cuda=cuda_ns, backends=SimpleNamespace(mps=mps_ns))


def test_resolve_prostt5_device_auto_prefers_cuda_then_mps():
    dev_cuda = structural_alphabet._resolve_prostt5_device(
        torch_module=_fake_torch_module(cuda_available=True, mps_available=True),
        device_opt="auto",
    )
    dev_mps = structural_alphabet._resolve_prostt5_device(
        torch_module=_fake_torch_module(cuda_available=False, mps_available=True),
        device_opt="auto",
    )
    dev_cpu = structural_alphabet._resolve_prostt5_device(
        torch_module=_fake_torch_module(cuda_available=False, mps_available=False),
        device_opt="auto",
    )
    assert dev_cuda == "cuda"
    assert dev_mps == "mps"
    assert dev_cpu == "cpu"


def test_resolve_prostt5_device_explicit_mps_requires_backend():
    with pytest.raises(ValueError, match="MPS is not available"):
        structural_alphabet._resolve_prostt5_device(
            torch_module=_fake_torch_module(cuda_available=False, mps_available=False),
            device_opt="mps",
        )
    dev = structural_alphabet._resolve_prostt5_device(
        torch_module=_fake_torch_module(cuda_available=False, mps_available=True),
        device_opt="mps",
    )
    assert dev == "mps"


def test_enable_mps_fallback_if_needed_sets_env_once(monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    changed = structural_alphabet._enable_mps_fallback_if_needed(device="mps")
    assert changed is True
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
    changed_again = structural_alphabet._enable_mps_fallback_if_needed(device="mps")
    assert changed_again is False


def test_enable_mps_fallback_if_needed_ignores_non_mps(monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    changed = structural_alphabet._enable_mps_fallback_if_needed(device="cpu")
    assert changed is False
    assert "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ


def test_enable_mps_fallback_for_option_if_needed_auto_on_darwin(monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.setattr(structural_alphabet.sys, "platform", "darwin")
    changed = structural_alphabet._enable_mps_fallback_for_option_if_needed(device_opt="auto")
    assert changed is True
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def test_enable_mps_fallback_for_option_if_needed_noop_off_darwin(monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.setattr(structural_alphabet.sys, "platform", "linux")
    changed = structural_alphabet._enable_mps_fallback_for_option_if_needed(device_opt="auto")
    assert changed is False
    assert "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ
