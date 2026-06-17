"""Tests for TextLoRAConfig and LocalMLXAdapter (Linux-friendly).

These verify the config contract, the pure command builder, and that the
adapter degrades cleanly (no blind fallback) on a non-Apple-Silicon host —
without requiring MLX to be installed.
"""

from __future__ import annotations

import sys

import pytest

from kestrel_feature_parametric_self import (
    LocalMLXAdapter,
    TextLoRAConfig,
    TrainerUnavailableError,
    build_lora_argv,
)
from kestrel_feature_parametric_self.text_types import DEFAULT_BASE_MODEL


def test_config_roundtrips_and_has_no_image_fields():
    cfg = TextLoRAConfig(data_dir="/tmp/corpus", adapter_path="/tmp/adapter")
    d = cfg.to_dict()
    assert TextLoRAConfig.from_dict(d) == cfg
    assert cfg.base_model == DEFAULT_BASE_MODEL
    # Strategy B: no image-bound fields leaked into the text config.
    for image_field in ("trigger_word", "resolution", "flux_version"):
        assert image_field not in d


def test_build_lora_argv_is_pure_and_complete():
    cfg = TextLoRAConfig(data_dir="/data", adapter_path="/adapters/a", iters=123, num_layers=8)
    argv = build_lora_argv(cfg)
    assert argv[:4] == [sys.executable, "-m", "mlx_lm", "lora"]
    assert "--train" in argv
    assert argv[argv.index("--data") + 1] == "/data"
    assert argv[argv.index("--iters") + 1] == "123"
    assert argv[argv.index("--num-layers") + 1] == "8"
    assert argv[argv.index("--adapter-path") + 1] == "/adapters/a"


def test_build_lora_argv_requires_paths():
    with pytest.raises(ValueError):
        build_lora_argv(TextLoRAConfig())  # no data_dir / adapter_path


def test_is_available_false_off_apple_silicon():
    adapter = LocalMLXAdapter()
    # On CI (Linux) this must be False; this test documents the platform gate.
    if sys.platform != "darwin":
        assert adapter.is_available() is False


async def test_start_training_raises_clearly_when_unavailable():
    adapter = LocalMLXAdapter()
    if adapter.is_available():
        pytest.skip("MLX available on this host; the unavailable-path test does not apply")
    cfg = TextLoRAConfig(data_dir="/data", adapter_path="/adapters/a")
    with pytest.raises(TrainerUnavailableError):
        await adapter.start_training("agent-1", cfg)


def test_mlx_available_handles_metal_runtime_error(monkeypatch):
    """import mlx_lm can succeed-then-raise RuntimeError (no Metal device) on a
    GPU-less host; _mlx_available must return False, never propagate (Emma review)."""
    import kestrel_feature_parametric_self.local_mlx_adapter as mod

    monkeypatch.setattr(mod.sys, "platform", "darwin")
    monkeypatch.setattr(mod.platform, "machine", lambda: "arm64")

    real_import = __import__

    def boom(name, *args, **kwargs):
        if name == "mlx_lm":
            raise RuntimeError("No Metal device available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", boom)
    assert mod._mlx_available() is False  # must not raise


def test_is_available_never_raises(monkeypatch):
    """is_available() is safe to call from the status tool on any host."""
    adapter = LocalMLXAdapter()
    assert isinstance(adapter.is_available(), bool)


async def test_cancel_all_terminates_live_jobs():
    """cancel_all terminates running subprocesses and skips already-exited ones."""
    from unittest.mock import MagicMock
    from kestrel_feature_parametric_self.local_mlx_adapter import _Job
    from kestrel_sovereign.features.training.types import TrainingState

    adapter = LocalMLXAdapter()
    live = MagicMock(); live.poll.return_value = None        # still running
    done = MagicMock(); done.poll.return_value = 0           # already exited
    adapter._jobs = {
        "a": _Job("a", "agent", TextLoRAConfig(), TrainingState.TRAINING, 0.0, process=live),
        "b": _Job("b", "agent", TextLoRAConfig(), TrainingState.COMPLETED, 0.0, process=done),
    }
    n = await adapter.cancel_all()
    assert n == 1
    live.terminate.assert_called_once()
    done.terminate.assert_not_called()
    assert adapter._jobs["a"].state == TrainingState.CANCELLED
