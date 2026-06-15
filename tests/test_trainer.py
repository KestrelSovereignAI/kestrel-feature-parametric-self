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
