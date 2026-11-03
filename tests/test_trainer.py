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
    build_eval_argv,
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


def test_build_lora_argv_passes_the_checkpoint_and_progress_cadence():
    """save_every sets which checkpoints the gate scores (#11); evaluation is progress only."""
    cfg = TextLoRAConfig(data_dir="/data", adapter_path="/adapters/a", steps_per_eval=200, save_every=100)
    argv = build_lora_argv(cfg)
    assert argv[argv.index("--steps-per-eval") + 1] == "200"
    assert argv[argv.index("--save-every") + 1] == "100"
    assert argv[argv.index("--val-batches") + 1] == "-1"
    defaults = build_lora_argv(TextLoRAConfig(data_dir="/data", adapter_path="/adapters/a"))
    assert defaults[defaults.index("--steps-per-eval") + 1] == "100"
    assert defaults[defaults.index("--save-every") + 1] == "100"


@pytest.mark.parametrize("steps_per_eval, save_every", [(100, 0), (0, 100), (-100, 100)])
def test_build_lora_argv_refuses_a_non_positive_cadence(steps_per_eval, save_every):
    cfg = TextLoRAConfig(
        data_dir="/data", adapter_path="/adapters/a",
        steps_per_eval=steps_per_eval, save_every=save_every,
    )
    with pytest.raises(ValueError):
        build_lora_argv(cfg)


def test_build_eval_argv_scores_one_adapter_like_the_training_eval():
    cfg = TextLoRAConfig(
        data_dir="/scratch/baseline", adapter_path="/scratch/baseline/adapter",
        batch_size=2, max_seq_length=1024,
    )
    argv = build_eval_argv(cfg)
    assert argv[:4] == [sys.executable, "-m", "mlx_lm", "lora"]
    assert "--test" in argv and "--train" not in argv
    assert argv[argv.index("--data") + 1] == "/scratch/baseline"
    assert argv[argv.index("--adapter-path") + 1] == "/scratch/baseline/adapter"
    assert argv[argv.index("--batch-size") + 1] == "2"
    assert argv[argv.index("--max-seq-length") + 1] == "1024"
    assert argv[argv.index("--test-batches") + 1] == "-1"


def test_build_eval_argv_scores_the_untuned_base_with_no_adapter():
    """mlx_lm loads no adapter for an empty --adapter-path, so it scores the base."""
    argv = build_eval_argv(TextLoRAConfig(data_dir="/scratch/scoring", adapter_path=None))
    assert argv[argv.index("--adapter-path") + 1] == ""
    with pytest.raises(ValueError):
        build_eval_argv(TextLoRAConfig(data_dir="/scratch/scoring", adapter_path=""))


def test_build_lora_argv_requires_paths():
    with pytest.raises(ValueError):
        build_lora_argv(TextLoRAConfig())  # no data_dir / adapter_path
    with pytest.raises(ValueError):
        build_eval_argv(TextLoRAConfig())


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
    with pytest.raises(TrainerUnavailableError):
        await adapter.start_evaluation("agent-1", cfg, log_dir="/scratch/scoring/base")


async def test_evaluation_is_a_tracked_job_with_its_own_log(tmp_path, monkeypatch):
    """Scoring jobs share the training lifecycle, so cancel/cancel_all cover them."""
    import subprocess
    from unittest.mock import MagicMock

    adapter = LocalMLXAdapter()
    monkeypatch.setattr(adapter, "is_available", lambda: True)
    launched = []
    child = MagicMock()
    child.poll.return_value = 0

    def _popen(argv, stdout, stderr):
        launched.append((argv, stdout.name))
        stdout.write("Testing\nTest loss 2.208, Test ppl 9.098.\n")
        return child

    monkeypatch.setattr(subprocess, "Popen", _popen)
    view = tmp_path / "scoring" / "candidate-0000200"
    config = TextLoRAConfig(data_dir=str(tmp_path / "scoring"), adapter_path=str(view / "adapter"))
    status = await adapter.start_evaluation("agent-1", config, log_dir=str(view))
    assert launched[0][0] == build_eval_argv(config)
    # The log sits beside the scored adapter view, never inside it.
    assert launched[0][1] == str(view / "eval.log")
    assert "Test loss 2.208" in adapter.read_training_log(status.job_id)
    assert status.job_id in adapter._jobs


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


async def test_cancel_all_terminates_and_confirms_live_jobs():
    """cancel_all confirms a child exited before callers may delete its inputs."""
    from unittest.mock import MagicMock
    from kestrel_feature_parametric_self.local_mlx_adapter import _Job
    from kestrel_sovereign.features.training.types import TrainingState

    adapter = LocalMLXAdapter()
    live = MagicMock(); live.poll.side_effect = [None, None, 0]
    done = MagicMock(); done.poll.return_value = 0           # already exited
    adapter._jobs = {
        "a": _Job("a", "agent", TextLoRAConfig(), TrainingState.TRAINING, 0.0, process=live),
        "b": _Job("b", "agent", TextLoRAConfig(), TrainingState.COMPLETED, 0.0, process=done),
    }
    result = await adapter.cancel_all()
    assert result.cancelled == 1
    assert result.all_stopped is True
    live.terminate.assert_called_once()
    done.terminate.assert_not_called()
    assert adapter._jobs["a"].state == TrainingState.CANCELLED


async def test_cancel_all_distinguishes_no_jobs_from_failed_stop_confirmation():
    """A zero count is informative only with the separate all_stopped flag."""
    from unittest.mock import MagicMock
    from kestrel_feature_parametric_self.local_mlx_adapter import _Job
    from kestrel_sovereign.features.training.types import TrainingState

    empty = await LocalMLXAdapter().cancel_all()
    assert empty.cancelled == 0
    assert empty.all_stopped is True

    adapter = LocalMLXAdapter()
    live = MagicMock(); live.poll.return_value = None
    adapter._jobs = {
        "a": _Job("a", "agent", TextLoRAConfig(), TrainingState.TRAINING, 0.0, process=live),
    }
    unresolved = await adapter.cancel_all()
    assert unresolved.cancelled == 0
    assert unresolved.all_stopped is False
    assert adapter._jobs["a"].state == TrainingState.TRAINING
