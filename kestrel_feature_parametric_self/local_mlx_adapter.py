"""Local MLX LoRA training adapter (Apple Silicon).

Wraps ``mlx_lm.lora`` to train a per-agent LoRA adapter on the agent's own
corpus (see ``corpus.py``). This is the local, text-native training path of
Strategy B (epic #1): it does **not** implement core's image
``TrainingProvider`` protocol (whose ``start_training`` takes ``avatar_data``),
but it **imports** the genuinely modality-neutral lifecycle types
(``TrainingState``, ``TrainingStatus``) from core rather than re-declaring
them — that shared state machine is what keeps the eventual Phase-3
generalization a refactor, not a rewrite.

``TrainingJob`` is intentionally NOT imported: it requires ``companion_id``,
``trigger_word`` and a ``config: TrainingConfig`` (all image-bound), so jobs
are tracked internally instead and surfaced via the neutral ``TrainingStatus``.

MLX is imported lazily and only used on Apple Silicon; on any other platform
``is_available()`` returns False and ``start_training`` raises a clear error
(no blind fallback).
"""

from __future__ import annotations

import platform
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional

from kestrel_sovereign.features.training.types import TrainingState, TrainingStatus

from .text_types import TextLoRAConfig

PROVIDER_NAME = "local_mlx"


class TrainerUnavailableError(RuntimeError):
    """Raised when local MLX training is requested on an unsupported host."""


@dataclass
class _Job:
    """Internal job record (not the image-bound core TrainingJob)."""

    job_id: str
    agent_id: str
    config: TextLoRAConfig
    state: TrainingState
    created_at: float
    process: Optional["object"] = None  # subprocess.Popen, set on launch
    log_path: Optional[str] = None      # captured mlx_lm.lora stdout/stderr
    error: Optional[str] = None


def _mlx_available() -> bool:
    """True only on Apple Silicon with mlx_lm importable."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    try:
        import mlx_lm  # noqa: F401  (lazy: never imported at module load)
    except ImportError:
        return False
    return True


def build_lora_argv(config: TextLoRAConfig) -> list[str]:
    """Build the ``mlx_lm.lora`` training argv for a config.

    Pure function (no MLX import, no side effects) so the command contract is
    unit-testable off Apple Silicon.
    """
    if not config.data_dir:
        raise ValueError("TextLoRAConfig.data_dir is required for training")
    if not config.adapter_path:
        raise ValueError("TextLoRAConfig.adapter_path is required for training")
    return [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model", config.base_model,
        "--train",
        "--data", config.data_dir,
        "--iters", str(config.iters),
        "--batch-size", str(config.batch_size),
        "--num-layers", str(config.num_layers),
        "--learning-rate", str(config.learning_rate),
        "--max-seq-length", str(config.max_seq_length),
        "--adapter-path", config.adapter_path,
    ]


class LocalMLXAdapter:
    """Trains a per-agent LoRA adapter locally via MLX."""

    def __init__(self) -> None:
        self._jobs: Dict[str, _Job] = {}

    @property
    def provider_name(self) -> str:
        return PROVIDER_NAME

    def is_available(self) -> bool:
        """True if this host can run local MLX training."""
        return _mlx_available()

    async def start_training(self, agent_id: str, config: TextLoRAConfig) -> TrainingStatus:
        """Launch a LoRA run and return its initial status.

        Raises:
            TrainerUnavailableError: if the host is not Apple Silicon with MLX.
            ValueError: if the config is missing required paths.
        """
        if not self.is_available():
            raise TrainerUnavailableError(
                "Local MLX training requires Apple Silicon (arm64 macOS) with "
                "the 'mlx-lm' extra installed; this host cannot run it."
            )

        argv = build_lora_argv(config)  # validates config before we spawn
        import subprocess  # lazy: only on the training host
        from pathlib import Path

        # Capture stdout+stderr to a log under the adapter dir so the fidelity
        # gate can read the validation loss mlx_lm.lora prints.
        adapter_dir = Path(config.adapter_path)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(adapter_dir / "train.log")
        log_fh = open(log_path, "w")  # closed in cleanup()/when the proc ends

        job_id = str(uuid.uuid4())
        proc = subprocess.Popen(argv, stdout=log_fh, stderr=subprocess.STDOUT)  # noqa: S603
        self._jobs[job_id] = _Job(
            job_id=job_id,
            agent_id=agent_id,
            config=config,
            state=TrainingState.TRAINING,
            created_at=time.monotonic(),
            process=proc,
            log_path=log_path,
        )
        return self._status(job_id)

    def read_training_log(self, job_id: str) -> str:
        """Return the captured training log for a job ('' if none yet)."""
        job = self._jobs.get(job_id)
        if job is None or not job.log_path:
            return ""
        try:
            with open(job.log_path, "r") as fh:
                return fh.read()
        except OSError:
            return ""

    async def get_status(self, job_id: str) -> TrainingStatus:
        """Poll a job, reconciling against the subprocess exit code."""
        job = self._jobs.get(job_id)
        if job is None:
            return TrainingStatus(
                job_id=job_id,
                state=TrainingState.FAILED,
                progress=0.0,
                error=f"unknown job_id: {job_id}",
            )
        proc = job.process
        if proc is not None and job.state == TrainingState.TRAINING:
            code = proc.poll()
            if code == 0:
                job.state = TrainingState.COMPLETED
            elif code is not None:
                job.state = TrainingState.FAILED
                job.error = f"mlx_lm.lora exited with code {code}"
        return self._status(job_id)

    async def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None or job.process is None:
            return False
        if job.process.poll() is None:
            job.process.terminate()
        job.state = TrainingState.CANCELLED
        return True

    async def cleanup(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)

    def _status(self, job_id: str) -> TrainingStatus:
        job = self._jobs[job_id]
        progress = 1.0 if job.state == TrainingState.COMPLETED else (
            0.0 if job.state in (TrainingState.PENDING, TrainingState.FAILED) else 0.5
        )
        return TrainingStatus(
            job_id=job_id,
            state=job.state,
            progress=progress,
            error=job.error,
            elapsed_seconds=time.monotonic() - job.created_at,
            provider_details={"provider": PROVIDER_NAME, "agent_id": job.agent_id},
        )
