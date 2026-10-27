"""Parametric Self feature for Kestrel Sovereign.

The owned parametric self: a per-agent local model (target: Gemma 4 31B,
4-bit MLX) that is nightly-finetuned during the sleep cycle on the agent's
own experience, and — once proven — consulted in the agent's reasoning loop
as a disposition prior and on-demand oracle.

This is the *parametric* counterpart to reflection's *symbolic* self-model
(``kestrel_feature_reflection.SelfModelManager``): the weights, not a trait
dict. It is **not** memory — RAG remains the factual layer.

See ``docs/TWO_BRAIN_ARCHITECTURE.md`` and
epic #1 for the design and phased build.

P1 adds the text-native training path: ``TextLoRAConfig``, the reflection-derived
corpus builder, and ``LocalMLXAdapter`` (Apple-Silicon MLX LoRA).

P2a adds the nightly orchestration behind a default-OFF gate: the ``FidelityGate``
(relative held-out val-loss promotion check on the run's best checkpoint),
``run_nightly_cycle`` (corpus→train→score→gate),
and ``ParametricSelfSleepHook`` (the sleep-cycle wrapper). Core sleep-hook
dispatch is P2b.
"""

from .corpus import CorpusStats, build_corpus
from .cycle import CycleResult, ServedAdapter, run_nightly_cycle
from .feature import ParametricSelfFeature
from .fidelity import (
    Baseline,
    Checkpoint,
    FidelityGate,
    GateDecision,
    GateOutcome,
    GateReason,
    ValSet,
    parse_final_val_loss,
    parse_latest_iter,
    parse_test_loss,
    recorded_inputs,
    saved_checkpoints,
)
from .local_mlx_adapter import (
    LocalMLXAdapter,
    TrainerUnavailableError,
    build_eval_argv,
    build_lora_argv,
)
from .release_evidence import (
    CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST,
    ExternalReleaseEvidenceEnvelope,
    ExternalReleaseEvidenceError,
    KiteErasureBackend,
    KiteErasurePreparation,
    ParametricSelfExternalEvidenceRunner,
    ParametricSelfKiteErasureHook,
)
from .sleep_hook import ParametricSelfSleepHook, create_parametric_self_sleep_hook
from .text_types import TextLoRAConfig

__all__ = [
    "ParametricSelfFeature",
    "TextLoRAConfig",
    "build_corpus",
    "CorpusStats",
    "LocalMLXAdapter",
    "TrainerUnavailableError",
    "build_lora_argv",
    "build_eval_argv",
    "CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST",
    "ExternalReleaseEvidenceEnvelope",
    "ExternalReleaseEvidenceError",
    "KiteErasureBackend",
    "KiteErasurePreparation",
    "ParametricSelfExternalEvidenceRunner",
    "ParametricSelfKiteErasureHook",
    "FidelityGate",
    "GateDecision",
    "GateOutcome",
    "GateReason",
    "Checkpoint",
    "Baseline",
    "ValSet",
    "saved_checkpoints",
    "recorded_inputs",
    "parse_final_val_loss",
    "parse_latest_iter",
    "parse_test_loss",
    "run_nightly_cycle",
    "CycleResult",
    "ServedAdapter",
    "ParametricSelfSleepHook",
    "create_parametric_self_sleep_hook",
]
