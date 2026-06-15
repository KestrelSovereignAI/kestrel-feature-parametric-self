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
corpus builder, and ``LocalMLXAdapter`` (Apple-Silicon MLX LoRA). Sleep-hook
wiring + the fidelity gate are P2.
"""

from .corpus import CorpusStats, build_corpus
from .feature import ParametricSelfFeature
from .local_mlx_adapter import LocalMLXAdapter, TrainerUnavailableError, build_lora_argv
from .text_types import TextLoRAConfig

__all__ = [
    "ParametricSelfFeature",
    "TextLoRAConfig",
    "build_corpus",
    "CorpusStats",
    "LocalMLXAdapter",
    "TrainerUnavailableError",
    "build_lora_argv",
]
