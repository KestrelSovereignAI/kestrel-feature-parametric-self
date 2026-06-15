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

P0 is a scaffold: the feature loads and registers, the sleep-hook surface is
stubbed, and the MLX trainer + reflection corpus land in P1/P2.
"""

from .feature import ParametricSelfFeature

__all__ = [
    "ParametricSelfFeature",
]
