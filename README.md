# kestrel-feature-parametric-self

The agent's **owned parametric self** for Kestrel Sovereign.

A per-agent local model (target: Gemma 4 31B, 4-bit MLX) that is
nightly-finetuned during the sleep cycle on the agent's own experience, and —
once proven — consulted in the agent's reasoning loop as a disposition prior
and on-demand oracle alongside the frontier model.

Slogan: **rent intelligence, own identity.** This is the *parametric*
counterpart to reflection's *symbolic* self-model (the weights, not a trait
dict). It is **not** memory — RAG remains the factual layer.

Design: [`docs/TWO_BRAIN_ARCHITECTURE.md`](docs/TWO_BRAIN_ARCHITECTURE.md).
Build plan: epic #1.

> **Status: P0 scaffold.** The feature loads and registers; the MLX trainer,
> reflection-derived corpus, fidelity gate, and in-loop integration land in
> later phases. The Apple-Silicon trainer is imported lazily, so the package
> installs and CI-validates on any platform.

## Installation

```bash
uv pip install kestrel-feature-parametric-self
```

The package registers `ParametricSelfFeature` through the
`kestrel_sovereign.features` entry point group.

The governed-corpus host surface is currently under development in core. Until
the first core release containing [#2817](https://github.com/KestrelSovereignAI/kestrel-sovereign/pull/2817),
this package source-pins the reviewed capability commit (`5932735a`) rather
than falsely advertising PyPI 0.49.5 as compatible. The coordinated release
must replace that pin with the published capability-bearing version.

## Governed training corpus

Nightly training remains disabled by default and requires an explicit
`GovernedCorpusPolicy` from the host/operator before it can use factual
examples. The feature asks the agent's storage capability for a checkpointed,
policy-pinned governed assertion snapshot after successful semantic
maintenance; it never reads factual graph rows or a local database directly.
Reflection insights remain a distinct source.

Each candidate retains an immutable, content-free manifest of its accepted
assertion/revision lineage and semantic checkpoint. A tombstone, stale
revision, missing manifest, or unverifiable host capability quarantines the
affected candidate or served adapter until it is rebuilt. This is intentionally
a visible no-op rather than a fallback to ungoverned factual training.

## Development

```bash
uv sync --extra test
uv run --extra test pytest
```
