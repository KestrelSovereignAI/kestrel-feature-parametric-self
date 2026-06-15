# kestrel-feature-parametric-self — Agent Instructions

This repository packages the Kestrel parametric-self feature: the agent's
owned, nightly-finetuned local model (the "second brain").

## Key Files

- `kestrel_feature_parametric_self/feature.py` — `ParametricSelfFeature` entry point + sleep-cycle hook.
- `kestrel_feature_parametric_self/component.yaml` — feature manifest.

## Design

- `docs/research/TWO_BRAIN_ARCHITECTURE.md` in `kestrel-sovereign` (rationale, phases, risks).
- Epic #1 (this repo) tracks the phased build.

## Conventions

- Depends on `kestrel-feature-reflection` as its corpus source (P1); does not live inside it.
- The MLX trainer is Apple-Silicon only and imported lazily — never at module import — so the package installs and CI-validates on Linux.
- This is the *parametric* self (weights), distinct from reflection's *symbolic* self-model. It is not memory.

## Running Tests

```bash
uv run --extra test pytest
```
