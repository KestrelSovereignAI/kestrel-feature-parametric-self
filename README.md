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
this package source-pins the reviewed capability commit (`265cf418`) rather
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

## Semantic release erasure evidence

The #2753 external-adapter drill is available through
`ParametricSelfExternalEvidenceRunner`. It must run only against an isolated
agent and receives the real, scoped core erasure action for that agent's fresh
governed assertion. The runner builds the corpus through core's governed
snapshot, stamps the candidate receipt, invokes the erasure, then requires the
feature's own lineage verifier to quarantine both the candidate and its served
eligibility before it signs anything.

The emitted envelope is content-free: it contains fixed core gate/spec/drill
bindings, positive/zero aggregates, opaque artifacts, and Ed25519
external-CI signatures. It is not a release-ready claim. Core verifies those
signatures later against its operator-owned `TrustedExecutionPolicy` and
attaches the envelope's `ExternalCapabilityReport` only when it matches the
fixed repository/revision contract.

Before a drill, the runner binds the immutable core external-adapter contract
digest, which covers the release-evidence schema/contract and ordered gate
specification digests. Before every invocation, the independent verifier
persists and issues a one-time freshness nonce; the runner accepts that nonce
as required input and binds it into every signed record and the report. Core
derives and consumes the corresponding receipt through its verifier-owned
ledger, rejecting unknown, replayed, or rewrapped nonces. Its complete fresh scratch tree
— including the candidate manifest and corpus files — is removed on success,
failure, or cancellation. With no custom erasure coordinator, the runner uses
core's scoped physical `erase_assertion` path for the exact assertion
represented in the governed snapshot. This is intentionally not lifecycle
`delete_assertion`: the canonical assertion row and its derived/index/corpus
eligibility are physically removed, while core retains only its blinded,
identity-free erasure audit/tombstone shell for operation replay protection.

During coordinated development, an unpushed core evidence commit is tested
with `uv run --no-sync` and `PYTHONPATH` pointing at that local core worktree.
The package's `pyproject.toml` and lockfile remain pinned to the current
published/reviewed core commit until a new core commit is pushed and can be
resolved by a clean installer; do not publish an unreachable Git SHA.

## Incomplete-shutdown recovery

If status reports a shutdown restored from a prior process, training remains
blocked and any retained per-run corpus stays in place. A new adapter's empty
job list cannot prove that an old trainer exited. After a sovereign operator
checks that every prior trainer process is absent, they must record that fresh
evidence explicitly:

```text
!parametric-self-recover-shutdown confirmed_process_absent=true evidence="ps check found no prior mlx_lm.lora process"
```

The command is sovereign-class gated. It refuses an omitted/false confirmation,
missing evidence, or a still-live feature-owned task; on success it
terminalizes the durable run record and removes only its recorded per-run
corpus files.

## Development

```bash
uv sync --extra test
uv run --extra test pytest
```

Until the coordinated core branch is published, contract-evidence tests use
the local core evidence worktree with `--no-sync` and `PYTHONPATH`; the package
source pin is not advanced ahead of that publication.
