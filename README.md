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
this package source-pins the reviewed capability commit (`797c0f8ba8c84a30f4762b943ac508c0e485d1c0`) rather
than falsely advertising PyPI 0.49.5 as compatible. The coordinated release
must replace that pin with the published capability-bearing version.

## Governed training corpus

Nightly training remains disabled by default and requires an explicit
`GovernedCorpusPolicy` from the host/operator before it can use any example.
The feature asks the agent's storage capability for a checkpointed,
policy-pinned governed assertion snapshot after successful semantic
maintenance; it never reads graph rows, reflection's tables or a local
database directly.

Reflection insights remain a distinct, separately labelled source, but they
arrive only as governed assertions whose source kind is
`REFLECTION_SOURCE_KIND` (`"agent_reflection"`), so the policy's consent,
source and exclusion rules apply to them as to every other example. A
reflection is admitted only when every reflection source names the session(s)
it was derived from and the snapshot's policy can exclude sessions and saved
items (kestrel-sovereign#3849). Otherwise it is left out and counted as
`reflection_unresolved` in the manifest. It is never admitted by default. On a
core release without those exclusions, no reflection reaches training. A
snapshot that includes an example its own policy excludes is refused outright.
A policy restored from its mapping keeps its exclusion sets. A core that cannot
apply a field refuses it rather than dropping it.

**No reflection reaches training yet.** Nothing publishes `"agent_reflection"`
assertions today. `kestrel-feature-reflection` stores insights only in its own
`reflection_insights` table. That table's `session_id` names the reflection
run, not the conversations the insight was drawn from, and nothing records
those conversation sessions. Until a producer records them and publishes
insights through core's governed write path, every reflection is unresolved and
left out.

Each candidate retains an immutable, content-free manifest of its accepted
assertion/revision lineage and semantic checkpoint. A tombstone, stale
revision, missing manifest, or unverifiable host capability quarantines the
affected candidate or served adapter until it is rebuilt. This is intentionally
a visible no-op rather than a fallback to ungoverned factual training.

An adapter is also quarantined if its manifest was built under an earlier
corpus policy, if any example lacks an assertion revision, or if a reflection
example names no session. Adapters from the old policy may hold reflection rows
read straight from reflection's table. Restart, adoption and rollback all refuse
them, so they must be rebuilt rather than grandfathered.

## Inference use

`inference_use` decides whether an adapter may influence a live turn. One
switch covers in-loop serving, oracle consultation and conditioning. `"off"`
is the only value until one of those routes exists, and anything else is
refused. `parametric-self-status` reports it, together with
`adapter_reaches_inference`. While it is off, a promoted, adopted or
rolled-back adapter is a training record only. Any future route must take its
adapter from `ParametricSelfFeature.adapter_for_inference()`, which returns
nothing while the switch is off.

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
specification digests. It also resolves its own clean, full Git revision and
binds that identity into every signed record and report; the verifier must
allow that exact revision. Before every invocation, the independent verifier
persists and issues a one-time freshness nonce; the runner accepts that nonce
as required input and binds it into every signed record and the report. Core
derives and consumes the corresponding receipt through its verifier-owned
ledger, rejecting unknown, replayed, or rewrapped nonces. The runner requires
an explicit owner-only trusted scratch root, creates a fresh owner-only tree
for its candidate manifest and governed corpus, rechecks it while plaintext is
live, and removes only that verified tree on success, failure, or cancellation.
With no custom erasure coordinator, the runner uses
core's scoped physical `erase_assertion` path for the exact assertion
represented in the governed snapshot. This is intentionally not lifecycle
`delete_assertion`: the canonical assertion row and its derived/index/corpus
eligibility are physically removed, while core retains only its blinded,
identity-free erasure audit/tombstone shell for operation replay protection.

The source and lockfile pin the published core evidence commit
`797c0f8ba8c84a30f4762b943ac508c0e485d1c0`, so a normal clean `uv sync` can
install the exact contract implementation. Do not advance that pin to a local
or otherwise unreachable Git SHA.

For Kite, the runner executes the explicit two-phase drill independently on
isolated SQLite and a core-created disposable PostgreSQL database: `prepare`
creates a candidate and proves candidate/served eligibility, core performs
`erase_prepared_assertion`, and `observe` proves invalidation. Only after both
observations agree and both backends are cleaned up does it sign the four
ordered envelope records; the capability report attests the three external
capability records. The standalone `parametric-self-release-evidence` command
accepts a one-argument factory for the runner-owned backend capability, a
verifier-issued nonce, an owner-private CI signing seed, and private
scratch/output paths. It never accepts an assertion ID, caller DSN, or
caller-supplied success result.

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

The contract-evidence tests run against the reachable source pin above; no
local-worktree `PYTHONPATH` override is required.
