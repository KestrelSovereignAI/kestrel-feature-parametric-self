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

## Promotion gate

The loss gate decides whether a trained candidate may be promoted. It measures
validation loss, which is fit to the agent's own data only. **It says nothing
about identity drift**; the behavioural fidelity exam (epic #1) is the real gate.

- Training saves a checkpoint every `save_every` steps. When it ends, every
  saved checkpoint file, including the final weights, is scored with
  `mlx_lm.lora --test` on the run's whole validation set. The gate picks the
  **best** checkpoint by that score, never the final one by default, and every
  decision names checkpoint ids. The in-training `Iter N` validation loss is
  progress only: mlx_lm measures it one optimizer step before it saves
  checkpoint N, so it never describes a saved file.
- The gate is **relative**. The baseline is the served adapter's promoted
  checkpoint, or the untuned base (`--test` with no adapter) when nothing is
  served. It is scored the same way, on the same validation set, in the same
  run; a stored loss is never a baseline.
- The baseline is whatever was served, on the configured base model, when the
  run started. The cycle checks both again just before it scores the baseline,
  and again before the gate decides. If either changed (the served adapter was
  quarantined or replaced, or the base model was reconfigured), the decision is
  `inconclusive` (`baseline_changed`). The changed weights are never loaded. If
  the baseline changes after a promote decision is recorded, that promotion is
  blocked.
- A served adapter is the baseline only while it passes the governed-lineage
  check that verifies it for serving at restart and on status. The cycle runs
  that check against a fresh governed snapshot just before it loads the
  adapter, so a revocation between runs is caught without a status call. If
  the check fails, or cannot run because the snapshot is unavailable, the
  adapter is quarantined and its weights are never loaded. The decision is
  then `inconclusive` (`baseline_changed`); it never falls back to the base.
- Order of checks: inconclusive inputs, then overfit, then margin; the first
  failure decides. Config keys, overridable through `set_config({"gate": {...}})`
  and persisted:
  - `gate.margin` (default `0.02`): promote only if
    `best < baseline × (1 − margin)`. Equality fails.
  - `gate.overfit_ratio` (default `1.20`): reject when `final / best` is
    greater. Equality passes.
  - `gate.overfit_check` (default `true`).
- `inconclusive` (no validation set, an empty one, one smaller than a single
  `batch_size` batch, which mlx_lm cannot score, a missing or non-finite loss
  for any checkpoint, whatever `gate.overfit_check` says, no checkpoints, a
  checkpoint or baseline that cannot be scored, a baseline that changed during
  the run or a served adapter whose lineage check failed) blocks promotion and
  is never reported healthy by `parametric-self-status` or the sleep hook.
  This holds for adopt too: a recorded run with one unscored checkpoint is not
  adopted. A run whose validation set is too small to score trains nothing.
- Every decision writes a record into run history **before** any promotion.
  The record holds the best/final checkpoint ids and losses, every
  checkpoint's score, the baseline source and loss, why the baseline could not
  be used, and the base model every score was measured on. It also holds the
  margin, the ratio, the validation-set digest, size and scoring batch size,
  and the decision and reason.
  Missing values are explicit nulls. If that write fails, nothing is promoted.
  A history write never replaces history it could not read intact.
- Status health comes from the latest gate attempt alone. An attempt that was
  inconclusive, or whose record did not persist, is not healthy, whatever an
  earlier run decided. Each training run's and adoption's history entry
  carries a pending gate marker from the moment it starts. If that entry
  cannot be written, the run or adoption does not start, so no gate attempt
  can go unrecorded. The marker is cleared by the same write that records the
  gate decision, or by the terminal entry. A run or adoption that is cancelled
  while its gate record is being written cannot tell whether the record
  landed. Its terminal entry leaves the marker as that write left it, and
  status reads that attempt from history, as it would after a restart. A run
  that stopped with the marker still set (a restart, a failed write, or a
  cancelled record write) may have reached the gate, so it is an unresolved
  attempt and is never healthy.
- Adopt re-takes the gate on the scores the candidate's own training run
  recorded, because its validation set is gone. If the baseline or the base
  model has changed since, or a served baseline fails its governed-lineage
  check, the result is `inconclusive`. If either changes, or
  the candidate is quarantined, while the decision is being recorded, nothing
  is adopted, as for a training run. Rollback restores the checkpoint an
  adapter was promoted with, at its promoted score. Quarantined adapters are
  never re-scored, adopted or rolled back to.
- `parametric-self-adapters` and `parametric-self-status` never read a
  training log for a score. Each candidate shows its served or promoted
  checkpoint and score, or else the best and final checkpoint scores its
  training run's gate recorded. A candidate is listed as recoverable only when
  nothing is served and adopt, re-taking the gate on those recorded scores,
  would promote it. For any other candidate that is not served, in progress or
  quarantined, the listing shows what adopt would decide.

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
