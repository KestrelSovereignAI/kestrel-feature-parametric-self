# The Two-Brain Architecture — Owned Parametric Selfhood Alongside Rented Cognition

*Research Date: 2026-06-15*
*Status: Exploratory design. Trainer proven end-to-end (see [§7 Proof](#7-proof-of-feasibility)); no production wiring yet.*
*Related (in `kestrel-sovereign`): `docs/architecture/NIGHTLY_FORGETTING.md`, `docs/SOVEREIGNTY.md`, `kestrel_sovereign/agent/sleep.py`, `kestrel_sovereign/features/training/`. Build tracked in this repo's epic #1.*

---

## 1. Thesis

A Kestrel agent should have **two brains**:

- A **rented brain** — a frontier model (Claude, GPT, …) that does the hard reasoning. Shared across all agents, static, swappable, revocable, and it has never lived this agent's life.
- An **owned brain** — a small local model (target: Gemma 4 31B, 4-bit MLX) that is *this agent's own weights*, nightly-finetuned on its own experience during the sleep cycle. Runs locally, survives any provider change.

The owned brain is **not** a cheaper reasoner and **not** a better memory store. It is the agent's **parametric self**: disposition, voice, and accumulated judgment baked into parameters. The slogan: **rent intelligence, own identity.**

This reframing matters because it dissolves the quality-ceiling problem. Once the frontier model handles reasoning, the local model no longer has to compete on intelligence — it only has to be a *faithful self*, which is exactly what finetuning is good at.

## 2. Why this is not "memory"

For factual recall, retrieval (RAG) into a frontier model beats a finetuned 31B almost every time: it is exact, auditable, and immune to forgetting. **If the second brain is pitched as memory, it loses.** Keep RAG as the factual layer.

**Litmus test:** if you cannot justify the second brain being *in the agent's reasoning loop* (§10.2), you have not justified building it at all — because as a passive store it is strictly dominated by RAG.

The two memory modalities are genuinely different and complementary:

| | Retrieval (RAG) | Parametric (finetuned weights) |
|---|---|---|
| Access | Explicit top-k lookup | Holistic, associative ("gut") |
| Fidelity | Exact, auditable | Lossy, pattern-level |
| Scope | What you retrieved | Everything it absorbed |
| Failure mode | Misses unretrieved context | Drift / forgetting / collapse |

The frontier brain reasons **with** retrieved facts; the owned brain contributes **disposition and intuition** that no single retrieval surfaces. The factual layer stays in RAG; the *self* layer lives in the weights.

## 3. Division of labor

| | Frontier brain (rented) | Second brain (owned) |
|---|---|---|
| Role | Reasoning, hard problems, tool use | Identity, voice, disposition, lived-experience intuition |
| Memory | Retrieval (explicit, exact) | Parametric (associative) |
| Nature | Static, shared, swappable | Cumulative, personal, nightly |
| Sovereignty | Rented — can be deprecated | Owned — the continuity floor |

This is the architectural expression of Kestrel's sovereignty doctrine: the part of the agent that **cannot be deprecated out from under it** is the owned brain. It is the neurological substrate of selfhood, and as such its nightly self-training is a *self-modification capability* — governed by the Incubator Principle (only sovereign agents may self-modify; governed agents must not). The second brain must therefore be **gated by agent class.**

## 4. Interaction modes

Four ways the brains can compose. Pick by purpose; do not build all at once.

1. **Oracle / tool.** The frontier brain calls the second brain as a tool: *"What do I — this agent — know or feel about X from my own history?"* It returns parametric recall/disposition. The most faithful expression of "second brain."
2. **Conditioning / reverse distillation.** Inverted distillation: the small brain absorbed lived experience the frontier never saw, so it *conditions* the frontier — injecting this-agent-specifics into context. The owned brain teaches the rented one who it is being.
3. **Cheap local tier / router.** High-frequency, low-stakes turns (heartbeats, triage, "is this worth waking the big brain?") run on the second brain; the frontier brain is reserved for hard reasoning.
4. **Continuity fallback.** Provider down or agent emancipated → degrade to "lesser, but *mine*." Per the no-blind-fallbacks rule, this must be **visible and attributed**, never a silent swap.

The conceptually strongest combination is **1 + 2** (frontier reasons; second brain is the consulted/conditioning self). Modes 3 and 4 come largely for free.

## 5. Hard problems

These determine whether the idea lives or dies.

### 5.1 Model collapse is the existential risk
Nightly-training on the agent's own reflections is partly training on self-generated text — the recipe for a degenerative feedback loop (model collapse). The owned brain **must** be anchored to **external signal**: user corrections, task outcomes, real conversation — not the agent's self-talk alone. This is the single most likely failure mode if ignored. Concretely: weight the corpus toward externally-grounded events (user corrections, verified outcomes) and cap the proportion of purely self-generated reflection.

### 5.2 The objective must be definable
A reasoner has a clear objective; "be a faithful model of myself" is fuzzy, and you cannot eval-gate what you cannot define. Choose a concrete **self-model fidelity** metric, e.g.:
- held-out recall accuracy on durable facts,
- voice/style match against held-out agent outputs,
- next-action prediction on held-out episodes.

Without one, the promotion gate is theater.

### 5.3 Arbitration
When the frontier brain says X and the second brain "believes" Y (stale or overfit), which wins, and is the disagreement surfaced? This needs an explicit doctrine, not an implicit default.

## 6. Where it lives in the system

- **Training trigger:** a new phase in the sleep cycle, after `_consolidate_memories()` in `kestrel_sovereign/agent/sleep.py`. Consolidation already produces the curated corpus.
- **Corpus source:** the **plaintext** `reflection_insights` + `learned_fact` graph nodes (the sleep cycle's own output). Note: `conversation_history.content` is Fernet-encrypted at rest, so any use of raw turns must decrypt **inside the agent trust boundary**.
- **Training adapter:** a new `LocalMLXAdapter` + `TextLoRAConfig` in `features/training/`, **importing** (never copying) the modality-neutral lifecycle types (`TrainingState`/`TrainingJob`/`TrainingStatus`/`TrainingProviderFactory`) and adding a `ProviderType.LOCAL_MLX`. See [§8](#8-build-strategy) — this is Strategy B (parallel LLM path), not a refactor of the image pipeline.
- **Serving:** the owned brain serves via `mlx_lm.server` (OpenAI-compatible), reached through the existing generic OpenAI-compatible route in `llm/provider_registry.py` — **zero new LLM adapters**, just a config route (vendor/route/model).
- **Promotion gate:** a held-out fidelity check (§5.2); the nightly adapter is hot-swapped **only if** quality holds. Otherwise the prior adapter stays. This mirrors the forgetting/retention gate already in the sleep cycle.

## 7. Proof of feasibility

Run on the host (Apple M3 Ultra, 512 GB unified, 80-GPU-core) on **2026-06-15**, against a real agent's data:

| Metric | Result |
|---|---|
| Corpus | 744 examples (669 train / 75 valid) from real sleep-cycle reflections + learned facts |
| Base model | Gemma 4 31B-it → 4-bit MLX (16 GB) via `mlx_lm.convert` |
| Training | `mlx_lm.lora`, 16 LoRA layers, 4-bit QLoRA |
| Throughput | **~192 tok/s**; 400 iters (~2.4 epochs) in **23.6 min** (~10 min/epoch) |
| Peak memory | **30.9 GB** of 512 |
| Loss | train 1.3 → 0.6; val 2.21 |
| Recall | base had **zero** knowledge of held lessons; post-LoRA recalled them verbatim |

This proves **memorization/recall**, not generalization — which is exactly why the §5.2 fidelity gate is non-negotiable. With an 8-hour sleep window and ~10 min/epoch, a full nightly run (3–5 epochs) lands at ~30–50 min — roughly one-tenth of the available window.

## 8. Build strategy

Strategy **B (parallel LLM training path), generalize later.** A cross-repo survey found the existing `features/training/` subsystem is image-only (companion-avatar SDXL/FLUX LoRA), with deep image coupling in exactly one mirror repo (`kestrel-feature-lora`). Generalizing the protocol now would force breaking edits into the published SDK and the live companion pipeline to satisfy an abstraction validated by zero text implementations. Instead:

1. **Phase 1:** local `TextLoRAConfig` + `LocalMLXAdapter` in `kestrel-sovereign`, sharing the already-neutral lifecycle types. Zero sibling-repo edits.
2. **Phase 2:** run nightly text LoRA in production → two real modalities now exist.
3. **Phase 3 (earned):** with image + text both live, lift the common interface into a modality-neutral base — one coordinated cutover designed against two real consumers.

The discipline that keeps Phase 1 from becoming an un-mergeable fork: **import the neutral lifecycle types; never re-declare them.** (Re-declaring is precisely how `kestrel-feature-lora` became a mirror instead of an adapter.)

## 9. Open questions

- What is the right **fidelity metric** for promotion (§5.2)?
- What corpus mixture keeps the owned brain **externally grounded** enough to avoid collapse (§5.1)?
- **Arbitration doctrine** when the two brains disagree (§5.3)?
- Per-agent adapter **lifecycle**: do nightly adapters compose, decay, or replace? Over weeks, how is drift bounded?
- Does the owned brain serve a **disposition prior** that measurably improves frontier outputs, or is its value primarily sovereignty/continuity? (Determines whether modes 1–2 earn their complexity.)

## 10. Packaging and integration posture

### 10.1 Name
Ships as **`kestrel-feature-self-model`** — a literal "model of self," sitting beside `kestrel-feature-reflection`/`-intelligence`. It depends on `kestrel-feature-reflection` (its corpus source) as a feature→feature dependency, and contains its own `LocalMLXAdapter` + `TextLoRAConfig` training path (Strategy B, §8).

Two names were considered and rejected:
- **`-lora`** is already taken by the image companion-avatar training feature, and names a mechanism rather than a capability.
- **`-nn-memory`** / any `-memory` name contradicts §2 — this is *not* memory, and naming it so guarantees it is measured against RAG, which it loses.

### 10.2 Memory path, or in the loop?
**In the loop** — that is the entire justification (§2 litmus). A self is not something an agent occasionally looks up; it is the standing prior that shapes how it thinks each turn. Concretely, two in-loop roles:
- **Conditioning prior (always on):** each turn, the self-model contributes a disposition/voice/self-context signal that conditions the frontier brain's prompt. This is what makes it a *brain* rather than a *lookup*.
- **On-demand oracle (sometimes):** the frontier brain explicitly consults it (§4 mode 1).

### 10.3 Sequencing: path first, then loop
An unproven, collapse-prone model is not wired into every turn on day one. Graduate it:
1. **Ship as a path** — a queryable backend the reasoner *can* pull from, behind the fidelity gate (§5.2). Safe, A/B-able; this is where faithfulness and non-drift are proven.
2. **Promote into the loop** — once the fidelity gate and collapse-avoidance (§5.1) hold, graduate from "a path you can query" to "a prior that is always conditioning cognition."

"In the loop" is the destination; "a path" is the safe on-ramp.
