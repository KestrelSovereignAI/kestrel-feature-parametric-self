"""Parametric Self feature — owned, nightly-finetuned, in-the-loop second brain.

P2a wires the real nightly training cycle behind a default-OFF gate: after
memory consolidation, ``on_post_consolidation`` builds a corpus from the night's
reflections, trains a candidate LoRA adapter, and promotes it only if it clears
the fidelity gate (§5.2 of ``docs/TWO_BRAIN_ARCHITECTURE.md``). Nothing runs
unless training is explicitly enabled for this agent AND the host can run MLX.

The sleep cycle calls a ``*SleepHook`` wrapper (see ``sleep_hook.py``), not this
feature method directly. How that wrapper gets dispatched is P2b: ``sleep.py``
today exposes a single ``agent.reflection_hook`` slot owned by reflection, so a
proper fix generalizes it to a sleep-hook list (epic #1).

Design boundary: this is the *parametric* self (weights). Reflection keeps the
*symbolic* self-model. This feature depends on reflection as its corpus source
but does not live inside it — it is active in the runtime reasoning loop, not
only during sleep.
"""

import asyncio
import json
import logging
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from kestrel_sdk.features.base import Feature, tool
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

from .cycle import TrainingShutdownIncomplete, run_nightly_cycle
from .fidelity import FidelityGate, parse_final_val_loss, parse_latest_iter
from .local_mlx_adapter import LocalMLXAdapter
from .text_types import TextLoRAConfig

logger = logging.getLogger(__name__)

# How many recent run-history entries to retain (append-only, capped).
_RUN_HISTORY_LIMIT = 50


def _utc_now_iso() -> str:
    """UTC timestamp for run-history records."""
    return datetime.now(timezone.utc).isoformat()


def _strip_key_prefix(value: Any) -> Any:
    """Tolerate a leaked ``key=value`` token from the command parser.

    Two command parsers are in play depending on host wiring: one splits
    ``enabled=false`` into ``"false"``, the other (positional) hands the whole
    token ``"enabled=false"`` to the param. A real value (a bool spelling or a
    uuid-hex adapter id) never contains ``=``, so taking the substring after the
    last ``=`` is a safe normalization that makes the tools correct under both.
    """
    if isinstance(value, str) and "=" in value:
        return value.rsplit("=", 1)[1]
    return value


def _as_bool(value: Any) -> bool:
    """Coerce a tool argument to bool.

    The command path delivers args as strings (e.g. ``enabled=false``), so a bare
    ``bool("false")`` would be ``True`` and silently enable training. Treat the
    usual falsey string spellings as False; otherwise fall back to truthiness.
    """
    if isinstance(value, str):
        return _strip_key_prefix(value).strip().lower() not in (
            "", "false", "0", "no", "off", "none",
        )
    return bool(value)


class ParametricSelfFeature(Feature):
    """The agent's owned parametric self.

    A per-agent local model nightly-finetuned on the agent's own experience and
    (once proven) consulted in the reasoning loop. Training is OFF by default;
    enablement is per-agent (the multi_agent.toml allowlist gates loading; this
    flag gates training within an agent that has it).
    """

    @property
    def tool_description(self) -> str:
        return (
            "Owned parametric self — a per-agent local model nightly-finetuned "
            "on the agent's own experience, consulted as a disposition prior and "
            "on-demand oracle alongside the frontier model"
        )

    def _ensure_training_lifecycle_state(self) -> None:
        """Backfill lifecycle state for direct/unit-level calls before initialize."""
        if not hasattr(self, "_manual_run_lock"):
            self._manual_run_lock = asyncio.Lock()
            self._manual_run_generation = 0
            self._manual_runs_enabled = True
            self._training_shutdown_incomplete = None
            self._cycle_task = None

    async def initialize(self) -> None:
        """Initialize the parametric-self feature (training off by default)."""
        self._adapter = LocalMLXAdapter()           # lazy MLX; inert off Apple Silicon
        self._gate = FidelityGate()                 # held-out promotion check
        self._training_enabled = False              # per-agent gate; default OFF
        self._base_config = TextLoRAConfig()        # base hyperparameters
        self._active_adapter_path: Optional[str] = None   # currently served adapter
        self._last_val_loss: Optional[float] = None        # served adapter's fidelity
        # The feature never infers a permissive training policy.  An operator or
        # host must supply this explicit governed-corpus release policy.
        self._governed_corpus_policy = None
        self._governed_inference_profile = None
        # A live snapshot permits incremental tombstone checks during this
        # process.  It is intentionally not treated as restart-durable proof.
        self._live_corpus_snapshot = None
        # Durable metadata is content-free: hashes, checkpoint pins, and exact
        # assertion/revision lineage only.  Adapter state lives separately from
        # the immutable candidate-side manifest.
        self._adapter_lineage: Dict[str, Dict[str, Any]] = {}
        self._quarantined_adapters: Dict[str, str] = {}
        # Optional overrides (tests / non-standard layouts); else resolved from agent.
        self._db_path: Optional[str] = None
        self._work_dir: Optional[str] = None
        self._sleep_hook = None                     # set in post_all_features_loaded
        # In-flight manual training run (train_now). Detached so the tool call
        # returns immediately; guarded so only one cycle runs at a time.
        self._training_task: Optional[asyncio.Task] = None
        # A sleep-triggered cycle is not detached like ``_training_task`` but it
        # can still be live while disable runs. Track it so teardown can cancel
        # every trigger, not only the manual command path.
        self._cycle_task: Optional[asyncio.Task] = None
        # Serializes the short reservation -> run-record -> task-publication
        # transition.  ``on_disable`` advances the generation before waiting on
        # this lock, so a command that was suspended while creating its durable
        # record can never publish a trainer after disable has begun.
        self._manual_run_lock = asyncio.Lock()
        self._manual_run_generation = 0
        self._manual_runs_enabled = True
        # Set only when cancellation cannot confirm the spawned child stopped.
        # It deliberately blocks new mutations until a clean lifecycle boundary
        # (normally process restart) rather than pretending a live child is gone.
        self._training_shutdown_incomplete: Optional[str] = None
        # The currently-running cycle's durable record (run_id, adapter_id,
        # trigger, started_at, state, adapter_path), or None when idle. Set when
        # a cycle begins so the introspection tools can distinguish "no run",
        # "run in progress", and "run completed/skipped/failed" instead of
        # treating an intermediate Val-loss snapshot as terminal (issue #17).
        self._active_run: Optional[Dict[str, Any]] = None
        # Cross-trigger serialization: nightly (on_post_consolidation) and manual
        # (train_now) cycles share the same corpus/work dir and the served-adapter
        # pointer, so only ONE may run at a time. Set/cleared synchronously inside
        # _run_training_cycle (no await between check and set → atomic in asyncio).
        self._cycle_in_flight = False

    def _get_tool_by_name(self, name: str):
        """Return this feature's tool by skill id (None if absent).

        The A2A command path prefers ``tool.parse_command_args`` when the handler
        exposes ``_get_tool_by_name`` — that parser is prefix-word-count aware and
        type-coerces positional args (so ``!parametric-self-enable false`` binds to
        ``enabled=False``). Without this method the host falls back to a naive
        positional splitter that emits an ``arg0`` kwarg our methods reject, so
        every command with an argument would fail. Provide it to opt into the
        richer parser (mirrors the core Feature base).
        """
        for t in self.get_tools():
            if t.name == name:
                return t
        return None

    async def get_config(self) -> Dict[str, Any]:
        return {
            "enable_nightly_training": self._training_enabled,
            "base_model": self._base_config.base_model,
            "active_adapter_path": self._active_adapter_path,
            "governed_corpus_policy": self._policy_mapping(),
        }

    async def set_config(self, config: Dict[str, Any]) -> None:
        prior_policy_digest = getattr(self._governed_corpus_policy, "digest", None)
        if "enable_nightly_training" in config:
            self._training_enabled = bool(config["enable_nightly_training"])
        if config.get("base_model"):
            self._base_config.base_model = str(config["base_model"])
        if "governed_corpus_policy" in config:
            self._governed_corpus_policy = self._coerce_policy(
                config["governed_corpus_policy"]
            )
        if "governed_inference_profile" in config:
            self._governed_inference_profile = config["governed_inference_profile"]
        # A policy change changes the set of facts that were authorized for an
        # adapter.  Do not leave a previously-trained adapter served until the
        # next status/train call happens to notice it.
        if (
            "governed_corpus_policy" in config
            and self._active_adapter_path
            and prior_policy_digest != getattr(self._governed_corpus_policy, "digest", None)
        ):
            await self._quarantine_adapter(
                self._active_adapter_path,
                "governed corpus policy updated; rebuild required",
            )
        # Persist so the enablement survives restarts (durable per-agent gate).
        await self._persist_config()

    def _policy_mapping(self) -> Optional[Dict[str, Any]]:
        policy = self._governed_corpus_policy
        serializer = getattr(policy, "to_mapping", None)
        if callable(serializer):
            try:
                value = serializer()
                return value if isinstance(value, dict) else None
            except Exception:
                return None
        return None

    @staticmethod
    def _coerce_policy(value: Any):
        """Accept only the public policy value or its canonical mapping.

        Imports are deliberately runtime-only: non-MLX Linux hosts can still
        install/import this feature, and an older host is reported as a visible
        unavailable capability rather than failing package import.
        """
        if value is None or hasattr(value, "digest"):
            return value
        if not isinstance(value, dict):
            raise ValueError("governed_corpus_policy must be a public policy or mapping")
        from kestrel_sovereign.knowledge import GovernedCorpusPolicy, OntologyRef

        pins = tuple(OntologyRef.from_mapping(item) for item in value["accepted_ontology_pins"])
        capabilities = tuple(
            (str(item["name"]), str(item["version"]))
            for item in value["accepted_semantic_capability_versions"]
        )
        return GovernedCorpusPolicy(
            policy_id=value["policy_id"], policy_version=value["policy_version"],
            accepted_epistemic_states=tuple(value["accepted_epistemic_states"]),
            accepted_visibility=tuple(value["accepted_visibility"]),
            accepted_privacy_classifications=tuple(value["accepted_privacy_classifications"]),
            accepted_consent_references=tuple(value["accepted_consent_references"]),
            accepted_grounding_classes=tuple(value["accepted_grounding_classes"]),
            accepted_source_kinds=tuple(value["accepted_source_kinds"]),
            accepted_ontology_pins=pins,
            accepted_semantic_capability_versions=capabilities,
            allow_inferred=bool(value.get("allow_inferred", False)),
            accepted_derivation_profiles=tuple(value.get("accepted_derivation_profiles", ())),
        )

    # ------------------------------------------------------------------
    # Per-agent config persistence (graph node, mirrors the sovereign base)
    # ------------------------------------------------------------------

    def _config_node_id(self) -> str:
        return f"feature_config:{self.name}"

    async def _persist_config(self) -> None:
        """Save the durable state (enable flag, base model, served adapter) to storage."""
        storage = getattr(self.agent, "storage", None)
        if storage is None:
            logger.debug("No storage to persist parametric-self config")
            return
        durable = {
            "enable_nightly_training": self._training_enabled,
            "base_model": self._base_config.base_model,
            # Served-adapter state must persist too: the status tool needs the
            # pointer after restart, and the fidelity gate needs prior_val_loss
            # as its anti-regression baseline (it's lost across restarts otherwise).
            "active_adapter_path": self._active_adapter_path,
            "last_val_loss": self._last_val_loss,
            "governed_corpus_policy": self._policy_mapping(),
            "adapter_lineage": self._adapter_lineage,
            "quarantined_adapters": self._quarantined_adapters,
        }
        try:
            from kestrel_sovereign.storage.async_graph_store import GraphNode
            await storage.add_node(GraphNode(
                node_id=self._config_node_id(),
                node_type="feature_config",
                label=f"{self.name} config",
                properties={"config": durable},
            ))
        except Exception as e:  # never let a persistence hiccup break the feature
            logger.warning("Failed to persist parametric-self config: %s", e)

    async def _restore_persisted_config(self) -> None:
        """Re-apply a previously persisted config on load (restart-durable enable)."""
        storage = getattr(self.agent, "storage", None)
        if storage is None:
            return
        # One guard around load + parse + apply: a malformed persisted config
        # (bad JSON after a manual edit / sync conflict) must be ignored, never
        # raise — post_all_features_loaded runs in the agent init loop, which
        # doesn't isolate per-hook exceptions, so a raise here aborts startup.
        try:
            node = await storage.get_node(self._config_node_id())
            if node is None:
                return
            cfg = node.properties.get("config")
            if isinstance(cfg, str):
                cfg = json.loads(cfg)
            if not isinstance(cfg, dict):
                return
            self._training_enabled = bool(cfg.get("enable_nightly_training", self._training_enabled))
            if cfg.get("base_model"):
                self._base_config.base_model = str(cfg["base_model"])
            # Restore served-adapter state so status + the regression gate
            # survive a restart.
            if cfg.get("active_adapter_path") is not None:
                self._active_adapter_path = str(cfg["active_adapter_path"])
            if cfg.get("last_val_loss") is not None:
                self._last_val_loss = float(cfg["last_val_loss"])
            if "governed_corpus_policy" in cfg:
                self._governed_corpus_policy = self._coerce_policy(
                    cfg.get("governed_corpus_policy")
                )
            if isinstance(cfg.get("adapter_lineage"), dict):
                self._adapter_lineage = dict(cfg["adapter_lineage"])
            if isinstance(cfg.get("quarantined_adapters"), dict):
                self._quarantined_adapters = {
                    str(path): str(reason)
                    for path, reason in cfg["quarantined_adapters"].items()
                }
        except Exception as e:
            logger.warning("Failed to restore parametric-self config (ignored): %s", e)

    @tool(
        name="parametric-self-status",
        description="Report parametric-self state: training enabled, trainer availability, served adapter, fidelity",
        category=ToolCategory.SYSTEM,
        # Single-token `!parametric-self-<verb>` prefixes (no spaces): the command
        # dispatcher routes these by exact match on the first token, so no prefix
        # shadows another and no-arg tools receive no stray positional. A
        # multi-token prefix would force the buggy first-startswith fallback AND
        # turn the verb into an `arg0` kwarg the no-arg method rejects. Verbs are
        # mutually non-nesting so even the startswith fallback stays unambiguous.
        command_prefix="!parametric-self-status",
    )
    async def parametric_self_status(self) -> ToolResult:
        """Report current state."""
        if self._active_adapter_path:
            await self._verify_adapter_lineage(self._active_adapter_path)
        # Surface the recovery state: a completed run can leave a valid candidate
        # on disk with no served pointer; expose those so the operator knows they
        # are adoptable via `!parametric-self-adopt` rather than appearing lost.
        recoverable = (
            [a["adapter_id"] for a in self._scan_candidates() if a["recoverable"]]
            if self._active_adapter_path is None
            else []
        )
        active_run = self._active_run_progress()
        data = {
            "training_enabled": self._training_enabled,
            "trainer_available": self._adapter.is_available(),
            "base_model": self._base_config.base_model,
            "served_adapter": self._active_adapter_path,
            "served_val_loss": self._last_val_loss,
            "recoverable_adapters": recoverable,
            "active_run": active_run,
            "training_shutdown_incomplete": self._training_shutdown_incomplete,
        }
        confirmation = (
            "Parametric-self "
            + ("ENABLED" if self._training_enabled else "disabled")
            + f"; trainer {'available' if data['trainer_available'] else 'unavailable on this host'}."
        )
        if active_run:
            confirmation += (
                f" A {active_run['trigger']} training run is in progress"
                f" (run {active_run['run_id']}, iter {active_run.get('last_seen_iter')},"
                f" latest val_loss {active_run.get('latest_val_loss')})."
            )
        if self._training_shutdown_incomplete:
            confirmation += f" WARNING: {self._training_shutdown_incomplete}."
        if recoverable:
            confirmation += (
                f" No adapter served; {len(recoverable)} valid candidate(s) recoverable via "
                "`!parametric-self-adopt`."
            )
        return ToolResult.ok(confirmation=confirmation, data=data)

    def _active_run_progress(self) -> Optional[Dict[str, Any]]:
        """The in-flight run record enriched with live progress, or None if idle.

        Reads the candidate's ``train.log`` to attach ``last_seen_iter`` and
        ``latest_val_loss`` so an operator monitoring a long detached run sees it
        advancing (iter 60 -> 230 -> 400) rather than mistaking an intermediate
        validation-loss snapshot for the final, terminal result (issue #17).
        """
        run = self._active_run
        if not run:
            return None
        progress = dict(run)
        last_seen_iter: Optional[int] = None
        latest_val_loss: Optional[float] = None
        adapter_path = run.get("adapter_path")
        if adapter_path:
            log = Path(adapter_path) / "train.log"
            if log.is_file():
                try:
                    text = log.read_text()
                    last_seen_iter = parse_latest_iter(text)
                    latest_val_loss = parse_final_val_loss(text)
                except Exception:
                    last_seen_iter = None
                    latest_val_loss = None
        progress["last_seen_iter"] = last_seen_iter
        progress["latest_val_loss"] = latest_val_loss
        return progress

    # ------------------------------------------------------------------
    # Incubator-Principle gate for the mutation tools
    # ------------------------------------------------------------------

    def _is_sovereign_class(self) -> bool:
        """True when this agent may self-modify (the Incubator Principle).

        Self-modifying the parametric self — training it, toggling nightly
        training, rolling back the served adapter — is reserved for
        sovereign-class agents. Governed/test instances (``agent.is_test_instance``)
        must not, on ANY path (manual tools or the nightly sleep cycle).
        """
        return not getattr(self.agent, "is_test_instance", False)

    def _hides_persisted_user_content(self) -> bool:
        """True when the agent's privacy mode forbids durably retaining
        user-authored content (ephemeral / temp-storage), so training must not
        bake it into adapter weights (F377).

        Best-effort + defensive: resolves the platform's privacy decision via
        ``kestrel_sovereign.features.storage_access.hides_persisted_user_content``
        (this feature already depends on kestrel-sovereign), degrading to
        ``False`` if that helper is unavailable on the host's sovereign build.
        """
        try:
            from kestrel_sovereign.features.storage_access import (
                hides_persisted_user_content,
            )
        except Exception:  # noqa: BLE001 - older host without the helper
            return False
        try:
            return bool(hides_persisted_user_content(self.agent))
        except Exception:  # noqa: BLE001 - never let a probe break the cycle
            return False

    def _resolved_governed_policy(self):
        for candidate in (
            getattr(self.agent, "parametric_self_governed_corpus_policy", None),
            self._governed_corpus_policy,
        ):
            if isinstance(getattr(candidate, "digest", None), str):
                return candidate
        return None

    def _resolved_inference_profile(self):
        if self._governed_inference_profile is not None:
            return self._governed_inference_profile
        return getattr(self.agent, "semantic_inference_profile", None)

    async def _request_governed_snapshot(self):
        """Read only through the host's policy-gated, checkpointed capability."""
        storage = getattr(self.agent, "storage", None)
        policy = self._resolved_governed_policy()
        reader = getattr(storage, "governed_assertion_corpus_snapshot", None)
        if policy is None:
            return None, "governed corpus policy is not configured"
        if not callable(reader):
            return None, "governed corpus capability unavailable on this host"
        try:
            snapshot = await reader(
                policy=policy,
                inference_profile=self._resolved_inference_profile(),
            )
        except Exception:
            # Corpus failures may carry provider/tenant/source details.  Keep
            # this operator-facing skip deliberately content-free.
            return None, "governed corpus unavailable or semantic maintenance incomplete"
        if not bool(getattr(snapshot, "verified", False)):
            return None, "governed corpus returned unverified snapshot"
        self._live_corpus_snapshot = snapshot
        return snapshot, None

    @staticmethod
    def _manifest_lineage(path: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            raw = json.loads((Path(path) / "corpus_manifest.json").read_text())
            expected = raw.pop("manifest_hash")
            if not isinstance(expected, str):
                return None, "candidate manifest hash missing"
            encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            import hashlib
            actual = "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            if actual != expected:
                return None, "candidate manifest hash mismatch"
            # Keep the verified receipt separate from the on-disk schema.  It
            # is compared with the durable feature-side stamp before this
            # manifest can authorize an existing adapter.
            raw["_verified_manifest_hash"] = expected
            return raw, None
        except Exception:
            return None, "candidate manifest unavailable"

    async def _quarantine_adapter(self, path: str, reason: str) -> None:
        self._quarantined_adapters[path] = reason
        if self._active_adapter_path == path:
            self._active_adapter_path = None
            self._last_val_loss = None
        lineage = self._adapter_lineage.setdefault(path, {})
        lineage["state"] = "invalid"
        lineage["invalidation_reason"] = reason
        await self._persist_config()

    @staticmethod
    def _manifest_assertion_pairs(manifest: Dict[str, Any]) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for example in manifest.get("examples", ()):
            lineage = example.get("lineage", {}) if isinstance(example, dict) else {}
            assertion_id, revision_id = lineage.get("assertion_id"), lineage.get("revision_id")
            if isinstance(assertion_id, str) and isinstance(revision_id, str):
                pairs.add((assertion_id, revision_id))
        return pairs

    @classmethod
    def _manifest_receipt_stamp(cls, manifest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return the immutable, content-free receipt persisted with an adapter.

        The manifest's self-hash detects accidental corruption, but cannot
        distinguish a replaced, correctly rehashed manifest from the one that
        trained the adapter.  The feature graph stores this independent stamp
        and the verifier compares it before accepting the manifest as lineage.
        """
        checkpoint = manifest.get("semantic_checkpoint")
        capabilities = manifest.get("capability_versions")
        manifest_hash = manifest.get("_verified_manifest_hash")
        if (
            not isinstance(manifest_hash, str)
            or not isinstance(manifest.get("snapshot_hash"), str)
            or not isinstance(manifest.get("policy_digest"), str)
            or not isinstance(checkpoint, Mapping)
            or not isinstance(capabilities, Mapping)
        ):
            return None
        checkpoint_signature = cls._manifest_checkpoint_signature(manifest)
        if checkpoint_signature is None or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in capabilities.items()
        ):
            return None
        return {
            "manifest_hash": manifest_hash,
            "snapshot_hash": manifest["snapshot_hash"],
            "policy_digest": manifest["policy_digest"],
            "semantic_checkpoint": {
                "tenant_id": checkpoint_signature[0],
                "generation": checkpoint_signature[1],
                "event_id": checkpoint_signature[2],
            },
            "capability_versions": dict(sorted(capabilities.items())),
        }

    def _persisted_receipt_problem(self, path: str, manifest: Dict[str, Any]) -> Optional[str]:
        """Reject a valid-looking manifest that is not the adapter's receipt."""
        persisted = self._adapter_lineage.get(path)
        # A manifest validates only itself.  Serving an untracked candidate
        # would let arbitrary old weights be paired with a freshly valid
        # manifest, so legacy/untracked artifacts are inspection-only until a
        # new governed training run creates their durable receipt.
        if not isinstance(persisted, dict):
            return "durable adapter lineage receipt unavailable; rebuild required"
        receipt = self._manifest_receipt_stamp(manifest)
        if receipt is None:
            return "candidate manifest governed evidence is malformed"
        for field, expected in receipt.items():
            if persisted.get(field) != expected:
                return "persisted adapter lineage receipt mismatch; rebuild required"
        return None

    @staticmethod
    def _checkpoint_signature(checkpoint: Any) -> Optional[tuple[str, int, Optional[str]]]:
        """Return the public checkpoint identity in the manifest's shape."""
        tenant_id = getattr(checkpoint, "tenant_id", None)
        generation = getattr(checkpoint, "generation", None)
        event_id = getattr(checkpoint, "latest_event_id", getattr(checkpoint, "event_id", None))
        if (
            not isinstance(tenant_id, str)
            or not tenant_id
            or not isinstance(generation, int)
            or not isinstance(event_id, (str, type(None)))
        ):
            return None
        return tenant_id, generation, event_id

    @staticmethod
    def _manifest_checkpoint_signature(manifest: Dict[str, Any]) -> Optional[tuple[str, int, Optional[str]]]:
        checkpoint = manifest.get("semantic_checkpoint")
        if not isinstance(checkpoint, dict):
            return None
        tenant_id = checkpoint.get("tenant_id")
        generation, event_id = checkpoint.get("generation"), checkpoint.get("event_id")
        if (
            not isinstance(tenant_id, str)
            or not tenant_id
            or not isinstance(generation, int)
            or not isinstance(event_id, (str, type(None)))
        ):
            return None
        return tenant_id, generation, event_id

    def _snapshot_pin_problem(
        self,
        manifest: Dict[str, Any],
        snapshot: Any,
        *,
        require_exact_snapshot: bool,
    ) -> Optional[str]:
        """Validate the complete receipt that authorized this adapter.

        An assertion/revision pair is not enough: the same pair can be exposed
        under a different policy, ontology/capability pin, or semantic state.
        The manifest is a receipt for one immutable snapshot.  A live snapshot
        must match it byte-for-byte at the public-contract level; a restart
        snapshot may be newer, but must retain the same policy/capability pins.
        """
        expected_policy = manifest.get("policy_digest")
        expected_hash = manifest.get("snapshot_hash")
        expected_capabilities = manifest.get("capability_versions")
        expected_checkpoint = self._manifest_checkpoint_signature(manifest)
        if (
            not isinstance(expected_policy, str)
            or not isinstance(expected_hash, str)
            or not isinstance(expected_capabilities, Mapping)
            or not all(isinstance(key, str) and isinstance(value, str)
                       for key, value in expected_capabilities.items())
            or expected_checkpoint is None
        ):
            return "candidate manifest governed evidence is malformed"

        policy = self._resolved_governed_policy()
        if getattr(policy, "digest", None) != expected_policy:
            return "governed corpus policy changed; rebuild required"
        if getattr(getattr(snapshot, "policy", None), "digest", None) != expected_policy:
            return "governed corpus policy evidence mismatch; rebuild required"
        capabilities = getattr(snapshot, "capability_versions", None)
        if not isinstance(capabilities, Mapping) or dict(capabilities) != dict(expected_capabilities):
            return "governed semantic capability pins changed; rebuild required"

        checkpoint = self._checkpoint_signature(getattr(snapshot, "checkpoint", None))
        snapshot_hash = getattr(snapshot, "snapshot_hash", None)
        if (
            checkpoint is None
            or getattr(snapshot, "tenant_id", None) != expected_checkpoint[0]
            or not isinstance(snapshot_hash, str)
        ):
            return "governed corpus snapshot evidence is malformed"
        if require_exact_snapshot:
            if checkpoint != expected_checkpoint or snapshot_hash != expected_hash:
                return "governed corpus snapshot receipt changed; rebuild required"
        # If a fresh read reports the exact same checkpoint, its receipt hash
        # must be identical.  A later checkpoint is allowed and its assertion
        # membership is checked below; it cannot silently change policy/pins.
        elif checkpoint == expected_checkpoint and snapshot_hash != expected_hash:
            return "governed corpus snapshot receipt changed; rebuild required"
        return None

    def _delta_pin_problem(self, manifest: Dict[str, Any], delta: Any) -> Optional[str]:
        """Check that delta evidence is rooted at the manifest's exact snapshot."""
        expected_checkpoint = self._manifest_checkpoint_signature(manifest)
        since_checkpoint = self._checkpoint_signature(getattr(delta, "since_checkpoint", None))
        checkpoint = self._checkpoint_signature(getattr(delta, "checkpoint", None))
        observability = getattr(delta, "observability", None)
        expected_policy = manifest.get("policy_digest")
        if (
            expected_checkpoint is None
            or since_checkpoint != expected_checkpoint
            or checkpoint is None
            or checkpoint[0] != expected_checkpoint[0]
            or not isinstance(getattr(delta, "snapshot_hash", None), str)
            or getattr(observability, "policy_digest", None) != expected_policy
        ):
            return "governed corpus delta evidence mismatch; adapter cannot be verified"
        return None

    async def _verify_adapter_lineage(self, path: str, *, before_promotion: bool = False) -> Optional[str]:
        """Quarantine an adapter when its exact governed inputs no longer hold."""
        manifest, manifest_error = self._manifest_lineage(path)
        if manifest_error:
            await self._quarantine_adapter(path, manifest_error)
            return manifest_error
        persisted_problem = self._persisted_receipt_problem(path, manifest or {})
        if persisted_problem:
            await self._quarantine_adapter(path, persisted_problem)
            return persisted_problem
        pairs = self._manifest_assertion_pairs(manifest or {})
        snapshot = self._live_corpus_snapshot
        storage = getattr(self.agent, "storage", None)
        policy = self._resolved_governed_policy()
        changes = getattr(storage, "governed_assertion_corpus_changes_since", None)
        if snapshot is not None and callable(changes) and policy is not None:
            pin_problem = self._snapshot_pin_problem(
                manifest or {}, snapshot, require_exact_snapshot=True,
            )
            if pin_problem:
                await self._quarantine_adapter(path, pin_problem)
                return pin_problem
            try:
                delta = await changes(
                    snapshot, policy=policy,
                    inference_profile=self._resolved_inference_profile(),
                )
            except Exception:
                reason = "governed corpus delta unavailable; adapter cannot be verified"
                await self._quarantine_adapter(path, reason)
                return reason
            pin_problem = self._delta_pin_problem(manifest or {}, delta)
            if pin_problem:
                await self._quarantine_adapter(path, pin_problem)
                return pin_problem
            tombstoned = {
                (item.assertion_id, item.revision_id)
                for item in getattr(delta, "tombstones", ())
                if isinstance(getattr(item, "assertion_id", None), str)
                and isinstance(getattr(item, "revision_id", None), str)
            }
            removed_ids = {
                item.assertion_id for item in getattr(delta, "tombstones", ())
                if isinstance(getattr(item, "assertion_id", None), str)
            }
            if tombstoned.intersection(pairs) or any(aid in removed_ids for aid, _ in pairs):
                reason = "governed assertion lineage invalidated; rebuild required"
                await self._quarantine_adapter(path, reason)
                return reason
            self._live_corpus_snapshot = None  # a delta is evidence only for its base snapshot
            return None

        # A process restart cannot reuse an in-memory snapshot as durable proof.
        # Rebuild a fresh approved snapshot and compare exact revisions.
        fresh, reason = await self._request_governed_snapshot()
        if fresh is None:
            await self._quarantine_adapter(path, reason or "governed corpus unavailable")
            return reason
        pin_problem = self._snapshot_pin_problem(
            manifest or {}, fresh, require_exact_snapshot=False,
        )
        if pin_problem:
            await self._quarantine_adapter(path, pin_problem)
            return pin_problem
        current = {
            (item.assertion.assertion_id, item.assertion.revision_id)
            for item in getattr(fresh, "examples", ())
        }
        if not pairs.issubset(current):
            reason = "governed assertion lineage no longer current; rebuild required"
            await self._quarantine_adapter(path, reason)
            return reason
        if before_promotion:
            # The fresh snapshot is now the correct base for a subsequent delta.
            self._live_corpus_snapshot = fresh
        else:
            # ``_request_governed_snapshot`` caches its result for a newly
            # built candidate. A restart/status verification may legitimately
            # observe a newer checkpoint, which is evidence for membership but
            # not the adapter's original delta base; never reuse it as one.
            self._live_corpus_snapshot = None
        return None

    def _require_sovereign_class(self) -> Optional[ToolResult]:
        """Return a refusal ``ToolResult`` for a governed agent, else ``None``.

        Note on the *caller* gate: ``CallerContext.is_sovereign`` is threaded
        only to the command handler for a fixed set of core governance commands
        — it is not passed to feature ``@tool`` methods (they dispatch via the
        A2A TaskManager, which carries no caller context). So caller-level
        gating is not available here without a core change; the agent-class gate
        is the meaningful self-modification boundary for this feature.
        """
        if not self._is_sovereign_class():
            return ToolResult.failed(
                "Refused: managing the parametric self is self-modification, which "
                "the Incubator Principle reserves for sovereign-class agents. This "
                "agent is a governed/test instance (is_test_instance=True) and may "
                "not train, toggle, or roll back its own parametric self."
            )
        return None

    def _candidates_dir(self) -> Optional[Path]:
        """The directory holding staged candidate adapters, or None if unresolved."""
        _, work_dir = self._resolve_paths()
        if not work_dir:
            return None
        return Path(work_dir) / "candidates"

    def _scan_candidates(self) -> List[Dict[str, Any]]:
        """Scan the candidates dir for staged adapters with parsed val_loss.

        Each entry carries ``served`` (is this the currently-served adapter),
        ``in_progress`` (this candidate belongs to the run currently training),
        and ``recoverable`` (a valid candidate — parseable val_loss — while NO
        adapter is served yet; the lifecycle state where a *completed* run left a
        candidate on disk but no served pointer, adoptable via
        ``parametric-self-adopt``). An in-progress candidate is never recoverable:
        its parsed ``val_loss`` is an intermediate snapshot, not a terminal
        result, so it must not be presented as adoptable (issue #17).
        """
        candidates = self._candidates_dir()
        served = self._active_adapter_path
        active_id = self._active_run.get("adapter_id") if self._active_run else None
        adapters: List[Dict[str, Any]] = []
        if candidates and candidates.is_dir():
            for d in sorted(candidates.iterdir()):
                if not d.is_dir():
                    continue
                val_loss = None
                log = d / "train.log"
                if log.is_file():
                    try:
                        val_loss = parse_final_val_loss(log.read_text())
                    except Exception:
                        val_loss = None
                in_progress = active_id is not None and d.name == active_id
                manifest, manifest_problem = self._manifest_lineage(str(d))
                receipt_problem = (
                    self._persisted_receipt_problem(str(d), manifest)
                    if manifest is not None else None
                )
                lineage_state = self._adapter_lineage.get(str(d), {}).get(
                    "state", "candidate" if manifest is not None else "untracked"
                )
                quarantined_reason = self._quarantined_adapters.get(str(d))
                adapters.append({
                    "adapter_id": d.name,
                    "path": str(d),
                    "val_loss": val_loss,
                    "served": str(d) == str(served) if served else False,
                    "in_progress": in_progress,
                    "recoverable": (
                        served is None and val_loss is not None and not in_progress
                        and manifest is not None and not quarantined_reason
                        and receipt_problem is None
                    ),
                    "lineage_state": lineage_state,
                    "quarantined_reason": quarantined_reason or manifest_problem or receipt_problem,
                })
        return adapters

    # ------------------------------------------------------------------
    # View tools (ungated) — introspect the parametric-self lifecycle
    # ------------------------------------------------------------------

    @tool(
        name="parametric-self-history",
        description="List recent parametric-self training runs (timestamp, trigger, corpus size, val_loss, promoted, reason)",
        category=ToolCategory.SYSTEM,
        command_prefix="!parametric-self-history",
    )
    async def parametric_self_history(self) -> ToolResult:
        """Report the recorded training-run history, most recent first."""
        runs = await self._load_run_history()
        recent = list(reversed(runs))  # most recent first for display
        if not recent:
            return ToolResult.ok(
                confirmation="No parametric-self training runs recorded yet.",
                data={"runs": []},
            )
        promoted = sum(1 for r in recent if r.get("promoted"))
        in_progress = sum(1 for r in recent if r.get("state") == "in_progress")
        header = f"{len(recent)} training run(s) recorded ({promoted} promoted"
        header += f", {in_progress} in progress):" if in_progress else "):"
        lines = [header]
        for r in recent[:10]:
            # Older entries predate the state field; default to a terminal label.
            state = r.get("state", "completed")
            lines.append(
                f"  {r.get('timestamp', '?')} [{r.get('trigger', '?')}] "
                f"state={state} trained={r.get('trained')} promoted={r.get('promoted')} "
                f"val_loss={r.get('val_loss')} corpus={r.get('corpus_train')} "
                f"— {r.get('reason', '')}".rstrip()
            )
        return ToolResult.ok(confirmation="\n".join(lines), data={"runs": recent})

    @tool(
        name="parametric-self-adapters",
        description="List candidate parametric-self adapters on disk with their val_loss, marking which is currently served",
        category=ToolCategory.SYSTEM,
        command_prefix="!parametric-self-adapters",
    )
    async def parametric_self_adapters(self) -> ToolResult:
        """List staged candidate adapters, their val_loss, and the served one."""
        served = self._active_adapter_path
        adapters = self._scan_candidates()
        recoverable = [a["adapter_id"] for a in adapters if a["recoverable"]]
        if not adapters:
            return ToolResult.ok(
                confirmation="No candidate adapters on disk yet.",
                data={"adapters": [], "served_adapter": served, "recoverable_adapters": []},
            )
        lines = [f"{len(adapters)} candidate adapter(s):"]
        for a in adapters:
            if a["served"]:
                mark = " (served)"
            elif a.get("in_progress"):
                mark = " (in progress — training, val_loss is intermediate)"
            elif a["recoverable"]:
                mark = " (recoverable — adopt with !parametric-self-adopt)"
            else:
                mark = ""
            lines.append(f"  {a['adapter_id']}  val_loss={a['val_loss']}{mark}")
        if recoverable:
            lines.append(
                f"No adapter is served; {len(recoverable)} valid candidate(s) can be "
                "adopted with `!parametric-self-adopt adapter_id=<id>`."
            )
        return ToolResult.ok(
            confirmation="\n".join(lines),
            data={
                "adapters": adapters,
                "served_adapter": served,
                "recoverable_adapters": recoverable,
            },
        )

    @tool(
        name="parametric-self-progress",
        description="Report the parametric-self training run currently in progress (run_id, trigger, state, last_seen_iter, latest_val_loss), or that none is active",
        category=ToolCategory.SYSTEM,
        command_prefix="!parametric-self-progress",
    )
    async def parametric_self_progress(self) -> ToolResult:
        """Report the in-flight run so a detached run is not mistaken for done.

        Distinguishes "no run exists" from "a run is still active": the run
        history is completion-only for terminal outcomes, so this is the
        first-class surface for an in-progress run and its live ``train.log``
        progress (issue #17).
        """
        active_run = self._active_run_progress()
        if not active_run:
            return ToolResult.ok(
                confirmation="No parametric-self training run is in progress.",
                data={"active_run": None},
            )
        return ToolResult.ok(
            confirmation=(
                f"A {active_run['trigger']} training run is in progress "
                f"(run {active_run['run_id']}, adapter {active_run.get('adapter_id')}, "
                f"started {active_run.get('started_at')}): iter {active_run.get('last_seen_iter')}, "
                f"latest val_loss {active_run.get('latest_val_loss')} (intermediate, not final)."
            ),
            data={"active_run": active_run},
        )

    # ------------------------------------------------------------------
    # Management tools (sovereign-class gated) — govern the parametric self
    # ------------------------------------------------------------------

    @tool(
        name="parametric-self-train-now",
        description="Trigger a parametric-self training run immediately (sovereign-class only); returns once the run has started, not when it finishes",
        category=ToolCategory.SYSTEM,
        command_prefix="!parametric-self-train",
    )
    async def parametric_self_train_now(self) -> ToolResult:
        """Kick off a training cycle detached; do not block for the full run."""
        self._ensure_training_lifecycle_state()
        gate = self._require_sovereign_class()
        if gate is not None:
            return gate
        if self._hides_persisted_user_content():
            # Same privacy refusal as the nightly chokepoint — the manual tool
            # detaches straight into ``_run_training_cycle_locked``, bypassing
            # the ``_run_training_cycle`` gate, so it must be checked here too or
            # ``!parametric-self-train`` would train under an ephemeral/temp
            # privacy mode (F377).
            return ToolResult.failed(
                "Training refused: the agent's privacy mode hides persisted "
                "user content, which would be baked into adapter weights."
            )
        if not self._adapter.is_available():
            return ToolResult.failed("Trainer unavailable on this host (MLX/Apple Silicon required).")
        db_path, work_dir = self._resolve_paths()
        if not work_dir:
            return ToolResult.failed("Could not resolve parametric-self work directory.")

        # Keep the reservation, durable record creation, and task publication
        # together.  ``on_disable`` invalidates the captured generation before
        # it waits for this lock, closing the otherwise possible race where a
        # command resumes from the record-store await and launches after disable.
        async with self._manual_run_lock:
            if not self._manual_runs_enabled:
                return ToolResult.failed("Parametric-self training is unavailable while this feature is disabled.")
            if self._training_shutdown_incomplete:
                return ToolResult.failed(
                    "Parametric-self training is blocked: "
                    f"{self._training_shutdown_incomplete}."
                )
            if self._cycle_in_flight or (
                self._training_task is not None and not self._training_task.done()
            ):
                return ToolResult.failed("A parametric-self training run is already in progress.")

            launch_generation = self._manual_run_generation
            # Reserve the cross-trigger guard HERE, synchronously, before
            # detaching: otherwise the nightly hook could fire in the same
            # event-loop turn and acquire it first.
            self._cycle_in_flight = True
            active_run: Optional[Dict[str, Any]] = None
            try:
                active_run = await self._begin_active_run(trigger="manual", work_dir=work_dir)
            except asyncio.CancelledError:
                await self._interrupt_active_run(
                    reason="run cancelled before manual training started",
                )
                self._cycle_in_flight = False
                raise

            if (
                not self._manual_runs_enabled
                or launch_generation != self._manual_run_generation
            ):
                await self._interrupt_active_run(
                    run_id=active_run["run_id"],
                    reason="run cancelled (feature disabled before launch)",
                )
                self._cycle_in_flight = False
                return ToolResult.failed(
                    "Parametric-self training did not start because the feature was disabled."
                )

            # Run detached: a full cycle is ~24 min; the tool returns immediately
            # and the run record already exists by the time started=True is
            # returned. The runner calls the LOCKED body (the guard is already
            # held) and clears it in finally. Errors are logged, not surfaced
            # (poll history/progress for the outcome).
            async def _runner() -> None:
                keep_guard_held = False
                try:
                    outcome = await self._run_training_cycle_locked(trigger="manual")
                    # The locked body owns normal full-cycle finalization.  It can
                    # also return before it creates a run (for example, when the
                    # governed corpus policy is absent or semantic maintenance has
                    # not produced a usable snapshot).  A detached manual cycle
                    # already has a durable in-progress record at this point, so
                    # finish that specific record for every such normal no-op.
                    # Matching the captured run id prevents a future lifecycle
                    # change from accidentally finalizing another run.
                    active = getattr(self, "_active_run", None)
                    if active is not None and active.get("run_id") == active_run["run_id"]:
                        trained = bool(outcome.get("trained", False))
                        await self._update_run_history(active_run["run_id"], {
                            "state": "completed" if trained else "skipped",
                            "timestamp": _utc_now_iso(),
                            "trained": trained,
                            "promoted": bool(outcome.get("promoted", False)),
                            "reason": outcome.get("reason", "training completed"),
                        })
                        self._active_run = None
                except asyncio.CancelledError as exc:
                    # A normal cycle cancellation confirms the child before
                    # this handler finalizes its run. An explicit incomplete
                    # shutdown instead keeps an observable nonterminal record.
                    incomplete = isinstance(exc, TrainingShutdownIncomplete)
                    if incomplete:
                        await self._mark_training_shutdown_incomplete(
                            run_id=active_run["run_id"], reason=str(exc),
                        )
                        keep_guard_held = True
                    else:
                        await self._interrupt_active_run(
                            run_id=active_run["run_id"], reason="run cancelled",
                        )
                    if self._training_task is asyncio.current_task():
                        self._training_task = None
                    raise
                except Exception as exc:
                    active = getattr(self, "_active_run", None)
                    if active is not None and active.get("run_id") == active_run["run_id"]:
                        await self._update_run_history(active_run["run_id"], {
                            "state": "failed",
                            "timestamp": _utc_now_iso(),
                            "reason": f"training error: {exc}",
                        })
                        self._active_run = None
                    logger.warning("parametric-self manual training run failed: %s", exc)
                finally:
                    if not keep_guard_held:
                        self._cycle_in_flight = False

            self._training_task = asyncio.create_task(_runner())
        return ToolResult.ok(
            confirmation=(
                "Parametric-self training run started in the background. "
                "Poll `!parametric-self-progress` while it runs (an intermediate "
                "val_loss is NOT the final result), and `!parametric-self-history` "
                "for the outcome."
            ),
            data={"started": True, "active_run": active_run},
        )

    @tool(
        name="parametric-self-set-enabled",
        description="Enable or disable nightly parametric-self training for this agent (sovereign-class only); persists across restarts",
        category=ToolCategory.SYSTEM,
        command_prefix="!parametric-self-enable",
    )
    async def parametric_self_set_enabled(self, enabled: bool) -> ToolResult:
        """Toggle the agent's own nightly training gate (durable)."""
        gate = self._require_sovereign_class()
        if gate is not None:
            return gate
        await self.set_config({"enable_nightly_training": _as_bool(enabled)})
        state = "ENABLED" if self._training_enabled else "disabled"
        return ToolResult.ok(
            confirmation=f"Nightly parametric-self training {state} for this agent.",
            data={"enable_nightly_training": self._training_enabled},
        )

    @tool(
        name="parametric-self-rollback",
        description="Roll the served parametric-self adapter back to a prior candidate (sovereign-class only); default = the previously-served promoted adapter",
        category=ToolCategory.SYSTEM,
        command_prefix="!parametric-self-rollback",
    )
    async def parametric_self_rollback(self, adapter_id: Optional[str] = None) -> ToolResult:
        """Revert the served adapter to a prior candidate and persist the change."""
        gate = self._require_sovereign_class()
        if gate is not None:
            return gate

        candidates = self._candidates_dir()
        if candidates is None:
            return ToolResult.failed("Could not resolve the candidates directory for this agent.")

        # Serialize against training: a concurrent cycle promoting a candidate
        # could overwrite this rollback, or evaluate its fidelity gate against a
        # served-adapter baseline this rollback is changing. Share the in-flight
        # guard so rollback and training never mutate served state at once.
        if self._cycle_in_flight:
            return ToolResult.failed(
                "A parametric-self training run is in progress; cannot roll back until it completes."
            )
        self._cycle_in_flight = True
        try:
            target_path: Optional[str] = None
            adapter_id = _strip_key_prefix(adapter_id) if adapter_id else adapter_id
            if adapter_id:
                # Only accept a simple child name — reject path separators, '..',
                # and absolute paths so a rollback can never serve a directory
                # outside the candidates tree.
                if Path(adapter_id).name != adapter_id or adapter_id in (".", ".."):
                    return ToolResult.failed(f"Invalid adapter id '{adapter_id}' (must be a candidate directory name).")
                target = candidates / adapter_id
                # Defense in depth: confirm the resolved path stays under candidates.
                if not target.is_dir() or candidates.resolve() not in target.resolve().parents:
                    return ToolResult.failed(f"No candidate adapter '{adapter_id}' on disk.")
                target_path = str(target)
            else:
                # Default: the most recent promoted adapter in history that is not
                # the one currently served (i.e. the previously-served adapter).
                runs = await self._load_run_history()
                promoted_paths = [r.get("adapter_path") for r in runs if r.get("promoted") and r.get("adapter_path")]
                prior = [p for p in reversed(promoted_paths) if p != self._active_adapter_path]
                if not prior:
                    return ToolResult.failed(
                        "No prior promoted adapter to roll back to. "
                        "Pass an adapter_id from `!parametric-self-adapters`."
                    )
                target_path = prior[0]
                if not Path(target_path).is_dir():
                    return ToolResult.failed(
                        f"Prior adapter '{target_path}' is no longer on disk. "
                        "Pass an adapter_id from `!parametric-self-adapters`."
                    )

            # Re-read the target's val_loss so the regression baseline tracks the
            # adapter we are now serving. Refuse to serve an adapter without a
            # parseable validation loss: serving it would bypass the fidelity
            # guarantee and leave the anti-regression gate with no baseline.
            val_loss: Optional[float] = None
            log = Path(target_path) / "train.log"
            if log.is_file():
                try:
                    val_loss = parse_final_val_loss(log.read_text())
                except Exception:
                    val_loss = None
            if val_loss is None:
                return ToolResult.failed(
                    f"Adapter '{Path(target_path).name}' has no parseable validation loss "
                    "(incomplete/failed run); refusing to serve it."
                )

            lineage_problem = await self._verify_adapter_lineage(target_path)
            if lineage_problem:
                return ToolResult.failed(
                    f"Adapter '{Path(target_path).name}' is quarantined: {lineage_problem}."
                )

            self._active_adapter_path = target_path
            self._last_val_loss = val_loss
            await self._persist_config()
            await self._append_run_history({
                "timestamp": _utc_now_iso(),
                "trigger": "rollback",
                "trained": False,
                "promoted": False,
                "val_loss": val_loss,
                "reason": f"rolled back served adapter to {Path(target_path).name}",
                "corpus_train": 0,
                "adapter_path": target_path,
            })
            return ToolResult.ok(
                confirmation=f"Served adapter rolled back to {Path(target_path).name} (val_loss={val_loss}).",
                data={"served_adapter": target_path, "served_val_loss": val_loss},
            )
        finally:
            self._cycle_in_flight = False

    @tool(
        name="parametric-self-adopt",
        description="Adopt a candidate parametric-self adapter as the served adapter (sovereign-class only); recovers a valid candidate left on disk with no served pointer",
        category=ToolCategory.SYSTEM,
        command_prefix="!parametric-self-adopt",
    )
    async def parametric_self_adopt(self, adapter_id: str) -> ToolResult:
        """Adopt a candidate adapter as the served one (explicit first adoption).

        This is the discoverable path for the recovery state where a completed
        run left a valid candidate on disk but no served-adapter pointer/history
        (e.g. a legacy/interrupted run). Unlike rollback — which reverts to a
        *previously promoted* adapter — adopt fronts a candidate for the first
        time, evaluating the same fidelity gate against the current baseline and
        recording an ``adopt`` history entry.
        """
        gate = self._require_sovereign_class()
        if gate is not None:
            return gate

        candidates = self._candidates_dir()
        if candidates is None:
            return ToolResult.failed("Could not resolve the candidates directory for this agent.")

        # Share the in-flight guard with training/rollback: adopting mutates the
        # served-adapter pointer, which a concurrent cycle also writes.
        if self._cycle_in_flight:
            return ToolResult.failed(
                "A parametric-self training run is in progress; cannot adopt until it completes."
            )
        self._cycle_in_flight = True
        try:
            adapter_id = _strip_key_prefix(adapter_id) if adapter_id else adapter_id
            if not adapter_id:
                return ToolResult.failed(
                    "An adapter_id is required (see `!parametric-self-adapters`)."
                )
            # Only accept a simple child name — same traversal protections as
            # rollback so adoption can never serve a directory outside candidates.
            if Path(adapter_id).name != adapter_id or adapter_id in (".", ".."):
                return ToolResult.failed(f"Invalid adapter id '{adapter_id}' (must be a candidate directory name).")
            target = candidates / adapter_id
            # Defense in depth: confirm the resolved path stays under candidates.
            if not target.is_dir() or candidates.resolve() not in target.resolve().parents:
                return ToolResult.failed(f"No candidate adapter '{adapter_id}' on disk.")
            target_path = str(target)

            # Require a parseable validation loss: serving an adapter without one
            # bypasses the fidelity guarantee and leaves the gate with no baseline.
            val_loss: Optional[float] = None
            log = target / "train.log"
            if log.is_file():
                try:
                    val_loss = parse_final_val_loss(log.read_text())
                except Exception:
                    val_loss = None
            if val_loss is None:
                return ToolResult.failed(
                    f"Adapter '{adapter_id}' has no parseable validation loss "
                    "(incomplete/failed run); refusing to serve it."
                )

            lineage_problem = await self._verify_adapter_lineage(target_path)
            if lineage_problem:
                return ToolResult.failed(
                    f"Adapter '{adapter_id}' is quarantined: {lineage_problem}."
                )

            # Same fidelity gate as a nightly promotion, against the current
            # baseline (``prior_val_loss`` is None in the first-adoption case).
            decision = self._gate.evaluate(val_loss, prior_val_loss=self._last_val_loss)
            if not decision.promote:
                return ToolResult.failed(
                    f"Adapter '{adapter_id}' fails the fidelity gate: {decision.reason}."
                )

            self._active_adapter_path = target_path
            self._last_val_loss = val_loss
            await self._persist_config()
            await self._append_run_history({
                "timestamp": _utc_now_iso(),
                "trigger": "adopt",
                "trained": False,
                "promoted": True,
                "val_loss": val_loss,
                "reason": f"adopted candidate adapter {adapter_id} as served",
                "corpus_train": 0,
                "adapter_path": target_path,
            })
            return ToolResult.ok(
                confirmation=f"Adopted adapter {adapter_id} as served (val_loss={val_loss}).",
                data={"served_adapter": target_path, "served_val_loss": val_loss},
            )
        finally:
            self._cycle_in_flight = False

    def _resolve_paths(self) -> Tuple[Optional[str], Optional[str]]:
        """Resolve optional reflection DB and required adapter working directory.

        The agent may expose a SQLite ``storage_path`` for the optional
        reflection source; factual training data never comes from that file.
        There is no ``data_dir`` attribute on the agent.
        """
        if self._db_path and self._work_dir:
            return self._db_path, self._work_dir
        configured_work = getattr(self.agent, "parametric_self_work_dir", None)
        explicit_work = self._work_dir or (
            configured_work if isinstance(configured_work, (str, Path)) and str(configured_work) else None
        )
        storage_path = getattr(self.agent, "storage_path", None)
        if not storage_path:
            return self._db_path, str(explicit_work) if explicit_work else None
        db = self._db_path or str(storage_path)
        work = explicit_work or str(Path(storage_path).parent / "parametric_self")
        return db, work

    async def on_post_consolidation(
        self,
        consolidation_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Feature-layer sleep hook: nightly LoRA training behind the fidelity gate.

        No-ops (returns ``trained=False`` with a reason) unless training is
        enabled for this agent and the host can run MLX. On a successful run the
        served-adapter pointer advances only if the gate promotes.
        """
        if not self._training_enabled:
            return {"trained": False, "promoted": False, "reason": "nightly training disabled for this agent"}
        return await self._run_training_cycle(trigger="nightly")

    async def _run_training_cycle(self, *, trigger: str) -> Dict[str, Any]:
        """Run one corpus->train->gate->promote cycle and record it in history.

        Shared by the nightly sleep hook (``on_post_consolidation``, gated on
        ``_training_enabled``) and the explicit ``parametric_self_train_now``
        tool (which bypasses that gate — a sovereign asking for a run is the
        authority). The served-adapter pointer advances only if the gate
        promotes. Every run (including no-op/failed runs that actually trained)
        appends one entry to the run-history store so the agent can introspect
        its own training lifecycle.

        Serialized across triggers: if a cycle is already in flight (nightly or
        manual), this returns a skip rather than racing on the shared corpus dir
        and served-adapter pointer.

        Incubator Principle is enforced HERE, the single chokepoint for both the
        manual tool and the nightly sleep hook: a governed/test instance never
        self-modifies, even if ``enable_nightly_training`` was set in persisted
        or externally-supplied config.
        """
        self._ensure_training_lifecycle_state()
        if not self._is_sovereign_class():
            return {"trained": False, "promoted": False,
                    "reason": "self-modification reserved for sovereign-class agents (Incubator Principle)"}
        if self._hides_persisted_user_content():
            # An ephemeral / temp-storage privacy mode means user-authored
            # content must not be durably retained. Training would bake the
            # night's reflections/facts into the LoRA weights — which cannot be
            # selectively forgotten — so skip the whole cycle (F377).
            return {"trained": False, "promoted": False,
                    "reason": "training skipped: privacy mode hides persisted user content"}
        # All triggers share the same lifecycle fence as manual launch. Disable
        # invalidates the generation before it waits on this lock, so a pending
        # sleep dispatch cannot begin a child after disable starts.
        async with self._manual_run_lock:
            if not self._manual_runs_enabled:
                return {"trained": False, "promoted": False, "reason": "parametric-self feature is disabled"}
            if self._training_shutdown_incomplete:
                return {
                    "trained": False,
                    "promoted": False,
                    "reason": self._training_shutdown_incomplete,
                }
            if self._cycle_in_flight:
                return {"trained": False, "promoted": False, "reason": "another training run already in progress"}
            launch_generation = self._manual_run_generation
            self._cycle_in_flight = True
            self._cycle_task = asyncio.current_task()
        keep_guard_held = False
        try:
            if (
                not self._manual_runs_enabled
                or launch_generation != self._manual_run_generation
            ):
                return {"trained": False, "promoted": False, "reason": "parametric-self feature is disabled"}
            return await self._run_training_cycle_locked(trigger=trigger)
        except TrainingShutdownIncomplete as exc:
            active_run = getattr(self, "_active_run", None)
            await self._mark_training_shutdown_incomplete(
                run_id=active_run.get("run_id") if active_run is not None else None,
                reason=str(exc),
            )
            keep_guard_held = True
            raise
        finally:
            if not keep_guard_held:
                self._cycle_in_flight = False
            if self._cycle_task is asyncio.current_task():
                self._cycle_task = None

    async def _run_training_cycle_locked(self, *, trigger: str) -> Dict[str, Any]:
        """Body of one cycle; only ever called with the in-flight guard held."""
        # An unsupported local trainer is an expected operational no-op. Check
        # it before requesting governed data so a Linux/non-MLX sleep cycle
        # reports SKIPPED rather than a misleading missing-policy failure. On a
        # supported host, corpus/policy evidence remains a hard prerequisite.
        if not self._adapter.is_available():
            return {"trained": False, "promoted": False, "reason": "trainer unavailable on this host"}
        db_path, work_dir = self._resolve_paths()
        if not work_dir:
            return {"trained": False, "promoted": False, "reason": "could not resolve parametric-self work directory"}

        governed_snapshot, corpus_reason = await self._request_governed_snapshot()
        if governed_snapshot is None:
            return {"trained": False, "promoted": False, "reason": corpus_reason}

        agent_id = getattr(self.agent, "agent_id", None) or getattr(self.agent, "name", "agent")
        config = TextLoRAConfig.from_dict(self._base_config.to_dict())

        # Manual ``train_now`` creates this record before returning started=True;
        # nightly/direct cycles create it here. In either path the same durable
        # record is later updated in place to terminal state.
        active_run = self._active_run
        if not (
            active_run
            and active_run.get("state") == "in_progress"
            and active_run.get("trigger") == trigger
        ):
            active_run = await self._begin_active_run(trigger=trigger, work_dir=work_dir)
        run_id = active_run["run_id"]
        adapter_id = active_run["adapter_id"]
        adapter_path = active_run["adapter_path"]

        try:
            result = await run_nightly_cycle(
                agent_id=str(agent_id),
                db_path=db_path,
                work_dir=work_dir,
                governed_snapshot=governed_snapshot,
                adapter=self._adapter,
                gate=self._gate,
                config=config,
                prior_val_loss=self._last_val_loss,
                adapter_id=adapter_id,
            )
        except asyncio.CancelledError:
            # Cancellation (e.g. on_disable) marks the record interrupted; that
            # durable update is done by on_disable, not here, because awaiting
            # storage during cancellation re-raises immediately.
            raise
        except Exception as exc:
            await self._update_run_history(run_id, {
                "state": "failed",
                "timestamp": _utc_now_iso(),
                "reason": f"training error: {exc}",
            })
            self._active_run = None
            raise

        manifest, manifest_error = self._manifest_lineage(adapter_path)
        receipt = self._manifest_receipt_stamp(manifest or {}) if manifest_error is None else None
        candidate_lineage = {
            "manifest_hash": result.corpus_manifest_hash,
            "manifest_path": result.corpus_manifest_path,
            "snapshot_hash": result.corpus_snapshot_hash,
            "policy_digest": result.corpus_policy_digest,
            "semantic_checkpoint_generation": result.semantic_checkpoint_generation,
            "semantic_checkpoint_id": result.semantic_checkpoint_id,
            "assertion_lineage": [list(pair) for pair in result.assertion_lineage],
            "state": "candidate",
        }
        if receipt is not None and result.corpus_manifest_hash == receipt["manifest_hash"]:
            candidate_lineage.update(receipt)
            self._adapter_lineage[adapter_path] = candidate_lineage

        if result.promoted and receipt is None:
            # A promoted adapter without the immutable corpus receipt is never
            # a valid served artifact, even if a trainer reports success.
            result.promoted = False
            result.promoted_adapter_path = None
            result.reason = "candidate missing governed corpus manifest"
        elif result.promoted and result.corpus_manifest_hash != receipt["manifest_hash"]:
            result.promoted = False
            result.promoted_adapter_path = None
            result.reason = "candidate manifest receipt does not match training result"

        if result.promoted and result.promoted_adapter_path:
            lifecycle_problem = await self._verify_adapter_lineage(
                result.promoted_adapter_path, before_promotion=True
            )
            if lifecycle_problem:
                result.promoted = False
                result.promoted_adapter_path = None
                result.reason = f"candidate quarantined: {lifecycle_problem}"

        if result.promoted and result.promoted_adapter_path:
            self._active_adapter_path = result.promoted_adapter_path
            self._last_val_loss = result.val_loss
            self._adapter_lineage[result.promoted_adapter_path]["state"] = "served"
            # Persist the new served adapter + its val loss so the pointer and
            # the regression baseline survive a restart.
            await self._persist_config()
        elif receipt is not None:
            # Candidate lineage is durable even when fidelity rejects it; an
            # operator can inspect exactly why it must not silently be served.
            await self._persist_config()

        logger.info(
            "parametric-self %s cycle: trained=%s promoted=%s val_loss=%s (%s)",
            trigger, result.trained, result.promoted, result.val_loss, result.reason,
        )
        outcome = {
            "trained": result.trained,
            "promoted": result.promoted,
            "val_loss": result.val_loss,
            "reason": result.reason,
            "corpus_train": result.corpus_train,
            "corpus_valid": result.corpus_valid,
            "corpus_manifest_hash": result.corpus_manifest_hash,
            "semantic_checkpoint_generation": result.semantic_checkpoint_generation,
        }
        await self._update_run_history(run_id, {
            "state": "completed",
            "timestamp": _utc_now_iso(),
            "trained": result.trained,
            "promoted": result.promoted,
            "val_loss": result.val_loss,
            "reason": result.reason,
            "corpus_train": result.corpus_train,
            # Record the served path only on promotion (matches the prior
            # contract); otherwise keep the candidate path for traceability.
            "adapter_path": result.promoted_adapter_path or adapter_path,
            "corpus_manifest_hash": result.corpus_manifest_hash,
            "semantic_checkpoint_generation": result.semantic_checkpoint_generation,
            "semantic_checkpoint_id": result.semantic_checkpoint_id,
            "corpus_snapshot_hash": result.corpus_snapshot_hash,
            "corpus_policy_digest": result.corpus_policy_digest,
        })
        self._active_run = None
        return outcome

    async def _begin_active_run(self, *, trigger: str, work_dir: str) -> Dict[str, Any]:
        """Create the durable in-progress record for a training run."""
        run_id = uuid.uuid4().hex[:12]
        adapter_id = uuid.uuid4().hex[:12]
        adapter_path = str(Path(work_dir) / "candidates" / adapter_id)
        started_at = _utc_now_iso()
        active_run = {
            "run_id": run_id,
            "adapter_id": adapter_id,
            "trigger": trigger,
            "started_at": started_at,
            "state": "in_progress",
            "adapter_path": adapter_path,
        }
        self._active_run = dict(active_run)
        await self._append_run_history({
            "run_id": run_id,
            "timestamp": started_at,
            "started_at": started_at,
            "trigger": trigger,
            "state": "in_progress",
            "trained": False,
            "promoted": False,
            "val_loss": None,
            "reason": "training in progress",
            "corpus_train": 0,
            "adapter_id": adapter_id,
            "adapter_path": adapter_path,
        })
        return dict(active_run)

    async def _interrupt_active_run(
        self, *, reason: str, run_id: Optional[str] = None,
    ) -> None:
        """Terminalize the matching active record after cancellation/disable.

        The matching id makes this safe when cancellation and teardown race: the
        first caller clears the record, and every later caller becomes a no-op.
        ``_update_run_history`` is already best effort, but clearing the local
        active state remains essential so the operator never sees a phantom run.
        """
        active_run = getattr(self, "_active_run", None)
        if active_run is None or (run_id is not None and active_run.get("run_id") != run_id):
            return
        try:
            await self._update_run_history(active_run["run_id"], {
                "state": "interrupted",
                "timestamp": _utc_now_iso(),
                "reason": reason,
            })
        except Exception as exc:  # cancellation cleanup must not mask cancellation
            logger.warning("Failed to mark parametric-self run interrupted: %s", exc)
        finally:
            if self._active_run is active_run:
                self._active_run = None

    async def _mark_training_shutdown_incomplete(
        self, *, reason: str, run_id: Optional[str] = None,
    ) -> None:
        """Expose an unconfirmed child shutdown without claiming it is terminal."""
        diagnostic = f"training shutdown incomplete: {reason}"
        self._training_shutdown_incomplete = diagnostic
        active_run = getattr(self, "_active_run", None)
        if active_run is None:
            return
        resolved_run_id = run_id or active_run.get("run_id")
        if active_run.get("run_id") != resolved_run_id:
            return
        active_run["state"] = "shutdown_incomplete"
        try:
            await self._update_run_history(resolved_run_id, {
                "state": "shutdown_incomplete",
                "timestamp": _utc_now_iso(),
                "reason": diagnostic,
            })
        except Exception as exc:  # preserve the local safety marker regardless
            logger.warning("Failed to mark parametric-self shutdown incomplete: %s", exc)

    async def _resolve_training_shutdown_incomplete(self) -> None:
        """Clear a prior incomplete-shutdown block after bulk stop confirmation."""
        reason = "run cancelled (feature disabled; bulk trainer stop confirmed)"
        active_run = getattr(self, "_active_run", None)
        run_ids = set()
        if active_run is not None and active_run.get("state") == "shutdown_incomplete":
            run_ids.add(active_run["run_id"])
        for entry in await self._load_run_history():
            if entry.get("state") == "shutdown_incomplete" and entry.get("run_id"):
                run_ids.add(entry["run_id"])
        for run_id in run_ids:
            await self._update_run_history(run_id, {
                "state": "interrupted",
                "timestamp": _utc_now_iso(),
                "reason": reason,
            })
        if active_run is not None and active_run.get("run_id") in run_ids:
            self._active_run = None
        self._training_shutdown_incomplete = None

    # ------------------------------------------------------------------
    # Run-history store (append-only, capped) — lets the agent introspect
    # its own training lifecycle (feedback_agent_must_introspect_lifecycle).
    # ------------------------------------------------------------------

    def _history_node_id(self) -> str:
        return f"parametric_self_runs:{self.name}"

    async def _append_run_history(self, entry: Dict[str, Any]) -> None:
        """Append one run record to the durable, capped history list."""
        storage = getattr(self.agent, "storage", None)
        if storage is None:
            return
        try:
            from kestrel_sovereign.storage.async_graph_store import GraphNode
            runs = await self._load_run_history()
            runs.append(entry)
            # Keep only the most recent N so the node stays bounded.
            runs = runs[-_RUN_HISTORY_LIMIT:]
            await storage.add_node(GraphNode(
                node_id=self._history_node_id(),
                node_type="parametric_self_runs",
                label=f"{self.name} training runs",
                properties={"runs": runs},
            ))
        except Exception as e:  # history is best-effort; never break a cycle
            logger.warning("Failed to append parametric-self run history: %s", e)

    async def _update_run_history(self, run_id: str, updates: Dict[str, Any]) -> None:
        """Merge ``updates`` into the most recent history entry with ``run_id``.

        Lets a run's record transition in place (``in_progress`` ->
        ``completed``/``skipped``/``failed``/``interrupted`` or an observable
        ``shutdown_incomplete`` state) instead of appending a second entry, so
        an in-flight run is one durable record an agent can poll to completion.
        No-ops if the entry is gone (capped out) — falls back to appending so the
        outcome is never silently lost.
        """
        storage = getattr(self.agent, "storage", None)
        if storage is None:
            return
        try:
            from kestrel_sovereign.storage.async_graph_store import GraphNode
            runs = await self._load_run_history()
            for entry in reversed(runs):
                if entry.get("run_id") == run_id:
                    entry.update(updates)
                    break
            else:
                merged = {"run_id": run_id}
                merged.update(updates)
                runs.append(merged)
                runs = runs[-_RUN_HISTORY_LIMIT:]
            await storage.add_node(GraphNode(
                node_id=self._history_node_id(),
                node_type="parametric_self_runs",
                label=f"{self.name} training runs",
                properties={"runs": runs},
            ))
        except Exception as e:  # history is best-effort; never break a cycle
            logger.warning("Failed to update parametric-self run history: %s", e)

    async def _reconcile_stale_runs(self) -> None:
        """Mark any persisted ``in_progress`` run as interrupted on load.

        A detached run does not survive a restart, so an ``in_progress`` entry
        found at startup is stale: its task is gone and it will never complete.
        Flipping it to ``interrupted`` keeps history honest and prevents the
        progress/status tools from reporting a run that is not actually running.
        """
        storage = getattr(self.agent, "storage", None)
        if storage is None:
            return
        try:
            from kestrel_sovereign.storage.async_graph_store import GraphNode
            runs = await self._load_run_history()
            changed = False
            for entry in runs:
                if entry.get("state") == "in_progress":
                    entry["state"] = "interrupted"
                    entry["reason"] = "run interrupted (process restarted before completion)"
                    changed = True
                elif entry.get("state") == "shutdown_incomplete":
                    # A replacement feature instance cannot prove the old
                    # process's child exited. Preserve the safety block and its
                    # durable diagnostic rather than silently admitting a new
                    # training mutation alongside a possible survivor.
                    self._training_shutdown_incomplete = str(
                        entry.get("reason", "training shutdown incomplete")
                    )
                    self._cycle_in_flight = True
            if changed:
                await storage.add_node(GraphNode(
                    node_id=self._history_node_id(),
                    node_type="parametric_self_runs",
                    label=f"{self.name} training runs",
                    properties={"runs": runs},
                ))
        except Exception as e:  # reconciliation is best-effort; never break load
            logger.warning("Failed to reconcile stale parametric-self runs: %s", e)

    async def _load_run_history(self) -> List[Dict[str, Any]]:
        """Load the run-history list (most recent last); [] if absent/malformed."""
        storage = getattr(self.agent, "storage", None)
        if storage is None:
            return []
        try:
            node = await storage.get_node(self._history_node_id())
            if node is None:
                return []
            runs = node.properties.get("runs")
            if isinstance(runs, str):
                runs = json.loads(runs)
            return list(runs) if isinstance(runs, list) else []
        except Exception as e:
            logger.warning("Failed to load parametric-self run history (ignored): %s", e)
            return []

    async def post_all_features_loaded(self, agent) -> None:
        """Register this feature's sleep-hook wrapper on the core sleep_hooks list.

        The wrapper mirrors reflection's ``ReflectionSleepHook`` and is appended
        to ``agent.sleep_hooks`` (kestrel-sovereign #1784), so nightly training
        fires after consolidation alongside reflection. Requires a core with the
        sleep-hook list; on an older core ``sleep_hooks`` is initialized here but
        the cycle won't dispatch it until core is upgraded (lockstep release).

        Also restores any persisted per-agent config (a durable
        ``enable_nightly_training``) now that storage is up, so the enablement
        survives restarts.
        """
        from .sleep_hook import create_parametric_self_sleep_hook

        self._manual_runs_enabled = True

        await self._restore_persisted_config()
        # A restored pointer is never trusted just because it was persisted.
        # Verify/quarantine it before this feature registers any serving-adjacent
        # hook or exposes the adapter through its normal control surface.
        if self._active_adapter_path:
            await self._verify_adapter_lineage(self._active_adapter_path)
        # A detached run can't survive a restart; reconcile any lingering
        # in_progress record so introspection never reports a dead run as active.
        await self._reconcile_stale_runs()

        if getattr(agent, "sleep_hooks", None) is None:
            agent.sleep_hooks = []
        # Idempotent re-enable: drop a previously-registered hook before adding
        # the fresh one (each call builds a new wrapper object).
        prior = getattr(self, "_sleep_hook", None)
        if prior is not None and prior in agent.sleep_hooks:
            agent.sleep_hooks.remove(prior)
        self._sleep_hook = create_parametric_self_sleep_hook(agent)
        if self._sleep_hook is not None:
            agent.sleep_hooks.append(self._sleep_hook)

    async def on_disable(self) -> None:
        """Unregister the sleep hook so a disabled feature stops running at sleep.

        ``post_all_features_loaded`` appends to ``agent.sleep_hooks`` manually,
        so teardown must remove it — otherwise a disabled/reloaded feature keeps
        training during sleep and re-enabling duplicates the hook.

        Also cancels any in-flight detached manual run (``train_now``): that task
        is not tracked by the framework once the tool returned, so without this a
        disabled/reloaded feature could still promote an adapter and persist
        config after teardown. Cancellation unwinds through ``_run_training_cycle``'s
        ``finally`` (clearing ``_cycle_in_flight``). Cancelling the asyncio task
        only stops the Python poller, so we also terminate the spawned
        ``mlx_lm.lora`` subprocess(es) via the adapter — otherwise an orphaned
        GPU-heavy job keeps running and writing into the adapter dir.
        """
        self._ensure_training_lifecycle_state()
        # Invalidate an in-progress command BEFORE awaiting the transition lock.
        # A command that is blocked on durable history creation will see this
        # generation mismatch and refuse to publish a detached trainer.
        self._manual_runs_enabled = False
        self._manual_run_generation += 1
        async with self._manual_run_lock:
            tasks = {
                task for task in (
                    getattr(self, "_training_task", None),
                    getattr(self, "_cycle_task", None),
                )
                if task is not None and not task.done() and task is not asyncio.current_task()
            }
            for task in tasks:
                task.cancel()
            for task in tasks:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:  # teardown must never raise
                    logger.warning("Parametric-self training task failed during disable: %s", exc)
            self._training_task = None
            self._cycle_task = None

            # The task-level cancellation path tears down a live child before it
            # finalizes history.  Call the adapter here as well for a task that
            # was cancelled before its runner began, and do it BEFORE state or
            # history cleanup so corpus cleanup can never race a live trainer.
            adapter = getattr(self, "_adapter", None)
            may_have_live_child = (
                self._active_run is not None or self._training_shutdown_incomplete is not None
            )
            shutdown_confirmed = not may_have_live_child
            if adapter is not None and hasattr(adapter, "cancel_all"):
                try:
                    shutdown = await adapter.cancel_all()
                    shutdown_confirmed = getattr(shutdown, "all_stopped", None) is True
                    if not shutdown_confirmed:
                        logger.critical(
                            "Parametric-self shutdown incomplete: bulk trainer stop did not confirm all children stopped"
                        )
                except Exception as exc:  # teardown must never raise
                    shutdown_confirmed = False
                    logger.warning("Failed to cancel parametric-self training subprocess(es): %s", exc)

            if not shutdown_confirmed:
                active_run = getattr(self, "_active_run", None)
                if active_run is not None:
                    await self._mark_training_shutdown_incomplete(
                        run_id=active_run["run_id"],
                        reason="bulk trainer stop did not confirm all children stopped",
                    )
                else:
                    self._training_shutdown_incomplete = (
                        "training shutdown incomplete: bulk trainer stop did not confirm all children stopped"
                    )
                self._cycle_in_flight = True
            else:
                # Force-clear the cross-trigger guard: if the cancel landed before
                # _runner started, its finally never ran and the guard would stay
                # stuck, permanently refusing later train/rollback on re-enable.
                self._cycle_in_flight = False
                if self._training_shutdown_incomplete is not None:
                    await self._resolve_training_shutdown_incomplete()
                else:
                    await self._interrupt_active_run(reason="run cancelled (feature disabled)")

        hook = getattr(self, "_sleep_hook", None)
        hooks = getattr(self.agent, "sleep_hooks", None)
        if hook is not None and hooks and hook in hooks:
            hooks.remove(hook)
        self._sleep_hook = None
