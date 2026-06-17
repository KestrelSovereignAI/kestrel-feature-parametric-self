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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from kestrel_sdk.features.base import Feature, tool
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

from .cycle import run_nightly_cycle
from .fidelity import FidelityGate, parse_final_val_loss
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

    async def initialize(self) -> None:
        """Initialize the parametric-self feature (training off by default)."""
        self._adapter = LocalMLXAdapter()           # lazy MLX; inert off Apple Silicon
        self._gate = FidelityGate()                 # held-out promotion check
        self._training_enabled = False              # per-agent gate; default OFF
        self._base_config = TextLoRAConfig()        # base hyperparameters
        self._active_adapter_path: Optional[str] = None   # currently served adapter
        self._last_val_loss: Optional[float] = None        # served adapter's fidelity
        # Optional overrides (tests / non-standard layouts); else resolved from agent.
        self._db_path: Optional[str] = None
        self._work_dir: Optional[str] = None
        self._sleep_hook = None                     # set in post_all_features_loaded
        # In-flight manual training run (train_now). Detached so the tool call
        # returns immediately; guarded so only one cycle runs at a time.
        self._training_task: Optional[asyncio.Task] = None
        # Cross-trigger serialization: nightly (on_post_consolidation) and manual
        # (train_now) cycles share the same corpus/work dir and the served-adapter
        # pointer, so only ONE may run at a time. Set/cleared synchronously inside
        # _run_training_cycle (no await between check and set → atomic in asyncio).
        self._cycle_in_flight = False

    async def get_config(self) -> Dict[str, Any]:
        return {
            "enable_nightly_training": self._training_enabled,
            "base_model": self._base_config.base_model,
            "active_adapter_path": self._active_adapter_path,
        }

    async def set_config(self, config: Dict[str, Any]) -> None:
        if "enable_nightly_training" in config:
            self._training_enabled = bool(config["enable_nightly_training"])
        if config.get("base_model"):
            self._base_config.base_model = str(config["base_model"])
        # Persist so the enablement survives restarts (durable per-agent gate).
        await self._persist_config()

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
        data = {
            "training_enabled": self._training_enabled,
            "trainer_available": self._adapter.is_available(),
            "base_model": self._base_config.base_model,
            "served_adapter": self._active_adapter_path,
            "served_val_loss": self._last_val_loss,
        }
        return ToolResult.ok(
            confirmation=(
                "Parametric-self "
                + ("ENABLED" if self._training_enabled else "disabled")
                + f"; trainer {'available' if data['trainer_available'] else 'unavailable on this host'}."
            ),
            data=data,
        )

    # ------------------------------------------------------------------
    # Incubator-Principle gate for the mutation tools
    # ------------------------------------------------------------------

    def _require_sovereign_class(self) -> Optional[ToolResult]:
        """Gate self-modification on agent class (the Incubator Principle).

        Managing the parametric self — training it, toggling nightly training,
        rolling back the served adapter — is the agent modifying its *own*
        weights. The Incubator Principle reserves self-modification for
        sovereign-class agents; governed/test instances
        (``agent.is_test_instance``) must not. Returns a refusal ``ToolResult``
        for a governed agent, else ``None``.

        Note on the *caller* gate: ``CallerContext.is_sovereign`` is threaded
        only to the command handler for a fixed set of core governance commands
        — it is not passed to feature ``@tool`` methods (they dispatch via the
        A2A TaskManager, which carries no caller context). So caller-level
        gating is not available here without a core change; the agent-class gate
        below is the meaningful self-modification boundary for this feature.
        """
        if getattr(self.agent, "is_test_instance", False):
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
        lines = [f"{len(recent)} training run(s) recorded ({promoted} promoted):"]
        for r in recent[:10]:
            lines.append(
                f"  {r.get('timestamp', '?')} [{r.get('trigger', '?')}] "
                f"trained={r.get('trained')} promoted={r.get('promoted')} "
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
        candidates = self._candidates_dir()
        served = self._active_adapter_path
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
                adapters.append({
                    "adapter_id": d.name,
                    "path": str(d),
                    "val_loss": val_loss,
                    "served": str(d) == str(served) if served else False,
                })
        if not adapters:
            return ToolResult.ok(
                confirmation="No candidate adapters on disk yet.",
                data={"adapters": [], "served_adapter": served},
            )
        lines = [f"{len(adapters)} candidate adapter(s):"]
        for a in adapters:
            mark = " (served)" if a["served"] else ""
            lines.append(f"  {a['adapter_id']}  val_loss={a['val_loss']}{mark}")
        return ToolResult.ok(
            confirmation="\n".join(lines),
            data={"adapters": adapters, "served_adapter": served},
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
        gate = self._require_sovereign_class()
        if gate is not None:
            return gate
        if not self._adapter.is_available():
            return ToolResult.failed("Trainer unavailable on this host (MLX/Apple Silicon required).")
        if self._cycle_in_flight or (self._training_task is not None and not self._training_task.done()):
            return ToolResult.failed("A parametric-self training run is already in progress.")

        # Run detached: a full cycle is ~24 min; the tool returns immediately and
        # the run records itself in history on completion. Errors are logged, not
        # surfaced to the caller (poll !parametric-self-history for the outcome).
        async def _runner() -> None:
            try:
                await self._run_training_cycle(trigger="manual")
            except Exception as exc:
                logger.warning("parametric-self manual training run failed: %s", exc)

        self._training_task = asyncio.create_task(_runner())
        return ToolResult.ok(
            confirmation=(
                "Parametric-self training run started in the background. "
                "Check `!parametric-self-history` for the outcome."
            ),
            data={"started": True},
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

    def _resolve_paths(self) -> Tuple[Optional[str], Optional[str]]:
        """Resolve (cognition_db_path, work_dir) for this agent.

        The agent exposes its cognition DB as ``storage_path`` (the SQLite file
        holding reflection_insights + graph_nodes); the data dir is its parent.
        There is no ``data_dir`` attribute on the agent.
        """
        if self._db_path and self._work_dir:
            return self._db_path, self._work_dir
        storage_path = getattr(self.agent, "storage_path", None)
        if not storage_path:
            return self._db_path, self._work_dir
        db = self._db_path or str(storage_path)
        work = self._work_dir or str(Path(storage_path).parent / "parametric_self")
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
        """
        if self._cycle_in_flight:
            return {"trained": False, "promoted": False, "reason": "another training run already in progress"}
        self._cycle_in_flight = True
        try:
            return await self._run_training_cycle_locked(trigger=trigger)
        finally:
            self._cycle_in_flight = False

    async def _run_training_cycle_locked(self, *, trigger: str) -> Dict[str, Any]:
        """Body of one cycle; only ever called with the in-flight guard held."""
        db_path, work_dir = self._resolve_paths()
        if not db_path or not work_dir:
            return {"trained": False, "promoted": False, "reason": "could not resolve agent storage_path"}

        agent_id = getattr(self.agent, "agent_id", None) or getattr(self.agent, "name", "agent")
        config = TextLoRAConfig.from_dict(self._base_config.to_dict())

        result = await run_nightly_cycle(
            agent_id=str(agent_id),
            db_path=db_path,
            work_dir=work_dir,
            adapter=self._adapter,
            gate=self._gate,
            config=config,
            prior_val_loss=self._last_val_loss,
        )

        if result.promoted and result.promoted_adapter_path:
            self._active_adapter_path = result.promoted_adapter_path
            self._last_val_loss = result.val_loss
            # Persist the new served adapter + its val loss so the pointer and
            # the regression baseline survive a restart.
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
        }
        await self._append_run_history({
            "timestamp": _utc_now_iso(),
            "trigger": trigger,
            "trained": result.trained,
            "promoted": result.promoted,
            "val_loss": result.val_loss,
            "reason": result.reason,
            "corpus_train": result.corpus_train,
            "adapter_path": result.promoted_adapter_path,
        })
        return outcome

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

        await self._restore_persisted_config()

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
        """
        hook = getattr(self, "_sleep_hook", None)
        hooks = getattr(self.agent, "sleep_hooks", None)
        if hook is not None and hooks and hook in hooks:
            hooks.remove(hook)
        self._sleep_hook = None
