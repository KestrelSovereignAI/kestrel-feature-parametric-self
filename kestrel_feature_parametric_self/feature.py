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

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from kestrel_sdk.features.base import Feature, tool
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

from .cycle import run_nightly_cycle
from .fidelity import FidelityGate
from .local_mlx_adapter import LocalMLXAdapter
from .text_types import TextLoRAConfig

logger = logging.getLogger(__name__)


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
        command_prefix="!parametric-self",
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
            "parametric-self nightly cycle: trained=%s promoted=%s val_loss=%s (%s)",
            result.trained, result.promoted, result.val_loss, result.reason,
        )
        return {
            "trained": result.trained,
            "promoted": result.promoted,
            "val_loss": result.val_loss,
            "reason": result.reason,
            "corpus_train": result.corpus_train,
        }

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
