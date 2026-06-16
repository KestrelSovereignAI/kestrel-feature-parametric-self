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

# Default cognition DB filename inside an agent's data_dir.
_DB_FILENAME = "kestrel_prime.db"


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
        """Resolve (cognition_db_path, work_dir) for this agent, best-effort."""
        if self._db_path and self._work_dir:
            return self._db_path, self._work_dir
        data_dir = getattr(self.agent, "data_dir", None) or getattr(self.agent, "_data_dir", None)
        if not data_dir:
            return self._db_path, self._work_dir
        base = Path(data_dir)
        db = self._db_path or str(base / _DB_FILENAME)
        work = self._work_dir or str(base / "parametric_self")
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
            return {"trained": False, "promoted": False, "reason": "could not resolve agent data_dir"}

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
        """Create this feature's sleep-hook wrapper once all features are up.

        The wrapper mirrors reflection's ``ReflectionSleepHook``. It is created
        here but its dispatch into the sleep cycle awaits the P2b core change
        (a sleep-hook list); we deliberately do NOT clobber ``agent.reflection_hook``.
        """
        from .sleep_hook import create_parametric_self_sleep_hook

        self._sleep_hook = create_parametric_self_sleep_hook(agent)
