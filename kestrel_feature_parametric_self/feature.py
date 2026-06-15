"""Parametric Self feature — owned, nightly-finetuned, in-the-loop second brain.

P0 scaffold. The feature loads, registers through the
``kestrel_sovereign.features`` entry point, exposes a status tool, and
declares the sleep-cycle hook surface it will use for nightly training. The
actual MLX LoRA trainer, the reflection-derived corpus, the fidelity gate,
and the in-loop conditioning/oracle integration land in later phases
(see epic #1 and ``docs/TWO_BRAIN_ARCHITECTURE.md``).

Design boundary: this is the *parametric* self (weights). Reflection keeps
the *symbolic* self-model (a trait dict). This feature depends on reflection
as its corpus source but does not live inside it, because it is active in the
runtime reasoning loop, not only during sleep.
"""

import logging
from typing import Any, Dict

from kestrel_sdk.features.base import Feature, tool
from kestrel_sdk.tools.base import ToolCategory
from kestrel_sdk.tools.result import ToolResult

logger = logging.getLogger(__name__)


class ParametricSelfFeature(Feature):
    """The agent's owned parametric self.

    A per-agent local model nightly-finetuned on the agent's own experience
    and (once proven) consulted in the reasoning loop. P0 wires the feature
    into the platform; behaviour arrives in P1+ per epic #1.
    """

    @property
    def tool_description(self) -> str:
        return (
            "Owned parametric self — a per-agent local model nightly-finetuned "
            "on the agent's own experience, consulted as a disposition prior and "
            "on-demand oracle alongside the frontier model"
        )

    async def initialize(self) -> None:
        """Initialize the parametric-self feature.

        P0 holds the slots the later phases fill: the MLX training adapter
        (P1), the fidelity gate (P2), and the served-adapter pointer (P3).
        They are ``None`` here so the feature loads cleanly on any platform —
        the Apple-Silicon trainer is imported lazily in P1, never at module
        import, so Linux installs and CI stay green.
        """
        self._adapter = None  # P1: LocalMLXAdapter (lazy, darwin/arm64 only)
        self._fidelity_gate = None  # P2: held-out promotion check
        self._active_adapter_path = None  # P3: currently served LoRA adapter

    @tool(
        name="parametric-self-status",
        description="Report the parametric-self feature state: configured base model, current adapter, and which build phase is live",
        category=ToolCategory.SYSTEM,
        command_prefix="!parametric-self",
    )
    async def parametric_self_status(self) -> ToolResult:
        """Report scaffold state.

        Returns:
            ToolResult describing the current (P0) state. Replaced with real
            adapter/training status in later phases.
        """
        data = {
            "phase": "P0 — scaffold",
            "trained_adapter": self._active_adapter_path,
            "trainer_available": False,
            "in_loop": False,
        }
        return ToolResult.ok(
            confirmation=(
                "Parametric-self is scaffolded (P0). Nightly training, the "
                "fidelity gate, and in-loop conditioning arrive in later phases "
                "(epic #1)."
            ),
            data=data,
        )

    async def on_post_consolidation(
        self,
        consolidation_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Feature-layer sleep hook: nightly adapter training (stub).

        Signature mirrors ``ReflectionFeature.on_post_consolidation`` exactly.
        The sleep cycle does NOT call feature methods directly — it invokes a
        ``*SleepHook`` wrapper (cf. reflection's ``ReflectionSleepHook``, whose
        ``on_post_consolidation(self, agent, consolidation_result)`` is the
        method ``sleep.py`` calls), and the wrapper delegates here with just
        ``consolidation_result``. So there is no sleep-context argument at this
        layer; this is the delegate target.

        P1/P2 add parametric-self's own SleepHook wrapper and resolve how it
        attaches: ``sleep.py`` currently exposes a single
        ``agent.reflection_hook`` slot that reflection owns, so a second
        feature's nightly hook needs either a core change (hook list) or
        chaining through reflection — tracked in epic #1 (P2).

        Fires after memory consolidation — the point at which reflection has
        produced the night's insights, the corpus this feature trains on. P0 is
        a no-op that records intent; P2 trains a LoRA adapter here and promotes
        it only if it clears the fidelity gate. Returns the legacy sleep-hook
        result dict shape so it slots in beside reflection's hook.
        """
        logger.debug(
            "parametric-self on_post_consolidation: training deferred to P2 "
            "(episodes_created=%s)",
            consolidation_result.get("episodes_created", 0),
        )
        return {
            "trained": False,
            "promoted": False,
            "reason": "scaffold — nightly training lands in P2 (epic #1)",
        }
