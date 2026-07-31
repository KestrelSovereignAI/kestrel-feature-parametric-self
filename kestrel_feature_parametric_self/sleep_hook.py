"""Sleep-cycle hook wrapper for parametric-self.

Mirrors reflection's ``ReflectionSleepHook``: the sleep cycle invokes this
wrapper (``on_pre_sleep(agent)`` / ``on_post_consolidation(agent, result)``),
and it delegates to the feature's feature-layer methods. Keeping the wrapper
separate from the feature matches the platform convention and keeps the
feature method signatures aligned with reflection's.

Note: dispatch of this wrapper into the sleep cycle awaits the P2b core change.
Today ``sleep.py`` exposes a single ``agent.reflection_hook`` slot owned by
reflection; generalizing it to a sleep-hook list is the proper fix (epic #1).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from kestrel_sovereign.agent.sleep import SleepHookContract, SleepHookPhase

logger = logging.getLogger(__name__)


class ParametricSelfSleepHook:
    """Integrates parametric-self nightly training into the sleep cycle."""

    # The core scheduler turns this declarative edge into a hard-success
    # prerequisite: training cannot consume a corpus after partial/failed
    # semantic maintenance, even if a hook was registered earlier.
    sleep_hook_contract = SleepHookContract(
        hook_id="kestrel_feature_parametric_self.training",
        phase=SleepHookPhase.TRAINING,
        after=("kestrel_sovereign.semantic_maintenance",),
    )

    def __init__(self, feature) -> None:
        self.feature = feature

    async def on_pre_sleep(self, agent) -> Dict[str, Any]:
        """Parametric-self does no pre-sleep work — it trains post-consolidation."""
        return {"success": True, "skipped": True, "reason": "parametric-self trains post-consolidation only"}

    async def on_post_consolidation(self, agent, consolidation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate to the feature's post-consolidation training cycle."""
        try:
            result = await self.feature.on_post_consolidation(consolidation_result)
        except Exception as exc:  # never let a training failure block sleep
            logger.warning("parametric-self post-consolidation failed: %s", exc)
            return {
                "success": False, "skipped": False, "trained": False,
                "promoted": False, "reason": f"error: {exc}",
            }
        if not isinstance(result, dict):
            return {
                "success": False, "skipped": False, "trained": False,
                "promoted": False, "reason": "invalid training hook result",
            }

        outcome = dict(result)
        reason = str(outcome.get("reason") or "")
        if outcome.get("trained") is True:
            outcome.update(success=True, skipped=False)
        elif reason.startswith((
            "nightly training disabled", "training disabled", "training skipped",
            "another training run", "trainer unavailable on this host",
            "empty corpus", "nightly training interrupted while feature was disabled",
        )):
            # Expected operational no-ops stay visible without poisoning the
            # sleep dependency graph.
            outcome.update(success=True, skipped=True)
        else:
            # In particular, unavailable/unverified governed corpus evidence
            # is a failure, never the core's implicit-success fallback.
            outcome.update(success=False, skipped=False)
        return outcome


def create_parametric_self_sleep_hook(agent) -> Optional[ParametricSelfSleepHook]:
    """Create the hook if the parametric-self feature is loaded on this agent."""
    feature = None
    if hasattr(agent, "get_feature"):
        feature = agent.get_feature("parametric_self") or agent.get_feature("ParametricSelfFeature")
    elif hasattr(agent, "features"):
        feature = agent.features.get("ParametricSelfFeature") or agent.features.get("parametric_self")

    if feature is None:
        logger.debug("parametric-self feature not found; sleep hook not created")
        return None
    return ParametricSelfSleepHook(feature)
