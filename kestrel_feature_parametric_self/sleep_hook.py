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

logger = logging.getLogger(__name__)


class ParametricSelfSleepHook:
    """Integrates parametric-self nightly training into the sleep cycle."""

    def __init__(self, feature) -> None:
        self.feature = feature

    async def on_pre_sleep(self, agent) -> Dict[str, Any]:
        """Parametric-self does no pre-sleep work — it trains post-consolidation."""
        return {"success": True, "skipped": True, "reason": "parametric-self trains post-consolidation only"}

    async def on_post_consolidation(self, agent, consolidation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate to the feature's post-consolidation training cycle."""
        try:
            return await self.feature.on_post_consolidation(consolidation_result)
        except Exception as exc:  # never let a training failure block sleep
            logger.warning("parametric-self post-consolidation failed: %s", exc)
            return {"trained": False, "promoted": False, "reason": f"error: {exc}"}


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
