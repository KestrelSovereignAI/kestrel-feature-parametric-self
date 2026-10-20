"""No adapter reaches a live turn while ``inference_use`` is off.

Meridian's consent to training (epic #1, 2026-10-20) is conditional on the
adapter having no influence on live reasoning: one default-off switch for
in-loop serving, oracle use and conditioning, read at load, shown in status,
honoured by every route an adapter could take into a turn, with a test.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from kestrel_sdk.tools.result import ToolResultStatus

from kestrel_feature_parametric_self import ParametricSelfFeature
from kestrel_feature_parametric_self.inference_use import (
    InferenceUse,
    parse_inference_use,
)


class _FakeStorage:
    def __init__(self):
        self.nodes = {}

    async def add_node(self, node):
        self.nodes[node.node_id] = node

    async def get_node(self, node_id):
        return self.nodes.get(node_id)


def _agent(storage=None):
    agent = MagicMock()
    agent.storage = storage if storage is not None else _FakeStorage()
    agent.sleep_hooks = []
    agent.is_test_instance = False
    return agent


async def _loaded(agent) -> ParametricSelfFeature:
    feature = ParametricSelfFeature(agent=agent)
    await feature.initialize()
    agent.get_feature = MagicMock(
        side_effect=lambda n: feature
        if n in ("parametric_self", "ParametricSelfFeature")
        else None
    )
    await feature.post_all_features_loaded(agent)
    return feature


async def test_switch_defaults_off_and_status_reports_it():
    feature = await _loaded(_agent())

    status = await feature.parametric_self_status()

    assert status.status is ToolResultStatus.OK
    assert status.data["inference_use"] == "off"
    assert status.data["adapter_reaches_inference"] is False
    assert status.data["inference_use_rejected"] is None
    assert "Inference use: off" in status.confirmation
    assert (await feature.get_config())["inference_use"] == "off"


async def test_a_served_adapter_still_does_not_reach_inference(tmp_path):
    feature = await _loaded(_agent())
    feature._active_adapter_path = str(tmp_path / "adapter")
    feature._last_val_loss = 1.0

    assert feature.adapter_for_inference() is None


@pytest.mark.parametrize(
    "value", ["on", "in_loop", "oracle", "conditioning", "OFF", "", True, False, 1, None]
)
async def test_set_config_refuses_anything_but_off_and_applies_nothing(value):
    feature = await _loaded(_agent())

    with pytest.raises(ValueError, match="'off' is the only value"):
        await feature.set_config(
            {"inference_use": value, "enable_nightly_training": True}
        )

    assert feature._inference_use is InferenceUse.OFF
    # The refusal covers the whole request, not only the switch.
    assert feature._training_enabled is False


async def test_off_persists_and_is_read_back_at_load():
    storage = _FakeStorage()
    first = await _loaded(_agent(storage))
    await first.set_config({"inference_use": "off"})
    persisted = storage.nodes[first._config_node_id()].properties["config"]
    assert persisted["inference_use"] == "off"

    second = await _loaded(_agent(storage))

    assert second._inference_use is InferenceUse.OFF
    assert second._inference_use_rejected is None


async def test_an_unsupported_persisted_value_stays_off_and_is_reported():
    storage = _FakeStorage()
    first = await _loaded(_agent(storage))
    await first.set_config({"enable_nightly_training": True})
    config = storage.nodes[first._config_node_id()].properties["config"]
    config["inference_use"] = "in_loop"  # written by something other than set_config

    second = await _loaded(_agent(storage))
    status = await second.parametric_self_status()

    assert second.adapter_for_inference() is None
    assert status.data["inference_use"] == "off"
    assert "in_loop" in status.data["inference_use_rejected"]
    assert "WARNING" in status.confirmation
    # The rest of the persisted config is still restored.
    assert second._training_enabled is True


def test_parse_accepts_only_off():
    assert parse_inference_use("off") is InferenceUse.OFF
    assert parse_inference_use(InferenceUse.OFF) is InferenceUse.OFF
    assert [member.value for member in InferenceUse] == ["off"]


def test_the_feature_has_no_route_into_a_turn():
    """Tripwire: a new route has to come through ``adapter_for_inference``.

    A feature reaches a live turn through a hook, an HTTP router, or a tool the
    model can call (an oracle is a tool). If this fails because one was added,
    gate it on ``adapter_for_inference()``, add its ``InferenceUse`` value, and
    get the consent recorded on epic #1 before turning it on for any agent.
    """
    feature = ParametricSelfFeature(agent=MagicMock())

    assert feature.get_hooks() == []
    assert feature.get_router() is None
    assert sorted(tool.name for tool in feature.get_tools()) == [
        "parametric-self-adapters",
        "parametric-self-adopt",
        "parametric-self-history",
        "parametric-self-progress",
        "parametric-self-recover-shutdown",
        "parametric-self-rollback",
        "parametric-self-set-enabled",
        "parametric-self-status",
        "parametric-self-train-now",
    ]
