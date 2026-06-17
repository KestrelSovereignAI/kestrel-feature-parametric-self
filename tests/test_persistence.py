"""enable_nightly_training persists across restarts via the agent graph store."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from kestrel_feature_parametric_self import ParametricSelfFeature


class _FakeStorage:
    """In-memory stand-in for the agent graph store (add_node/get_node)."""

    def __init__(self):
        self.nodes = {}

    async def add_node(self, node):
        self.nodes[node.node_id] = node

    async def get_node(self, node_id):
        return self.nodes.get(node_id)


def _agent(storage):
    agent = MagicMock()
    agent.storage = storage
    agent.sleep_hooks = []
    return agent


async def test_enable_persists_then_restores_across_restart():
    storage = _FakeStorage()
    agent = _agent(storage)

    # Session 1: enable training
    f1 = ParametricSelfFeature(agent=agent)
    await f1.initialize()
    assert f1._training_enabled is False
    await f1.set_config({"enable_nightly_training": True})
    assert f1._config_node_id() in storage.nodes  # persisted

    # Session 2 (restart): fresh instance, same storage → default off, then restored
    f2 = ParametricSelfFeature(agent=agent)
    await f2.initialize()
    assert f2._training_enabled is False           # default before restore
    agent.get_feature = MagicMock(
        side_effect=lambda n: f2 if n in ("parametric_self", "ParametricSelfFeature") else None
    )
    await f2.post_all_features_loaded(agent)        # restore runs here
    assert f2._training_enabled is True             # survived the "restart"


async def test_served_adapter_and_val_loss_persist_across_restart():
    """The served-adapter pointer + last val loss survive a restart, so status
    keeps the pointer and the regression gate keeps its baseline."""
    storage = _FakeStorage()
    agent = _agent(storage)

    f1 = ParametricSelfFeature(agent=agent)
    await f1.initialize()
    f1._active_adapter_path = "/data/parametric_self/candidates/abc123"
    f1._last_val_loss = 2.31
    await f1._persist_config()

    f2 = ParametricSelfFeature(agent=agent)
    await f2.initialize()
    assert f2._active_adapter_path is None and f2._last_val_loss is None  # defaults
    await f2._restore_persisted_config()
    assert f2._active_adapter_path == "/data/parametric_self/candidates/abc123"
    assert f2._last_val_loss == 2.31


async def test_no_storage_is_graceful():
    agent = _agent(storage=None)
    f = ParametricSelfFeature(agent=agent)
    await f.initialize()
    await f.set_config({"enable_nightly_training": True})  # must not raise
    assert f._training_enabled is True                     # in-memory still works


async def test_malformed_persisted_config_does_not_raise():
    """A bad JSON config string must be ignored, never abort startup (codex P2)."""
    class _BadNode:
        properties = {"config": "{not valid json"}

    class _BadStorage:
        async def get_node(self, node_id):
            return _BadNode()

    agent = _agent(storage=_BadStorage())
    f = ParametricSelfFeature(agent=agent)
    await f.initialize()

    await f._restore_persisted_config()  # must not raise
    assert f._training_enabled is False  # defaults preserved


async def test_base_model_override_persists():
    storage = _FakeStorage()
    agent = _agent(storage)
    f = ParametricSelfFeature(agent=agent)
    await f.initialize()
    await f.set_config({"base_model": "gemma-4-26B-A4B-mlx-4bit"})

    f2 = ParametricSelfFeature(agent=agent)
    await f2.initialize()
    await f2._restore_persisted_config()
    assert f2._base_config.base_model == "gemma-4-26B-A4B-mlx-4bit"
