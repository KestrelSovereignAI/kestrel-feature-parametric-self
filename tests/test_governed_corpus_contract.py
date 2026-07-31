"""Feature-side evidence for the public governed corpus integration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kestrel_feature_parametric_self import ParametricSelfFeature, ParametricSelfSleepHook, build_corpus


def _snapshot(*, revision="revision:one"):
    assertion = SimpleNamespace(
        assertion_id="assertion:one", revision_id=revision,
        subject=SimpleNamespace(value="https://example.test/agent"),
        predicate=SimpleNamespace(value="https://example.test/lesson"),
        object=SimpleNamespace(lexical_form="governed lesson"),
    )
    return SimpleNamespace(
        verified=True,
        examples=(SimpleNamespace(
            assertion=assertion, content_hash=f"sha256:{revision}", source_occurrences=(),
            decision=SimpleNamespace(included=True, reason=SimpleNamespace(value="included")),
        ),),
        snapshot_hash=f"sha256:snapshot:{revision}",
        policy=SimpleNamespace(digest="sha256:policy"),
        checkpoint=SimpleNamespace(generation=4, latest_event_id="event:4"),
        capability_versions={"semantic_maintenance": "1"},
    )


class _Host:
    def __init__(self, snapshot, *, fail_snapshot=False, tombstones=()):
        self.snapshot = snapshot
        self.fail_snapshot = fail_snapshot
        self.tombstones = tombstones
        self.nodes = {}

    async def governed_assertion_corpus_snapshot(self, **_kwargs):
        if self.fail_snapshot:
            raise RuntimeError("semantic maintenance incomplete")
        return self.snapshot

    async def governed_assertion_corpus_changes_since(self, _snapshot_value, **_kwargs):
        return SimpleNamespace(tombstones=self.tombstones)

    async def add_node(self, node):
        self.nodes[node.node_id] = node

    async def get_node(self, node_id):
        return self.nodes.get(node_id)


async def _feature(host, tmp_path):
    agent = MagicMock()
    agent.storage = host
    agent.storage_path = None  # proves a PostgreSQL-style host need not expose SQLite
    agent.parametric_self_work_dir = str(tmp_path / "work")
    agent.parametric_self_governed_corpus_policy = SimpleNamespace(digest="sha256:policy")
    agent.semantic_inference_profile = None
    agent.is_test_instance = False
    agent.agent_id = "agent"
    feature = ParametricSelfFeature(agent=agent)
    await feature.initialize()
    return feature


async def test_unavailable_or_incomplete_host_capability_is_a_visible_skip(tmp_path):
    feature = await _feature(_Host(_snapshot(), fail_snapshot=True), tmp_path)
    outcome = await feature._run_training_cycle_locked(trigger="nightly")
    assert outcome == {
        "trained": False, "promoted": False,
        "reason": "governed corpus unavailable or semantic maintenance incomplete",
    }


def test_training_hook_declares_successful_semantic_maintenance_prerequisite():
    contract = ParametricSelfSleepHook.sleep_hook_contract
    assert contract.hook_id == "kestrel_feature_parametric_self.training"
    assert contract.phase.value == "training"
    assert contract.after == ("kestrel_sovereign.semantic_maintenance",)


async def test_tombstone_quarantines_served_adapter_and_clean_rebuild_is_eligible(tmp_path):
    snapshot = _snapshot()
    candidate = tmp_path / "candidate"
    build_corpus(
        None, str(tmp_path / "corpus"), governed_snapshot=snapshot,
        manifest_dir=str(candidate),
    )
    tombstone = SimpleNamespace(
        assertion_id="assertion:one", revision_id="revision:one", operation="deleted",
    )
    host = _Host(snapshot, tombstones=(tombstone,))
    feature = await _feature(host, tmp_path)
    feature._active_adapter_path = str(candidate)
    feature._live_corpus_snapshot = snapshot

    reason = await feature._verify_adapter_lineage(str(candidate))
    assert "invalidated" in reason
    assert feature._active_adapter_path is None
    assert feature._adapter_lineage[str(candidate)]["state"] == "invalid"

    clean_candidate = tmp_path / "clean-candidate"
    build_corpus(
        None, str(tmp_path / "clean-corpus"), governed_snapshot=_snapshot(revision="revision:two"),
        manifest_dir=str(clean_candidate),
    )
    clean_host = _Host(_snapshot(revision="revision:two"))
    clean_feature = await _feature(clean_host, tmp_path)
    assert await clean_feature._verify_adapter_lineage(str(clean_candidate)) is None


async def test_adapter_without_manifest_is_never_served(tmp_path):
    host = _Host(_snapshot())
    feature = await _feature(host, tmp_path)
    candidate = Path(tmp_path / "work" / "candidates" / "candidate")
    candidate.parent.mkdir(parents=True)
    candidate.mkdir()
    (candidate / "train.log").write_text("Val loss 1.2\n")
    outcome = await feature.parametric_self_adopt(candidate.name)
    assert outcome.status.value == "error"
    assert "quarantined" in (outcome.error or "")
