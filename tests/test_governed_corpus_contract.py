"""Feature-side evidence for the public governed corpus integration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kestrel_feature_parametric_self import ParametricSelfFeature, ParametricSelfSleepHook, build_corpus


def _snapshot(
    *, revision="revision:one", policy_digest="sha256:policy",
    capability_versions=None, generation=4, event_id="event:4", snapshot_hash=None,
):
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
        snapshot_hash=snapshot_hash or f"sha256:snapshot:{revision}",
        policy=SimpleNamespace(digest=policy_digest),
        checkpoint=SimpleNamespace(generation=generation, latest_event_id=event_id),
        capability_versions=capability_versions or {"semantic_maintenance": "1"},
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
        return SimpleNamespace(
            tombstones=self.tombstones,
            since_checkpoint=_snapshot_value.checkpoint,
            checkpoint=self.snapshot.checkpoint,
            snapshot_hash=f"sha256:delta:{self.snapshot.snapshot_hash}",
            observability=SimpleNamespace(policy_digest=self.snapshot.policy.digest),
        )

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
    agent.sleep_hooks = []
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


async def test_same_lineage_with_changed_policy_or_capability_pins_is_quarantined(tmp_path):
    baseline = _snapshot()
    candidate = tmp_path / "candidate"
    build_corpus(None, str(tmp_path / "corpus"), governed_snapshot=baseline, manifest_dir=str(candidate))

    policy_host = _Host(baseline)
    policy_feature = await _feature(policy_host, tmp_path)
    policy_feature._live_corpus_snapshot = baseline
    policy_feature.agent.parametric_self_governed_corpus_policy = SimpleNamespace(digest="sha256:new-policy")
    reason = await policy_feature._verify_adapter_lineage(str(candidate))
    assert reason == "governed corpus policy changed; rebuild required"
    assert policy_feature._active_adapter_path is None

    changed_pins = _snapshot(capability_versions={"semantic_maintenance": "2"})
    pin_feature = await _feature(_Host(changed_pins), tmp_path)
    pin_feature._live_corpus_snapshot = changed_pins
    reason = await pin_feature._verify_adapter_lineage(str(candidate))
    assert reason == "governed semantic capability pins changed; rebuild required"


async def test_same_checkpoint_with_changed_snapshot_receipt_is_quarantined(tmp_path):
    baseline = _snapshot()
    candidate = tmp_path / "candidate"
    build_corpus(None, str(tmp_path / "corpus"), governed_snapshot=baseline, manifest_dir=str(candidate))

    altered = _snapshot(snapshot_hash="sha256:other-receipt")
    feature = await _feature(_Host(altered), tmp_path)
    feature._live_corpus_snapshot = altered
    reason = await feature._verify_adapter_lineage(str(candidate))
    assert reason == "governed corpus snapshot receipt changed; rebuild required"


async def test_delta_must_be_rooted_at_the_exact_manifest_checkpoint(tmp_path):
    baseline = _snapshot()
    candidate = tmp_path / "candidate"
    build_corpus(None, str(tmp_path / "corpus"), governed_snapshot=baseline, manifest_dir=str(candidate))
    host = _Host(baseline)

    async def wrong_base(_snapshot_value, **_kwargs):
        return SimpleNamespace(
            tombstones=(),
            since_checkpoint=SimpleNamespace(generation=3, latest_event_id="event:3"),
            checkpoint=baseline.checkpoint,
            snapshot_hash="sha256:delta",
            observability=SimpleNamespace(policy_digest="sha256:policy"),
        )

    host.governed_assertion_corpus_changes_since = wrong_base
    feature = await _feature(host, tmp_path)
    feature._live_corpus_snapshot = baseline
    reason = await feature._verify_adapter_lineage(str(candidate))
    assert reason == "governed corpus delta evidence mismatch; adapter cannot be verified"


async def test_explicit_policy_update_immediately_invalidates_served_adapter(tmp_path):
    feature = await _feature(_Host(_snapshot()), tmp_path)
    feature._active_adapter_path = str(tmp_path / "served")
    await feature.set_config({"governed_corpus_policy": SimpleNamespace(digest="sha256:new-policy")})
    assert feature._active_adapter_path is None
    assert feature._quarantined_adapters[str(tmp_path / "served")] == (
        "governed corpus policy updated; rebuild required"
    )


async def test_restart_quarantines_missing_or_stale_manifest_before_hook_registration(tmp_path):
    """A persisted pointer is verified before restart exposes it as served."""
    baseline = _snapshot()
    missing = tmp_path / "missing"
    missing.mkdir()
    host = _Host(baseline)
    first = await _feature(host, tmp_path)
    first._active_adapter_path = str(missing)
    await first._persist_config()

    restarted = await _feature(host, tmp_path)
    restarted.agent.get_feature = MagicMock(return_value=restarted)
    await restarted.post_all_features_loaded(restarted.agent)
    assert restarted._active_adapter_path is None
    assert "manifest" in restarted._quarantined_adapters[str(missing)]

    candidate = tmp_path / "stale"
    build_corpus(None, str(tmp_path / "corpus"), governed_snapshot=baseline, manifest_dir=str(candidate))
    stale_host = _Host(_snapshot(revision="revision:two", generation=5, event_id="event:5"))
    first = await _feature(stale_host, tmp_path)
    first._active_adapter_path = str(candidate)
    await first._persist_config()
    restarted = await _feature(stale_host, tmp_path)
    restarted.agent.get_feature = MagicMock(return_value=restarted)
    await restarted.post_all_features_loaded(restarted.agent)
    assert restarted._active_adapter_path is None
    assert "lineage no longer current" in restarted._quarantined_adapters[str(candidate)]
