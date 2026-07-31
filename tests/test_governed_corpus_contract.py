"""Feature-side evidence for the public governed corpus integration."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kestrel_feature_parametric_self import ParametricSelfFeature, ParametricSelfSleepHook, build_corpus


def _snapshot(
    *, revision="revision:one", policy_digest="sha256:policy",
    capability_versions=None, generation=4, event_id="event:4", snapshot_hash=None,
    tenant_id="tenant:test",
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
        tenant_id=tenant_id,
        policy=SimpleNamespace(digest=policy_digest),
        checkpoint=SimpleNamespace(tenant_id=tenant_id, generation=generation, latest_event_id=event_id),
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


def _stamp(feature, candidate) -> None:
    manifest, error = feature._manifest_lineage(str(candidate))
    assert error is None
    receipt = feature._manifest_receipt_stamp(manifest or {})
    assert receipt is not None
    feature._adapter_lineage[str(candidate)] = {**receipt, "state": "candidate"}


async def test_unavailable_or_incomplete_host_capability_is_a_visible_skip(tmp_path):
    feature = await _feature(_Host(_snapshot(), fail_snapshot=True), tmp_path)
    # This test exercises the governed-capability seam, not local MLX support.
    feature._adapter.is_available = lambda: True
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
    _stamp(feature, candidate)
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
    _stamp(clean_feature, clean_candidate)
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


async def test_untracked_candidate_with_valid_manifest_is_inspection_only_not_adoptable(tmp_path):
    host = _Host(_snapshot())
    feature = await _feature(host, tmp_path)
    candidate = tmp_path / "work" / "candidates" / "untracked"
    candidate.mkdir(parents=True)
    (candidate / "train.log").write_text("Val loss 1.2\n")
    build_corpus(None, str(tmp_path / "corpus"), governed_snapshot=host.snapshot, manifest_dir=str(candidate))

    outcome = await feature.parametric_self_adopt("untracked")
    assert outcome.status.value == "error"
    assert "durable adapter lineage receipt unavailable" in (outcome.error or "")
    assert feature._active_adapter_path is None


async def test_same_lineage_with_changed_policy_or_capability_pins_is_quarantined(tmp_path):
    baseline = _snapshot()
    candidate = tmp_path / "candidate"
    build_corpus(None, str(tmp_path / "corpus"), governed_snapshot=baseline, manifest_dir=str(candidate))

    policy_host = _Host(baseline)
    policy_feature = await _feature(policy_host, tmp_path)
    _stamp(policy_feature, candidate)
    policy_feature._live_corpus_snapshot = baseline
    policy_feature.agent.parametric_self_governed_corpus_policy = SimpleNamespace(digest="sha256:new-policy")
    reason = await policy_feature._verify_adapter_lineage(str(candidate))
    assert reason == "governed corpus policy changed; rebuild required"
    assert policy_feature._active_adapter_path is None

    changed_pins = _snapshot(capability_versions={"semantic_maintenance": "2"})
    pin_feature = await _feature(_Host(changed_pins), tmp_path)
    _stamp(pin_feature, candidate)
    pin_feature._live_corpus_snapshot = changed_pins
    reason = await pin_feature._verify_adapter_lineage(str(candidate))
    assert reason == "governed semantic capability pins changed; rebuild required"


async def test_same_checkpoint_with_changed_snapshot_receipt_is_quarantined(tmp_path):
    baseline = _snapshot()
    candidate = tmp_path / "candidate"
    build_corpus(None, str(tmp_path / "corpus"), governed_snapshot=baseline, manifest_dir=str(candidate))

    altered = _snapshot(snapshot_hash="sha256:other-receipt")
    feature = await _feature(_Host(altered), tmp_path)
    _stamp(feature, candidate)
    feature._live_corpus_snapshot = altered
    reason = await feature._verify_adapter_lineage(str(candidate))
    assert reason == "governed corpus snapshot receipt changed; rebuild required"


async def test_rehashed_replacement_manifest_cannot_relabel_existing_adapter_weights(tmp_path):
    """The durable receipt, not a self-consistent replacement manifest, binds weights."""
    trained_snapshot = _snapshot(revision="revision:one")
    candidate = tmp_path / "candidate"
    build_corpus(None, str(tmp_path / "corpus-old"), governed_snapshot=trained_snapshot, manifest_dir=str(candidate))
    host = _Host(_snapshot(revision="revision:two", generation=5, event_id="event:5"))
    feature = await _feature(host, tmp_path)
    original_manifest, error = feature._manifest_lineage(str(candidate))
    assert error is None
    feature._adapter_lineage[str(candidate)] = {
        **(feature._manifest_receipt_stamp(original_manifest or {}) or {}),
        "state": "served",
    }
    feature._active_adapter_path = str(candidate)

    replacement = tmp_path / "replacement"
    build_corpus(
        None, str(tmp_path / "corpus-new"), governed_snapshot=host.snapshot,
        manifest_dir=str(replacement),
    )
    target = candidate / "corpus_manifest.json"
    target.chmod(0o644)
    target.write_text((replacement / "corpus_manifest.json").read_text())

    reason = await feature._verify_adapter_lineage(str(candidate))
    assert reason == "persisted adapter lineage receipt mismatch; rebuild required"
    assert feature._active_adapter_path is None


async def test_delta_must_be_rooted_at_the_exact_manifest_checkpoint(tmp_path):
    baseline = _snapshot()
    candidate = tmp_path / "candidate"
    build_corpus(None, str(tmp_path / "corpus"), governed_snapshot=baseline, manifest_dir=str(candidate))
    host = _Host(baseline)
    feature = await _feature(host, tmp_path)
    _stamp(feature, candidate)

    async def wrong_base(_snapshot_value, **_kwargs):
        return SimpleNamespace(
            tombstones=(),
            since_checkpoint=SimpleNamespace(tenant_id="tenant:test", generation=3, latest_event_id="event:3"),
            checkpoint=baseline.checkpoint,
            snapshot_hash="sha256:delta",
            observability=SimpleNamespace(policy_digest="sha256:policy"),
        )

    host.governed_assertion_corpus_changes_since = wrong_base
    feature._live_corpus_snapshot = baseline
    reason = await feature._verify_adapter_lineage(str(candidate))
    assert reason == "governed corpus delta evidence mismatch; adapter cannot be verified"


async def test_delta_from_a_foreign_tenant_is_never_accepted(tmp_path):
    baseline = _snapshot()
    candidate = tmp_path / "candidate"
    build_corpus(None, str(tmp_path / "corpus"), governed_snapshot=baseline, manifest_dir=str(candidate))
    host = _Host(baseline)
    feature = await _feature(host, tmp_path)
    _stamp(feature, candidate)

    async def foreign_tenant(_snapshot_value, **_kwargs):
        return SimpleNamespace(
            tombstones=(),
            since_checkpoint=SimpleNamespace(
                tenant_id="tenant:other", generation=4, latest_event_id="event:4",
            ),
            checkpoint=SimpleNamespace(
                tenant_id="tenant:other", generation=5, latest_event_id="event:5",
            ),
            snapshot_hash="sha256:delta",
            observability=SimpleNamespace(policy_digest="sha256:policy"),
        )

    host.governed_assertion_corpus_changes_since = foreign_tenant
    feature._live_corpus_snapshot = baseline
    reason = await feature._verify_adapter_lineage(str(candidate))
    assert reason == "governed corpus delta evidence mismatch; adapter cannot be verified"


async def test_real_core_mappingproxy_capability_versions_are_accepted(tmp_path):
    """The public core snapshot freezes its version map; do not reject it as non-dict."""
    from kestrel_sovereign.knowledge.corpus import (
        CORPUS_SCHEMA_VERSION,
        CorpusCheckpoint,
        GovernedCorpusObservability,
        GovernedCorpusSnapshot,
    )

    checkpoint = CorpusCheckpoint("tenant:test", 4, "event:4")
    snapshot = GovernedCorpusSnapshot(
        CORPUS_SCHEMA_VERSION,
        "tenant:test",
        checkpoint,
        MappingProxyType({"semantic_maintenance": "1"}),
        SimpleNamespace(digest="sha256:policy"),
        (),
        "sha256:snapshot",
        GovernedCorpusObservability(0, 0, {}, "sha256:snapshot", "sha256:policy", 4),
    )
    feature = await _feature(_Host(snapshot), tmp_path)
    manifest = {
        "policy_digest": "sha256:policy",
        "snapshot_hash": "sha256:snapshot",
        "semantic_checkpoint": {
            "tenant_id": "tenant:test", "generation": 4, "event_id": "event:4",
        },
        "capability_versions": {"semantic_maintenance": "1"},
    }
    assert feature._snapshot_pin_problem(manifest, snapshot, require_exact_snapshot=True) is None


def test_governed_core_dependency_is_source_pinned_and_exposes_its_contract():
    """Never resolve a released pre-capability core under a compatible-looking floor."""
    from kestrel_sovereign.knowledge import GovernedCorpusPolicy, GovernedCorpusSnapshot
    from kestrel_sovereign.storage.async_storage import AsyncStorage

    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    requirements = pyproject.read_text()
    assert "kestrel-sovereign @ git+https://github.com/KestrelSovereignAI/kestrel-sovereign.git@5932735a7db02228877985f66b9f1eb56d564f27" in requirements
    assert "kestrel-sovereign>=0.49.5" not in requirements
    assert GovernedCorpusPolicy is not None
    assert GovernedCorpusSnapshot is not None
    assert callable(getattr(AsyncStorage, "governed_assertion_corpus_snapshot", None))
    assert callable(getattr(AsyncStorage, "governed_assertion_corpus_changes_since", None))


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
    _stamp(first, candidate)
    await first._persist_config()
    restarted = await _feature(stale_host, tmp_path)
    restarted.agent.get_feature = MagicMock(return_value=restarted)
    await restarted.post_all_features_loaded(restarted.agent)
    assert restarted._active_adapter_path is None
    assert "lineage no longer current" in restarted._quarantined_adapters[str(candidate)]


async def test_restart_quarantines_when_persisted_adapter_receipt_is_lost(tmp_path):
    snapshot = _snapshot()
    candidate = tmp_path / "candidate"
    build_corpus(None, str(tmp_path / "corpus"), governed_snapshot=snapshot, manifest_dir=str(candidate))
    host = _Host(snapshot)
    first = await _feature(host, tmp_path)
    _stamp(first, candidate)
    first._active_adapter_path = str(candidate)
    await first._persist_config()
    # Simulate a partial persistence loss while the adapter directory remains.
    host.nodes[first._config_node_id()].properties["config"]["adapter_lineage"] = {}

    restarted = await _feature(host, tmp_path)
    restarted.agent.get_feature = MagicMock(return_value=restarted)
    await restarted.post_all_features_loaded(restarted.agent)
    assert restarted._active_adapter_path is None
    assert restarted._quarantined_adapters[str(candidate)] == (
        "durable adapter lineage receipt unavailable; rebuild required"
    )
