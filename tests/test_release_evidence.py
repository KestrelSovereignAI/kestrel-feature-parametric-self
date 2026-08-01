"""External #2753 evidence must exercise the real core value contracts."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
import types
from argparse import Namespace
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from kestrel_sovereign.knowledge.assertion import (
    IRI,
    Assertion,
    DirectLineage,
    EpistemicState,
    Literal,
    OntologyRef,
    SourceOccurrence,
    Visibility,
)
from kestrel_sovereign.knowledge.corpus import (
    CORPUS_SCHEMA_VERSION,
    CorpusCheckpoint,
    CorpusEligibilityDecision,
    CorpusEligibilityReason,
    CorpusValidationStatus,
    GovernedCorpusDelta,
    GovernedCorpusExample,
    GovernedCorpusObservability,
    GovernedCorpusPolicy,
    GovernedCorpusSnapshot,
    GovernedCorpusTombstone,
)
from kestrel_sovereign.knowledge.release_evidence import (
    apply_evidence_records,
    attach_external_capability_report,
    release_evidence_template,
)
from kestrel_sovereign.knowledge.release_evidence_execution import (
    CatalogSigningIdentity,
)
from kestrel_sovereign.knowledge.release_evidence_freshness import (
    ExternalFreshnessLedger,
)
from kestrel_sovereign.knowledge.release_evidence_models import (
    ExecutionSource,
    ExternalCapabilityReport,
    ReleaseEvidenceError,
    TrustedExecutionPolicy,
)
from kestrel_sovereign.knowledge.release_evidence_verifier import (
    combine_external_envelope_submission,
    load_external_envelope,
)
from kestrel_sovereign.knowledge.shacl_validation import (
    ValidationState,
    ValidationWriteAction,
)

from kestrel_feature_parametric_self import ParametricSelfFeature
from kestrel_feature_parametric_self import release_evidence as release_evidence_module
from kestrel_feature_parametric_self.release_evidence import (
    CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST,
    EXTERNAL_GATE_IDS,
    ExternalReleaseEvidenceError,
    KiteErasureBackend,
    ParametricSelfExternalEvidenceRunner,
    ParametricSelfKiteErasureHook,
)

_REAL_RUNNER_REVISION_RESOLVER = release_evidence_module._resolve_clean_evidence_runner_revision


@pytest.fixture(autouse=True)
def _stable_runner_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep integration fixtures independent of this worktree's edit state.

    Production resolves the clean checkout itself; dedicated unit tests below
    exercise that resolver.  During a source-tree test run the checkout is
    intentionally dirty, so its value cannot be an honest release identity.
    """
    monkeypatch.setattr(
        release_evidence_module,
        "_resolve_clean_evidence_runner_revision",
        lambda: "a" * 40,
    )


def _snapshot() -> GovernedCorpusSnapshot:
    ontology = OntologyRef(
        "https://example.test/ontology",
        "1.0.0",
        "sha256:evidence-ontology",
        "semantic-kb-v1",
    )
    assertion = Assertion(
        tenant_id="tenant-evidence",
        owning_agent_id="agent-evidence",
        subject=IRI("https://example.test/subject"),
        predicate=IRI("https://example.test/predicate"),
        object=Literal("governed lesson"),
        revision_id="revision-evidence",
        confidence="0.9",
        confidence_method="operator",
        confidence_basis="operator-attested",
        epistemic_state=EpistemicState.ASSERTED,
        asserted_at=datetime(2026, 7, 31, tzinfo=UTC),
        ontology_version=ontology,
        lineage=DirectLineage(("source-evidence",)),
        privacy_classification="normal",
        release_policy_reference="policy-training-v1",
        visibility=Visibility.PRIVATE,
    )
    policy = GovernedCorpusPolicy(
        policy_id="evidence-policy",
        policy_version="1",
        accepted_epistemic_states=(EpistemicState.ASSERTED,),
        accepted_visibility=(Visibility.PRIVATE,),
        accepted_privacy_classifications=("normal",),
        accepted_consent_references=("policy-training-v1",),
        accepted_grounding_classes=("operator-attested",),
        accepted_source_kinds=("operator-note",),
        accepted_ontology_pins=(ontology,),
        accepted_semantic_capability_versions=(("semantic_maintenance", "v3"),),
    )
    source = SourceOccurrence(
        "source-evidence",
        "operator-note",
        "evidence-source",
        datetime(2026, 7, 31, tzinfo=UTC),
    )
    validation = CorpusValidationStatus(ValidationState.CONFORMS, ValidationWriteAction.ACCEPT)
    example = GovernedCorpusExample(
        assertion,
        (source,),
        validation,
        CorpusEligibilityDecision(
            True, CorpusEligibilityReason.INCLUDED, policy.digest, ValidationState.CONFORMS
        ),
        "content-hash-evidence",
        "split-key-evidence",
    )
    checkpoint = CorpusCheckpoint("tenant-evidence", 4, "event-evidence")
    snapshot_hash = "snapshot-hash-evidence"
    return GovernedCorpusSnapshot(
        CORPUS_SCHEMA_VERSION,
        "tenant-evidence",
        checkpoint,
        {"semantic_maintenance": "v3"},
        policy,
        (example,),
        snapshot_hash,
        GovernedCorpusObservability(1, 1, {}, snapshot_hash, policy.digest, 4),
    )


class _CoreBackedErasureStorage:
    """Concrete core snapshot/delta values, never a mocked release contract."""

    def __init__(self, snapshot: GovernedCorpusSnapshot, *, backend: str = "sqlite") -> None:
        self.snapshot = snapshot
        self.backend_type = backend
        self.erased = False
        self.closed = False
        self.nodes: dict[str, object] = {}
        self.snapshot_requests: list[dict[str, object]] = []
        self.delta_requests: list[dict[str, object]] = []

    async def governed_assertion_corpus_snapshot(self, **kwargs):
        self.snapshot_requests.append(dict(kwargs))
        return self.snapshot

    async def governed_assertion_corpus_changes_since(self, snapshot, **kwargs):
        self.delta_requests.append(dict(kwargs))
        assert snapshot is self.snapshot
        tombstones = ()
        checkpoint = snapshot.checkpoint
        if self.erased:
            example = snapshot.examples[0]
            checkpoint = CorpusCheckpoint("tenant-evidence", 5, "event-erased")
            tombstones = (
                GovernedCorpusTombstone(
                    "event-erased",
                    example.assertion.assertion_id,
                    example.assertion.revision_id,
                    "deleted",
                    5,
                    "deleted",
                ),
            )
        return GovernedCorpusDelta(
            snapshot.checkpoint,
            checkpoint,
            (),
            tombstones,
            "delta-erased" if self.erased else "delta-clean",
            GovernedCorpusObservability(
                len(tombstones), 0, {},
                "delta-erased" if self.erased else "delta-clean",
                snapshot.policy.digest,
                checkpoint.generation,
            ),
        )

    async def erase_assertion(self, assertion_id, *, operation_id):
        assert assertion_id == self.snapshot.examples[0].assertion.assertion_id
        assert operation_id.startswith("parametric-self-release-erasure:")
        self.erased = True

    async def get_assertion(self, assertion_id, *, include_inactive=False):
        if (
            not self.erased
            and assertion_id == self.snapshot.examples[0].assertion.assertion_id
        ):
            return self.snapshot.examples[0].assertion
        return None

    async def add_node(self, node) -> None:
        self.nodes[node.node_id] = node

    async def get_node(self, node_id):
        return self.nodes.get(node_id)

    async def close(self) -> None:
        self.closed = True


async def _feature(
    storage: _CoreBackedErasureStorage,
    tmp_path: Path,
    *,
    agent_id: str = "agent-evidence",
    is_test_instance: bool = False,
) -> ParametricSelfFeature:
    agent = MagicMock()
    agent.storage = storage
    agent.storage_path = None
    agent.parametric_self_work_dir = str(tmp_path / "work")
    agent.parametric_self_governed_corpus_policy = storage.snapshot.policy
    agent.semantic_inference_profile = None
    agent.is_test_instance = is_test_instance
    agent.agent_id = agent_id
    agent.sleep_hooks = []
    feature = ParametricSelfFeature(agent=agent)
    await feature.initialize()
    feature._governed_corpus_policy = storage.snapshot.policy
    return feature


def _identity() -> CatalogSigningIdentity:
    return CatalogSigningIdentity(
        issuer_id="parametric_self_ci",
        key_id="release_evidence_key",
        private_key=Ed25519PrivateKey.from_private_bytes(b"\x08" * 32),
        source=ExecutionSource.EXTERNAL_CI,
    )


class _FakeDisposablePostgres:
    """Core-created database stand-in for producer mutation tests."""

    created: list["_FakeDisposablePostgres"] = []

    def __init__(self) -> None:
        self.dsn = "postgresql://core-created-disposable.invalid/release"
        self.closed = False

    @classmethod
    async def create(cls) -> "_FakeDisposablePostgres":
        instance = cls()
        cls.created.append(instance)
        return instance

    async def close(self) -> None:
        self.closed = True

    async def __aenter__(self) -> "_FakeDisposablePostgres":
        return self

    async def __aexit__(self, *_args) -> bool:
        await self.close()
        return False


@pytest.fixture(autouse=True)
def _fake_core_postgres_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeDisposablePostgres.created.clear()
    monkeypatch.setattr(release_evidence_module, "DisposablePostgresDatabase", _FakeDisposablePostgres)


def _dual_feature_factory(tmp_path: Path):
    """Build a fresh core-backed test feature for each runner-owned backend."""
    storages: dict[str, _CoreBackedErasureStorage] = {}

    async def make_feature(backend: KiteErasureBackend) -> ParametricSelfFeature:
        storage = _CoreBackedErasureStorage(_snapshot(), backend=backend.backend)
        # The production factory calls ``await backend.open_storage()``. Tests
        # install a concrete core-contract storage double without creating a
        # real database file, while retaining the exact identity check.
        backend._storage = storage
        storages[backend.backend] = storage
        return await _feature(
            storage,
            tmp_path / backend.backend,
            agent_id=backend.agent_id,
            is_test_instance=True,
        )

    return make_feature, storages


def _install_test_feature_builder(
    monkeypatch: pytest.MonkeyPatch,
    runner: ParametricSelfExternalEvidenceRunner,
    builder,
) -> None:
    """Keep storage doubles behind a private runner seam, never a caller API."""
    monkeypatch.setattr(runner, "_build_isolated_feature", builder)


async def test_external_evidence_runs_real_core_snapshot_to_quarantine_and_signs(tmp_path, monkeypatch):
    factory, storages = _dual_feature_factory(tmp_path)
    identity = _identity()
    ledger = ExternalFreshnessLedger(
        tmp_path / "verifier-freshness.sqlite", trusted_root=tmp_path
    )
    run_nonce = ledger.issue_challenge()

    runner = ParametricSelfExternalEvidenceRunner(identity)
    _install_test_feature_builder(monkeypatch, runner, factory)
    envelope = await runner.run(
        scratch_dir=tmp_path / "fresh-drill",
        trusted_scratch_root=tmp_path,
        run_nonce=run_nonce,
    )

    assert envelope.core_release_evidence_contract_digest == CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST
    assert tuple(record.gate_id for record in envelope.records) == EXTERNAL_GATE_IDS
    assert all(record.passed for record in envelope.records)
    assert all(record.observation == {"erased_count": 2, "remaining_count": 0} for record in envelope.records)
    assert set(storages) == {"sqlite", "postgres"}
    assert all(storage.erased and storage.closed for storage in storages.values())
    assert not (tmp_path / "fresh-drill-sqlite").exists()
    assert not (tmp_path / "fresh-drill-postgres").exists()
    assert _FakeDisposablePostgres.created and _FakeDisposablePostgres.created[-1].closed
    assert len(envelope.run_nonce) == 64
    assert envelope.run_nonce == run_nonce
    assert len(envelope.report.freshness_receipt) == 64
    assert tuple(item.gate_id for item in envelope.report.attestations) == EXTERNAL_GATE_IDS[1:]
    assert {record.external_run_nonce for record in envelope.records} == {run_nonce}
    assert len(envelope.evidence_runner_revision) == 40
    assert {record.external_evidence_runner_revision for record in envelope.records} == {
        envelope.evidence_runner_revision
    }

    policy = TrustedExecutionPolicy((identity.trusted_key(("external_ci",)),))
    evidence = apply_evidence_records(
        release_evidence_template(), envelope.records, trust_policy=policy
    )
    attached = attach_external_capability_report(
        evidence,
        envelope.report,
        freshness_ledger=ledger,
        expected_evidence_runner_revision=envelope.evidence_runner_revision,
    )
    assert attached.external_capabilities == (envelope.report,)
    # The atomic producer envelope is accepted at the core verifier boundary,
    # but it cannot make a release ready without every core record and budget.
    assert attached.ready is False
    with pytest.raises(ReleaseEvidenceError, match="already consumed"):
        attach_external_capability_report(
            evidence,
            envelope.report,
            freshness_ledger=ledger,
            expected_evidence_runner_revision=envelope.evidence_runner_revision,
        )

    rewrap_nonce = ledger.issue_challenge()
    rewrapped = ExternalCapabilityReport.attest(
        capability_id=envelope.report.capability_id,
        repository=envelope.report.repository,
        capability_source_revision=envelope.report.capability_source_revision,
        evidence_runner_revision=envelope.report.evidence_runner_revision,
        core_release_evidence_contract_digest=envelope.report.core_release_evidence_contract_digest,
        run_nonce=rewrap_nonce,
        attestations=envelope.report.attestations,
    )
    with pytest.raises(ReleaseEvidenceError, match="nonce"):
        attach_external_capability_report(
            evidence,
            rewrapped,
            freshness_ledger=ledger,
            expected_evidence_runner_revision=envelope.evidence_runner_revision,
        )

    unknown_factory, _ = _dual_feature_factory(tmp_path / "unknown")

    unknown_runner = ParametricSelfExternalEvidenceRunner(identity)
    _install_test_feature_builder(monkeypatch, unknown_runner, unknown_factory)
    unknown_envelope = await unknown_runner.run(
        scratch_dir=tmp_path / "unknown-drill",
        trusted_scratch_root=tmp_path,
        run_nonce="f" * 64,
    )
    unknown_evidence = apply_evidence_records(
        release_evidence_template(), unknown_envelope.records, trust_policy=policy
    )
    with pytest.raises(ReleaseEvidenceError, match="not an issued pending"):
        attach_external_capability_report(
            unknown_evidence,
            unknown_envelope.report,
            freshness_ledger=ledger,
            expected_evidence_runner_revision=unknown_envelope.evidence_runner_revision,
        )

    output = tmp_path / "external-evidence.json"
    envelope.write(output)
    loaded_envelope = load_external_envelope(output)
    assembled_records, assembled_report = combine_external_envelope_submission(
        records=(), envelope=loaded_envelope
    )
    assert assembled_records == envelope.records
    assert assembled_report == envelope.report
    payload = json.loads(output.read_text())
    rendered = json.dumps(payload)
    assert payload["trust_status"] == "external_signature_requires_core_policy_verification"
    assert "governed lesson" not in rendered
    assert "tenant-evidence" not in rendered
    assert str(tmp_path) not in rendered


async def test_kite_sqlite_backend_opens_real_storage_and_physically_erases(tmp_path, monkeypatch):
    """Exercise the runner-owned SQLite authority without a storage double."""
    from kestrel_sovereign.knowledge import InferenceProfile
    from kestrel_sovereign.privacy import PrivacyMode
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

    trusted_root = tmp_path / "trusted-scratch"
    trusted_root.mkdir(mode=0o700)
    backend = KiteErasureBackend(
        "sqlite", trusted_root, trusted_root / "sqlite-live-drill"
    )
    physical_erasure: dict[str, object] = {}

    async def factory(owned_backend: KiteErasureBackend) -> ParametricSelfFeature:
        assert owned_backend is backend
        raw = await owned_backend.open_storage()
        governed = PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL)
        saved = await governed.save_explicit_fact(
            subject="user",
            predicate="preferred_deploy_region",
            value="private e2e value",
            confidence=0.9,
            invocation_id="release-evidence-e2e",
        )
        assert saved.saved
        profile = InferenceProfile(
            OntologyRef(
                "http://www.w3.org/2000/01/rdf-schema#",
                "1.0.0",
                "e362812917fddab7cfab3dc35553ad292725e8f264e05f376077340e91034db5",
                "semantic-kb-v1",
            ),
            "1.0.0",
        )
        await raw.run_semantic_maintenance(profile)
        assertion = (await raw.assertion_inference_inputs())[0]
        capabilities = await raw.semantic_maintenance_capability_versions(profile)
        policy = GovernedCorpusPolicy(
            policy_id="release-evidence-e2e",
            policy_version="1",
            accepted_epistemic_states=(EpistemicState.REPORTED,),
            accepted_visibility=(Visibility.PRIVATE,),
            accepted_privacy_classifications=("normal",),
            accepted_consent_references=("policy:privacy:normal-v1",),
            accepted_grounding_classes=("explicit-tool-invocation",),
            accepted_source_kinds=("agent_tool_invocation",),
            accepted_ontology_pins=(assertion.ontology_version,),
            accepted_semantic_capability_versions=tuple(capabilities.items()),
        )
        original_erase = raw.erase_assertion

        async def erase_and_verify(assertion_id: str, *, operation_id: str | None) -> None:
            await original_erase(assertion_id, operation_id=operation_id)
            physical_erasure["assertion_id"] = assertion_id
            physical_erasure["canonical_row"] = await raw.get_assertion(
                assertion_id, include_inactive=True
            )
            physical_erasure["row_count"] = await raw.db.fetchval(
                "SELECT COUNT(*) FROM semantic_assertions WHERE assertion_id = ?",
                (assertion_id,),
            )
            physical_erasure["tombstone_count"] = await raw.db.fetchval(
                "SELECT COUNT(*) FROM semantic_assertion_erased_operation_tombstones"
            )

        raw.erase_assertion = erase_and_verify
        agent = types.SimpleNamespace(
            storage=raw,
            storage_path=None,
            parametric_self_work_dir=str(tmp_path / "real-feature-work"),
            parametric_self_governed_corpus_policy=policy,
            semantic_inference_profile=profile,
            is_test_instance=True,
            agent_id=owned_backend.agent_id,
            sleep_hooks=[],
        )
        feature = ParametricSelfFeature(agent=agent)
        await feature.initialize()
        feature._governed_corpus_policy = policy
        return feature

    runner = ParametricSelfExternalEvidenceRunner(_identity())
    _install_test_feature_builder(monkeypatch, runner, factory)
    observation = await runner._run_backend(backend, run_nonce="c" * 64)

    assert observation.observation == {"erased_count": 1, "remaining_count": 0}
    assert physical_erasure["canonical_row"] is None
    assert physical_erasure["row_count"] == 0
    assert physical_erasure["tombstone_count"] == 1
    state_path = backend._sqlite_state_path
    assert backend._closed
    assert state_path is not None and not state_path.exists()
    assert not backend.scratch_dir.exists()
    # AsyncStorage's destructive-audit sidecar and SQLite WAL/SHM files must
    # stay beneath the exact owned state directory and leave no trusted-root
    # residue after the runner-owned close.
    assert list(trusted_root.iterdir()) == []


async def test_kite_sqlite_cleanup_retries_the_exact_owned_directory(tmp_path, monkeypatch):
    trusted_root = tmp_path / "trusted-retry"
    trusted_root.mkdir(mode=0o700)
    backend = KiteErasureBackend("sqlite", trusted_root, trusted_root / "unused-drill")
    await backend.open_storage()
    state_dir = backend._sqlite_state_path.parent
    original_cleanup = release_evidence_module._cleanup_private_scratch_tree
    attempts = 0

    def fail_once(path, identities) -> None:
        nonlocal attempts
        if Path(path) == state_dir and attempts == 0:
            attempts += 1
            raise ExternalReleaseEvidenceError("simulated owned-directory cleanup failure")
        original_cleanup(path, identities)

    monkeypatch.setattr(release_evidence_module, "_cleanup_private_scratch_tree", fail_once)
    with pytest.raises(release_evidence_module._ExternalCleanupFailure):
        await backend.close()
    assert state_dir.exists()

    await backend.close()
    assert attempts == 1
    assert list(trusted_root.iterdir()) == []


async def test_external_evidence_fails_closed_when_erasure_does_not_change_core_delta(tmp_path, monkeypatch):
    factory, storages = _dual_feature_factory(tmp_path)
    runner = ParametricSelfExternalEvidenceRunner(_identity())
    _install_test_feature_builder(monkeypatch, runner, factory)

    async def no_op_erase(*_args) -> None:
        return None

    monkeypatch.setattr(runner, "_erase_snapshot_assertion", no_op_erase)

    with pytest.raises(ExternalReleaseEvidenceError, match="did not remove"):
        await runner.run(
            scratch_dir=tmp_path / "fresh-drill",
            trusted_scratch_root=tmp_path,
            run_nonce="a" * 64,
        )
    # Failed observation fails closed: leaving a candidate marked served after
    # a purported erasure would be a more dangerous state than quarantining it.
    assert set(storages) == {"sqlite"}
    assert storages["sqlite"].closed
    assert not (tmp_path / "fresh-drill-sqlite").exists()
    assert _FakeDisposablePostgres.created[-1].closed


async def test_cancellation_during_prepare_cleans_plaintext_and_preserves_cancellation(
    tmp_path, monkeypatch
):
    base_factory, storages = _dual_feature_factory(tmp_path)
    runner = ParametricSelfExternalEvidenceRunner(_identity())
    sealed = False

    async def cancelled_prepare(backend: KiteErasureBackend):
        feature = await base_factory(backend)

        async def cancel_snapshot(**_kwargs):
            raise asyncio.CancelledError()

        feature._request_governed_snapshot = cancel_snapshot
        return feature

    def refuse_if_sealed(*_args, **_kwargs):
        nonlocal sealed
        sealed = True
        raise AssertionError("cancellation must not sign evidence")

    _install_test_feature_builder(monkeypatch, runner, cancelled_prepare)
    monkeypatch.setattr(runner, "_seal_envelope", refuse_if_sealed)
    with pytest.raises(asyncio.CancelledError):
        await runner.run(
            scratch_dir=tmp_path / "cancel-prepare",
            trusted_scratch_root=tmp_path,
            run_nonce="a" * 64,
        )

    assert storages["sqlite"].closed
    assert not (tmp_path / "cancel-prepare-sqlite").exists()
    assert _FakeDisposablePostgres.created[-1].closed
    assert not sealed


async def test_cancellation_between_prepare_and_observe_cleans_and_preserves_cancellation(
    tmp_path, monkeypatch
):
    factory, storages = _dual_feature_factory(tmp_path)
    runner = ParametricSelfExternalEvidenceRunner(_identity())

    async def cancel_erase(*_args) -> None:
        raise asyncio.CancelledError()

    _install_test_feature_builder(monkeypatch, runner, factory)
    monkeypatch.setattr(runner, "_erase_snapshot_assertion", cancel_erase)
    monkeypatch.setattr(
        runner, "_seal_envelope", lambda *_args, **_kwargs: pytest.fail("must not sign")
    )
    with pytest.raises(asyncio.CancelledError):
        await runner.run(
            scratch_dir=tmp_path / "cancel-erase",
            trusted_scratch_root=tmp_path,
            run_nonce="b" * 64,
        )

    assert storages["sqlite"].closed
    assert not (tmp_path / "cancel-erase-sqlite").exists()
    assert _FakeDisposablePostgres.created[-1].closed


async def test_cancellation_during_observe_cleans_and_preserves_cancellation(
    tmp_path, monkeypatch
):
    base_factory, storages = _dual_feature_factory(tmp_path)
    runner = ParametricSelfExternalEvidenceRunner(_identity())

    async def cancelled_observe(backend: KiteErasureBackend):
        feature = await base_factory(backend)
        original_verify = feature._verify_adapter_lineage

        async def verify(candidate_path, **kwargs):
            if not kwargs.get("before_promotion", False):
                raise asyncio.CancelledError()
            return await original_verify(candidate_path, **kwargs)

        feature._verify_adapter_lineage = verify
        return feature

    _install_test_feature_builder(monkeypatch, runner, cancelled_observe)
    monkeypatch.setattr(
        runner, "_seal_envelope", lambda *_args, **_kwargs: pytest.fail("must not sign")
    )
    with pytest.raises(asyncio.CancelledError):
        await runner.run(
            scratch_dir=tmp_path / "cancel-observe",
            trusted_scratch_root=tmp_path,
            run_nonce="c" * 64,
        )

    assert storages["sqlite"].closed
    assert not (tmp_path / "cancel-observe-sqlite").exists()
    assert _FakeDisposablePostgres.created[-1].closed


async def test_storage_close_cancellation_still_removes_owned_sqlite_residue(tmp_path):
    trusted_root = tmp_path / "trusted-cancel-close"
    trusted_root.mkdir(mode=0o700)
    backend = KiteErasureBackend("sqlite", trusted_root, trusted_root / "unused-drill")
    storage = await backend.open_storage()
    original_close = storage.close

    async def cancel_close() -> None:
        raise asyncio.CancelledError()

    storage.close = cancel_close
    with pytest.raises(asyncio.CancelledError):
        await backend.close()

    assert list(trusted_root.iterdir()) == []
    await original_close()


async def test_cancellation_and_cleanup_failure_are_aggregated_without_signing(
    tmp_path, monkeypatch
):
    factory, _ = _dual_feature_factory(tmp_path)
    runner = ParametricSelfExternalEvidenceRunner(_identity())
    original_close = KiteErasureBackend.close
    sealed = False

    async def cancel_erase(*_args) -> None:
        raise asyncio.CancelledError()

    async def fail_close(backend: KiteErasureBackend) -> None:
        await original_close(backend)
        if backend.backend == "sqlite":
            raise ExternalReleaseEvidenceError("simulated cleanup failure")

    def refuse_if_sealed(*_args, **_kwargs):
        nonlocal sealed
        sealed = True
        raise AssertionError("cancellation must not sign evidence")

    _install_test_feature_builder(monkeypatch, runner, factory)
    monkeypatch.setattr(runner, "_erase_snapshot_assertion", cancel_erase)
    monkeypatch.setattr(KiteErasureBackend, "close", fail_close)
    monkeypatch.setattr(runner, "_seal_envelope", refuse_if_sealed)
    with pytest.raises(BaseExceptionGroup) as failed:
        await runner.run(
            scratch_dir=tmp_path / "cancel-combined",
            trusted_scratch_root=tmp_path,
            run_nonce="d" * 64,
        )

    def flatten(error):
        if isinstance(error, BaseExceptionGroup):
            return [item for child in error.exceptions for item in flatten(child)]
        return [error]

    assert any(isinstance(error, asyncio.CancelledError) for error in flatten(failed.value))
    assert any(
        isinstance(error, release_evidence_module._ExternalCleanupFailure)
        for error in flatten(failed.value)
    )
    assert not sealed


@pytest.mark.parametrize("interrupt", (KeyboardInterrupt, SystemExit))
async def test_interrupts_rethrow_unchanged_after_cleanup(tmp_path, monkeypatch, interrupt):
    factory, storages = _dual_feature_factory(tmp_path)
    runner = ParametricSelfExternalEvidenceRunner(_identity())

    async def interrupt_erase(*_args) -> None:
        raise interrupt()

    _install_test_feature_builder(monkeypatch, runner, factory)
    monkeypatch.setattr(runner, "_erase_snapshot_assertion", interrupt_erase)
    monkeypatch.setattr(
        runner, "_seal_envelope", lambda *_args, **_kwargs: pytest.fail("must not sign")
    )
    with pytest.raises(interrupt):
        await runner.run(
            scratch_dir=tmp_path / f"interrupt-{interrupt.__name__}",
            trusted_scratch_root=tmp_path,
            run_nonce="e" * 64,
        )

    assert storages["sqlite"].closed
    assert _FakeDisposablePostgres.created[-1].closed
    assert not (tmp_path / f"interrupt-{interrupt.__name__}-sqlite").exists()


@pytest.mark.parametrize("interrupt", (asyncio.CancelledError, KeyboardInterrupt, SystemExit))
def test_cleanup_aggregation_keeps_base_exception_leaves_at_top_level(interrupt):
    execution = RuntimeError("execution failure")
    interrupted_cleanup = interrupt()
    ordinary_cleanup_one = OSError("ordinary cleanup one")
    ordinary_cleanup_two = ValueError("ordinary cleanup two")

    with pytest.raises(BaseExceptionGroup) as raised:
        release_evidence_module._raise_execution_or_cleanup_failure(
            execution,
            [
                interrupted_cleanup,
                ordinary_cleanup_one,
                ordinary_cleanup_two,
            ],
        )

    group = raised.value
    assert not isinstance(group, Exception)
    assert group.exceptions[0] is execution
    assert group.exceptions[1] is interrupted_cleanup
    assert isinstance(group.exceptions[2], release_evidence_module._ExternalCleanupFailure)
    assert group.exceptions[2].failures == (
        ordinary_cleanup_one,
        ordinary_cleanup_two,
    )


async def test_external_evidence_refuses_a_skipped_or_failing_backend_and_cleans_up(tmp_path, monkeypatch):
    base_factory, storages = _dual_feature_factory(tmp_path)

    async def failing_postgres(backend: KiteErasureBackend):
        if backend.backend == "postgres":
            raise RuntimeError("backend failed")
        return await base_factory(backend)

    runner = ParametricSelfExternalEvidenceRunner(_identity())
    _install_test_feature_builder(monkeypatch, runner, failing_postgres)
    with pytest.raises(RuntimeError, match="backend failed"):
        await runner.run(
            scratch_dir=tmp_path / "failing-drill",
            trusted_scratch_root=tmp_path,
            run_nonce="e" * 64,
        )
    assert set(storages) == {"sqlite"}
    assert storages["sqlite"].closed
    assert _FakeDisposablePostgres.created[-1].closed


async def test_external_evidence_aggregates_all_backend_cleanup_failures(
    tmp_path, monkeypatch
):
    """One backend's cleanup failure must not skip the other backend's close."""
    runner = ParametricSelfExternalEvidenceRunner(_identity())
    closed: list[str] = []

    async def bypass_drill(*_args, **_kwargs) -> object:
        return object()

    async def failing_backend_close(backend: KiteErasureBackend) -> None:
        closed.append(backend.backend)
        raise ExternalReleaseEvidenceError(f"simulated {backend.backend} cleanup failure")

    monkeypatch.setattr(runner, "_run_backend", bypass_drill)
    monkeypatch.setattr(KiteErasureBackend, "close", failing_backend_close)

    with pytest.raises(release_evidence_module._ExternalCleanupFailure) as refused:
        await runner.run(
            scratch_dir=tmp_path / "cleanup-order-drill",
            trusted_scratch_root=tmp_path,
            run_nonce="0" * 64,
        )

    assert closed == ["sqlite", "postgres"]
    assert len(refused.value.failures) == 2
    assert _FakeDisposablePostgres.created[-1].closed


async def test_external_evidence_rejects_backend_substitution(tmp_path, monkeypatch):
    base_factory, _ = _dual_feature_factory(tmp_path)

    async def substituted(backend: KiteErasureBackend):
        feature = await base_factory(backend)
        if backend.backend == "postgres":
            feature.agent.storage.backend_type = "sqlite"
        return feature

    runner = ParametricSelfExternalEvidenceRunner(_identity())
    _install_test_feature_builder(monkeypatch, runner, substituted)
    with pytest.raises(ExternalReleaseEvidenceError, match="backend does not match"):
        await runner.run(
            scratch_dir=tmp_path / "substituted-drill",
            trusted_scratch_root=tmp_path,
            run_nonce="f" * 64,
        )
    assert _FakeDisposablePostgres.created[-1].closed


async def test_external_evidence_rejects_still_active_serving(tmp_path, monkeypatch):
    base_factory, _ = _dual_feature_factory(tmp_path / "active")

    async def still_serving(backend: KiteErasureBackend):
        feature = await base_factory(backend)
        original = feature._verify_adapter_lineage

        async def verify(candidate_path, **kwargs):
            result = await original(candidate_path, **kwargs)
            if not kwargs.get("before_promotion", False) and result:
                feature._active_adapter_path = candidate_path
            return result

        feature._verify_adapter_lineage = verify
        return feature

    active_runner = ParametricSelfExternalEvidenceRunner(_identity())
    _install_test_feature_builder(monkeypatch, active_runner, still_serving)
    with pytest.raises(ExternalReleaseEvidenceError, match="did not reject served"):
        await active_runner.run(
            scratch_dir=tmp_path / "active-drill",
            trusted_scratch_root=tmp_path,
            run_nonce="8" * 64,
        )


async def test_external_evidence_rejects_actual_erasure_of_a_second_assertion(
    tmp_path, monkeypatch
):
    """A real tombstone for a decoy assertion cannot stand in for the target."""
    from kestrel_sovereign.knowledge.assertion import DirectLineage, Literal, SourceOccurrence

    trusted_root = tmp_path / "trusted"
    trusted_root.mkdir(mode=0o700)
    backend = KiteErasureBackend("sqlite", trusted_root, trusted_root / "second-assertion")
    runner = ParametricSelfExternalEvidenceRunner(_identity())
    original_builder = runner._build_isolated_feature
    erased: dict[str, object] = {}

    async def two_assertion_builder(owned_backend: KiteErasureBackend):
        feature = await original_builder(owned_backend)
        raw = feature.agent.storage
        primary = (await raw.assertion_inference_inputs())[0]
        source = SourceOccurrence(
            "release-evidence-decoy-source",
            "agent_tool_invocation",
            "release-evidence-decoy",
            datetime(2026, 8, 1, tzinfo=UTC),
        )
        decoy = replace(
            primary,
            object=Literal("release-evidence-decoy"),
            revision_id="release-evidence-decoy-revision",
            asserted_at=datetime(2026, 8, 1, tzinfo=UTC),
            lineage=DirectLineage((source.source_occurrence_id,)),
            assertion_id=None,
        )
        result = await raw.put_assertion(
            decoy,
            source_occurrences=(source,),
            operation_id="release-evidence-decoy-operation",
        )
        assert result.accepted
        await raw.run_semantic_maintenance(feature.agent.semantic_inference_profile)
        return feature

    async def erase_only_decoy(feature, snapshot, run_nonce) -> None:
        target_id = snapshot.examples[0].assertion.assertion_id
        inputs = await feature.agent.storage.assertion_inference_inputs()
        decoy = next(item for item in inputs if item.assertion_id != target_id)
        await feature.agent.storage.erase_assertion(
            decoy.assertion_id,
            operation_id=f"parametric-self-release-erasure:{run_nonce}",
        )
        erased["decoy_id"] = decoy.assertion_id
        erased["target_id"] = target_id
        erased["decoy_missing"] = await feature.agent.storage.get_assertion(
            decoy.assertion_id, include_inactive=True
        )
        erased["tombstone_count"] = await feature.agent.storage.db.fetchval(
            "SELECT COUNT(*) FROM semantic_assertion_erased_operation_tombstones"
        )

    _install_test_feature_builder(monkeypatch, runner, two_assertion_builder)
    monkeypatch.setattr(runner, "_erase_snapshot_assertion", erase_only_decoy)
    with pytest.raises(ExternalReleaseEvidenceError, match="did not remove"):
        await runner._run_backend(backend, run_nonce="9" * 64)

    assert erased["decoy_id"] != erased["target_id"]
    assert erased["decoy_missing"] is None
    assert erased["tombstone_count"] == 1
    assert list(trusted_root.iterdir()) == []


async def test_external_evidence_rejects_mixed_nonce_revision_and_artifact(tmp_path, monkeypatch):
    factory, _ = _dual_feature_factory(tmp_path)
    runner = ParametricSelfExternalEvidenceRunner(_identity())
    _install_test_feature_builder(monkeypatch, runner, factory)
    envelope = await runner.run(
        scratch_dir=tmp_path / "mixed-drill",
        trusted_scratch_root=tmp_path,
        run_nonce="7" * 64,
    )
    with pytest.raises(ExternalReleaseEvidenceError):
        replace(envelope, run_nonce="6" * 64)
    with pytest.raises(ExternalReleaseEvidenceError):
        replace(envelope, evidence_runner_revision="b" * 40)
    with pytest.raises(ExternalReleaseEvidenceError):
        replace(
            envelope,
            records=(
                envelope.records[0],
                replace(envelope.records[1], artifact=envelope.records[0].artifact),
                *envelope.records[2:],
            ),
        )


def test_cli_rejects_caller_postgres_dsn_injection() -> None:
    parser = release_evidence_module._build_cli_parser()
    with pytest.raises(SystemExit, match="2"):
        parser.parse_args(["--postgres-dsn", "postgresql://caller.invalid/prod"])


async def test_kite_hook_proves_pre_erase_eligibility_then_observes_server_erasure(tmp_path):
    storage = _CoreBackedErasureStorage(_snapshot())
    feature = await _feature(storage, tmp_path)
    hook = ParametricSelfKiteErasureHook(ParametricSelfExternalEvidenceRunner(_identity()))

    prepared = await hook.prepare(
        feature,
        scratch_dir=tmp_path / "two-phase-drill",
        trusted_scratch_root=tmp_path,
        run_nonce="1" * 64,
    )
    candidate_path = feature._active_adapter_path
    assert candidate_path is not None
    assert feature._adapter_lineage[candidate_path]["candidate_eligibility"] == "accepted"
    assert feature._adapter_lineage[candidate_path]["served_eligibility"] == "accepted"

    # The server-owned hook selects the precise assertion from the snapshot;
    # neither the test nor a CLI argument supplies an assertion identifier.
    await hook.erase_prepared_assertion(prepared)
    # A caller cannot replace the correlated base with a later/empty cache
    # between the server erase and the observation.
    feature._live_corpus_snapshot = None
    observation = await hook.observe(prepared, backend="sqlite")

    assert storage.erased is True
    assert observation.observation == {"erased_count": 1, "remaining_count": 0}
    assert feature._active_adapter_path is None
    with pytest.raises(ExternalReleaseEvidenceError, match="already consumed"):
        await hook.observe(prepared, backend="sqlite")


async def test_kite_hook_retains_original_consumer_identity_after_substitution_attempt(tmp_path):
    storage = _CoreBackedErasureStorage(_snapshot())
    feature = await _feature(storage, tmp_path)
    identity = _identity()
    hook = ParametricSelfKiteErasureHook(ParametricSelfExternalEvidenceRunner(identity))

    prepared = await hook.prepare(
        feature,
        scratch_dir=tmp_path / "identity-bound-drill",
        trusted_scratch_root=tmp_path,
        run_nonce="4" * 64,
    )
    feature._governed_artifact_consumer = {
        "consumer_id": "substituted-consumer",
        "consumer_key_id": "substituted-key",
        "consumer_public_key": "f" * 64,
        "retention_seconds": 1.0,
    }

    await hook.erase_prepared_assertion(prepared)
    observation = await hook.observe(prepared, backend="sqlite")

    assert observation.observation == {"erased_count": 1, "remaining_count": 0}
    registrations = storage.snapshot_requests + storage.delta_requests
    assert len(registrations) >= 3
    assert {request["consumer_id"] for request in registrations} == {
        identity.issuer_id
    }
    assert {request["consumer_key_id"] for request in registrations} == {
        identity.key_id
    }
    assert {request["consumer_public_key"] for request in registrations} == {
        identity.public_key
    }
    assert all(request["artifact_id"] for request in registrations)


async def test_external_evidence_cli_owns_feature_construction_and_runs_two_phases(tmp_path, monkeypatch):
    factory, storages = _dual_feature_factory(tmp_path)
    monkeypatch.setattr(
        ParametricSelfExternalEvidenceRunner,
        "_build_isolated_feature",
        lambda _runner, backend: factory(backend),
    )
    key_file = tmp_path / "external-ci.key"
    key_file.write_text("08" * 32, encoding="ascii")
    key_file.chmod(0o600)
    output = tmp_path / "external-evidence.json"
    args = Namespace(
        signing_key_file=key_file,
        issuer_id="parametric_self_ci",
        key_id="release_evidence_key",
        run_nonce="2" * 64,
        scratch_dir=tmp_path / "cli-drill",
        trusted_scratch_root=tmp_path,
        output=output,
    )

    envelope = await release_evidence_module._run_cli(args)

    assert set(storages) == {"sqlite", "postgres"}
    assert all(storage.erased and storage.closed for storage in storages.values())
    assert output.exists()
    assert output.stat().st_mode & 0o777 == 0o600
    assert json.loads(output.read_text())["run_nonce"] == envelope.run_nonce


def test_cli_rejects_caller_selected_executable_factory_surface() -> None:
    parser = release_evidence_module._build_cli_parser()
    assert "feature_factory" not in {action.dest for action in parser._actions}
    with pytest.raises(SystemExit, match="2"):
        parser.parse_args(["--feature-factory", "attacker.module:erase_everything"])


def test_runner_exposes_no_callback_or_dsn_mutation_surface() -> None:
    import inspect

    run_parameters = inspect.signature(ParametricSelfExternalEvidenceRunner.run).parameters
    assert set(run_parameters) == {
        "self", "scratch_dir", "trusted_scratch_root", "run_nonce"
    }
    assert "_load_kite_feature_factory" not in vars(release_evidence_module)
    assert set(inspect.signature(KiteErasureBackend.open_storage).parameters) == {"self"}


async def test_atomic_publication_unlinks_post_link_fsync_failure_and_allows_retry(
    tmp_path, monkeypatch
):
    factory, _ = _dual_feature_factory(tmp_path)
    runner = ParametricSelfExternalEvidenceRunner(_identity())
    _install_test_feature_builder(monkeypatch, runner, factory)
    envelope = await runner.run(
        scratch_dir=tmp_path / "atomic-drill",
        trusted_scratch_root=tmp_path,
        run_nonce="3" * 64,
    )
    output = tmp_path / "atomic-evidence.json"
    parent_identity = (output.parent.stat().st_dev, output.parent.stat().st_ino)
    original_fsync = os.fsync

    def fail_only_parent_fsync(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) == parent_identity:
            raise OSError("simulated post-link directory fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_only_parent_fsync)
    with pytest.raises(ExternalReleaseEvidenceError, match="could not be published"):
        release_evidence_module._write_cli_envelope_atomic(
            envelope, output, parent_identity
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".atomic-evidence.json.tmp-*"))

    monkeypatch.setattr(os, "fsync", original_fsync)
    release_evidence_module._write_cli_envelope_atomic(envelope, output, parent_identity)
    assert output.exists()


@pytest.mark.parametrize(
    "failure", ("core_workload", "factory_setup", "factory_forged_refusal")
)
async def test_cli_redacts_infrastructure_and_factory_setup_failures(
    tmp_path, monkeypatch, failure
):
    """The CLI must not reveal DSNs or paths from dependencies it invokes."""
    from kestrel_sovereign.knowledge.release_evidence_execution import (
        CatalogWorkloadUnavailable,
    )

    secret = "postgresql://release-user:secret@private.invalid/release /private/state.db"

    async def unsafe_factory(_backend: KiteErasureBackend) -> ParametricSelfFeature:
        if failure == "factory_forged_refusal":
            raise ExternalReleaseEvidenceError(secret)
        raise RuntimeError(secret)

    monkeypatch.setattr(
        ParametricSelfExternalEvidenceRunner,
        "_build_isolated_feature",
        lambda _runner, backend: unsafe_factory(backend),
    )
    if failure == "core_workload":

        class UnavailableCoreWorkload:
            @classmethod
            async def create(cls):
                raise CatalogWorkloadUnavailable(secret)

        monkeypatch.setattr(
            release_evidence_module, "DisposablePostgresDatabase", UnavailableCoreWorkload
        )

    key_file = tmp_path / "external-ci.key"
    key_file.write_text("08" * 32, encoding="ascii")
    key_file.chmod(0o600)
    output = tmp_path / "external-evidence.json"
    args = Namespace(
        signing_key_file=key_file,
        issuer_id="parametric_self_ci",
        key_id="release_evidence_key",
        run_nonce="d" * 64,
        scratch_dir=tmp_path / "redaction-drill",
        trusted_scratch_root=tmp_path,
        output=output,
    )

    with pytest.raises(ExternalReleaseEvidenceError) as refused:
        await release_evidence_module._run_cli(args)

    assert str(refused.value) == "external evidence execution is unavailable"
    assert secret not in str(refused.value)
    assert str(tmp_path) not in str(refused.value)
    assert not output.exists()
    assert not (tmp_path / "redaction-drill").exists()


@pytest.mark.parametrize("interrupt", (asyncio.CancelledError, KeyboardInterrupt, SystemExit))
async def test_cli_redacts_ordinary_group_leaves_while_preserving_interrupts(
    tmp_path, monkeypatch, interrupt
):
    secret = "postgresql://release-user:secret@private.invalid/release /private/state.db"
    interrupt_leaf = interrupt()
    raw_execution = RuntimeError(secret)
    raw_cleanup = OSError(secret)

    async def mixed_failure(_runner, **_kwargs):
        nested_ordinary_failures = ExceptionGroup(
            "raw nested details",
            [raw_execution, release_evidence_module._ExternalCleanupFailure([raw_cleanup])],
        )
        raise BaseExceptionGroup(
            "raw mixed details", [nested_ordinary_failures, interrupt_leaf]
        )

    monkeypatch.setattr(ParametricSelfExternalEvidenceRunner, "run", mixed_failure)
    key_file = tmp_path / "external-ci.key"
    key_file.write_text("08" * 32, encoding="ascii")
    key_file.chmod(0o600)
    output = tmp_path / "external-evidence.json"
    args = Namespace(
        signing_key_file=key_file,
        issuer_id="parametric_self_ci",
        key_id="release_evidence_key",
        run_nonce="d" * 64,
        scratch_dir=tmp_path / "redaction-drill",
        trusted_scratch_root=tmp_path,
        output=output,
    )

    with pytest.raises(BaseExceptionGroup) as escaped:
        await release_evidence_module._run_cli(args)

    group = escaped.value
    assert not isinstance(group, Exception)
    assert group.exceptions[0] is interrupt_leaf
    assert isinstance(group.exceptions[1], ExternalReleaseEvidenceError)
    assert str(group.exceptions[1]) == "external evidence execution is unavailable"
    rendered = "".join(traceback.format_exception(group))
    assert secret not in rendered
    assert str(tmp_path) not in rendered
    assert not output.exists()


@pytest.mark.parametrize(
    "output_case",
    ("relative", "existing", "existing_non_private", "missing_parent", "non_private_parent"),
)
async def test_cli_rejects_unsafe_output_before_physical_erasure(
    tmp_path, monkeypatch, output_case
):
    factory, storages = _dual_feature_factory(tmp_path)
    monkeypatch.setattr(
        ParametricSelfExternalEvidenceRunner,
        "_build_isolated_feature",
        lambda _runner, backend: factory(backend),
    )
    private_parent = tmp_path / "private-output"
    private_parent.mkdir(mode=0o700)
    if output_case == "relative":
        output = Path("relative-external-evidence.json")
    elif output_case in {"existing", "existing_non_private"}:
        output = private_parent / "existing.json"
        output.write_text("occupied", encoding="utf-8")
        output.chmod(0o644 if output_case == "existing_non_private" else 0o600)
    elif output_case == "missing_parent":
        output = tmp_path / "missing" / "evidence.json"
    else:
        non_private_parent = tmp_path / "shared-output"
        non_private_parent.mkdir(mode=0o755)
        output = non_private_parent / "evidence.json"

    args = Namespace(
        signing_key_file=tmp_path / "unused.key",
        issuer_id="parametric_self_ci",
        key_id="release_evidence_key",
        run_nonce="3" * 64,
        scratch_dir=tmp_path / "must-not-exist",
        trusted_scratch_root=tmp_path,
        output=output,
    )

    with pytest.raises(ExternalReleaseEvidenceError, match="output"):
        await release_evidence_module._run_cli(args)

    assert not storages
    assert not (tmp_path / "must-not-exist").exists()


async def test_external_evidence_scratch_tree_is_private_while_plaintext_is_live(tmp_path, monkeypatch):
    base_factory, _ = _dual_feature_factory(tmp_path)
    trusted_root = tmp_path / "trusted-scratch"
    trusted_root.mkdir(mode=0o700)
    scratch_dir = trusted_root / "drill"

    async def factory(backend: KiteErasureBackend):
        feature = await base_factory(backend)
        storage = feature.agent.storage
        original = storage.erase_assertion

        async def erase(*args, **kwargs) -> None:
            for directory in (
                backend.scratch_dir,
                backend.scratch_dir / "candidate",
                backend.scratch_dir / "corpus",
            ):
                metadata = directory.stat()
                assert metadata.st_uid == os.geteuid()
                assert metadata.st_mode & 0o777 == 0o700
            for path in backend.scratch_dir.rglob("*"):
                if path.is_file():
                    assert path.stat().st_mode & 0o777 == 0o600
            await original(*args, **kwargs)

        storage.erase_assertion = erase
        return feature

    runner = ParametricSelfExternalEvidenceRunner(_identity())
    _install_test_feature_builder(monkeypatch, runner, factory)
    await runner.run(
        scratch_dir=scratch_dir,
        trusted_scratch_root=trusted_root,
        run_nonce="b" * 64,
    )
    assert not (trusted_root / "drill-sqlite").exists()
    assert not (trusted_root / "drill-postgres").exists()


async def test_external_evidence_refuses_unsafe_or_reused_scratch_paths(tmp_path):
    trusted_root = tmp_path / "trusted-scratch"
    trusted_root.mkdir(mode=0o700)
    unsafe_root = tmp_path / "unsafe-scratch"
    unsafe_root.mkdir(mode=0o755)

    with pytest.raises(ExternalReleaseEvidenceError, match="not private"):
        await ParametricSelfExternalEvidenceRunner(_identity()).run(
            scratch_dir=unsafe_root / "drill",
            trusted_scratch_root=unsafe_root,
            run_nonce="c" * 64,
        )
    with pytest.raises(ExternalReleaseEvidenceError, match="directly inside"):
        await ParametricSelfExternalEvidenceRunner(_identity()).run(
            scratch_dir=trusted_root / "nested" / "drill",
            trusted_scratch_root=trusted_root,
            run_nonce="c" * 64,
        )
    linked = trusted_root / "linked"
    linked.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ExternalReleaseEvidenceError, match="directly inside"):
        await ParametricSelfExternalEvidenceRunner(_identity()).run(
            scratch_dir=linked / "drill",
            trusted_scratch_root=trusted_root,
            run_nonce="c" * 64,
        )
    existing = trusted_root / "existing-sqlite"
    existing.mkdir(mode=0o700)
    with pytest.raises(ExternalReleaseEvidenceError, match="must be fresh"):
        await ParametricSelfExternalEvidenceRunner(_identity()).run(
            scratch_dir=trusted_root / "existing",
            trusted_scratch_root=trusted_root,
            run_nonce="c" * 64,
        )


async def test_external_evidence_detects_scratch_tree_replacement_race(tmp_path, monkeypatch):
    factory, _ = _dual_feature_factory(tmp_path)
    scratch_dir = tmp_path / "race-drill"
    original = release_evidence_module.build_corpus

    def replace_corpus(*args, **kwargs):
        result = original(*args, **kwargs)
        corpus = Path(args[1])
        corpus.rename(corpus.with_name("replaced-corpus"))
        corpus.mkdir(mode=0o700)
        return result

    monkeypatch.setattr(release_evidence_module, "build_corpus", replace_corpus)
    runner = ParametricSelfExternalEvidenceRunner(_identity())
    _install_test_feature_builder(monkeypatch, runner, factory)
    sealed = False

    def refuse_if_sealed(*_args, **_kwargs):
        nonlocal sealed
        sealed = True
        raise AssertionError("a replaced scratch tree must never be signed")

    monkeypatch.setattr(runner, "_seal_envelope", refuse_if_sealed)
    with pytest.raises(ExceptionGroup) as refused:
        await runner.run(
            scratch_dir=scratch_dir,
            trusted_scratch_root=tmp_path,
            run_nonce="d" * 64,
        )
    assert any(
        isinstance(error, ExternalReleaseEvidenceError)
        and "changed during execution" in str(error)
        for error in refused.value.exceptions
    )
    # The cleanup guard refuses to delete a path whose identity changed.
    assert (tmp_path / "race-drill-sqlite").exists()
    assert not sealed


async def test_external_evidence_refuses_non_external_signer(tmp_path):
    wrong_identity = CatalogSigningIdentity(
        issuer_id="core_ci",
        key_id="wrong_source",
        private_key=Ed25519PrivateKey.from_private_bytes(b"\x09" * 32),
    )
    with pytest.raises(ExternalReleaseEvidenceError, match="external_ci"):
        ParametricSelfExternalEvidenceRunner(wrong_identity)


async def test_external_evidence_binds_unique_freshness_to_every_signed_record(tmp_path, monkeypatch):
    first_factory, _ = _dual_feature_factory(tmp_path / "one")
    second_factory, _ = _dual_feature_factory(tmp_path / "two")
    runner = ParametricSelfExternalEvidenceRunner(_identity())

    ledger = ExternalFreshnessLedger(
        tmp_path / "freshness-ledger.sqlite", trusted_root=tmp_path
    )
    _install_test_feature_builder(monkeypatch, runner, first_factory)
    left = await runner.run(
        scratch_dir=tmp_path / "drill-one",
        trusted_scratch_root=tmp_path,
        run_nonce=ledger.issue_challenge(),
    )
    _install_test_feature_builder(monkeypatch, runner, second_factory)
    right = await runner.run(
        scratch_dir=tmp_path / "drill-two",
        trusted_scratch_root=tmp_path,
        run_nonce=ledger.issue_challenge(),
    )
    assert left.run_nonce != right.run_nonce
    assert left.report.freshness_receipt != right.report.freshness_receipt
    assert {record.artifact.artifact_digest for record in left.records}.isdisjoint(
        {record.artifact.artifact_digest for record in right.records}
    )


@pytest.mark.skipif(
    not (
        os.environ.get("KESTREL_SEMANTIC_RELEASE_ISOLATED") == "1"
        and os.environ.get("KESTREL_SEMANTIC_RELEASE_ISOLATED_POSTGRES_ADMIN_DSN")
    ),
    reason=(
        "requires a disposable PostgreSQL authority: "
        "KESTREL_SEMANTIC_RELEASE_ISOLATED=1 "
        "KESTREL_SEMANTIC_RELEASE_ISOLATED_POSTGRES_ADMIN_DSN=postgresql://..."
    ),
)
async def test_real_disposable_postgres_dual_backend_removes_database_and_sqlite_residue(
    tmp_path, monkeypatch
):
    """Opt-in live gate: core must drop its DB and no SQLite-owned file remains."""
    from kestrel_sovereign.knowledge.release_evidence_postgres import (
        DisposablePostgresDatabase as CoreDisposablePostgresDatabase,
    )

    observed: dict[str, object] = {}
    original_close = CoreDisposablePostgresDatabase.close

    async def tracked_close(database) -> None:
        observed["database_name"] = database.database_name
        await original_close(database)
        observed["closed"] = database._closed

    monkeypatch.setattr(
        release_evidence_module,
        "DisposablePostgresDatabase",
        CoreDisposablePostgresDatabase,
    )
    monkeypatch.setattr(CoreDisposablePostgresDatabase, "close", tracked_close)
    trusted_root = tmp_path / "trusted"
    trusted_root.mkdir(mode=0o700)

    envelope = await ParametricSelfExternalEvidenceRunner(_identity()).run(
        scratch_dir=trusted_root / "live-dual-drill",
        trusted_scratch_root=trusted_root,
        run_nonce="1" * 64,
    )

    assert all(record.passed for record in envelope.records)
    assert observed["database_name"]
    assert observed["closed"] is True
    assert list(trusted_root.iterdir()) == []


def test_external_evidence_uses_the_immutable_core_contract_digest() -> None:
    assert len(CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST) == 64
    assert set(CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST) <= set("0123456789abcdef")


def test_external_evidence_refuses_dirty_or_unverifiable_runner_checkout(monkeypatch):
    monkeypatch.setattr(
        release_evidence_module,
        "_resolve_clean_evidence_runner_revision",
        _REAL_RUNNER_REVISION_RESOLVER,
    )
    monkeypatch.setattr(
        release_evidence_module.subprocess,
        "run",
        lambda *_args, **_kwargs: __import__("subprocess").CompletedProcess(
            (), 0, stdout=" M runner.py\n", stderr=""
        ),
    )
    with pytest.raises(ExternalReleaseEvidenceError, match="not clean and verifiable"):
        release_evidence_module._resolve_clean_evidence_runner_revision()

    def unavailable(*_args, **_kwargs):
        raise OSError("unavailable")

    monkeypatch.setattr(release_evidence_module.subprocess, "run", unavailable)
    with pytest.raises(ExternalReleaseEvidenceError, match="not clean and verifiable"):
        release_evidence_module._resolve_clean_evidence_runner_revision()
