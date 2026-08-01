"""External #2753 evidence must exercise the real core value contracts."""

from __future__ import annotations

import json
import os
import sys
import types
from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
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

    def __init__(self, snapshot: GovernedCorpusSnapshot) -> None:
        self.snapshot = snapshot
        self.erased = False
        self.nodes: dict[str, object] = {}

    async def governed_assertion_corpus_snapshot(self, **_kwargs):
        return self.snapshot

    async def governed_assertion_corpus_changes_since(self, snapshot, **_kwargs):
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

    async def add_node(self, node) -> None:
        self.nodes[node.node_id] = node

    async def get_node(self, node_id):
        return self.nodes.get(node_id)


async def _feature(storage: _CoreBackedErasureStorage, tmp_path: Path) -> ParametricSelfFeature:
    agent = MagicMock()
    agent.storage = storage
    agent.storage_path = None
    agent.parametric_self_work_dir = str(tmp_path / "work")
    agent.parametric_self_governed_corpus_policy = storage.snapshot.policy
    agent.semantic_inference_profile = None
    agent.is_test_instance = False
    agent.agent_id = "agent-evidence"
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


async def test_external_evidence_runs_real_core_snapshot_to_quarantine_and_signs(tmp_path):
    storage = _CoreBackedErasureStorage(_snapshot())
    feature = await _feature(storage, tmp_path)
    identity = _identity()
    ledger = ExternalFreshnessLedger(
        tmp_path / "verifier-freshness.sqlite", trusted_root=tmp_path
    )
    run_nonce = ledger.issue_challenge()

    async def erase() -> None:
        storage.erased = True

    envelope = await ParametricSelfExternalEvidenceRunner(identity).run(
        feature,
        scratch_dir=tmp_path / "fresh-drill",
        trusted_scratch_root=tmp_path,
        run_nonce=run_nonce,
        erase=erase,
    )

    assert envelope.core_release_evidence_contract_digest == CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST
    assert tuple(record.gate_id for record in envelope.records) == EXTERNAL_GATE_IDS
    assert all(record.passed for record in envelope.records)
    assert all(record.observation == {"erased_count": 1, "remaining_count": 0} for record in envelope.records)
    assert feature._active_adapter_path is None
    assert feature._adapter_lineage
    assert {lineage["state"] for lineage in feature._adapter_lineage.values()} == {"invalid"}
    assert not (tmp_path / "fresh-drill").exists()
    assert len(envelope.run_nonce) == 64
    assert envelope.run_nonce == run_nonce
    assert len(envelope.report.freshness_receipt) == 64
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
    with pytest.raises(ReleaseEvidenceError, match="external run_nonce"):
        attach_external_capability_report(
            evidence,
            rewrapped,
            freshness_ledger=ledger,
            expected_evidence_runner_revision=envelope.evidence_runner_revision,
        )

    unknown_storage = _CoreBackedErasureStorage(_snapshot())
    unknown_feature = await _feature(unknown_storage, tmp_path / "unknown")

    async def erase_unknown() -> None:
        unknown_storage.erased = True

    unknown_envelope = await ParametricSelfExternalEvidenceRunner(identity).run(
        unknown_feature,
        scratch_dir=tmp_path / "unknown-drill",
        trusted_scratch_root=tmp_path,
        run_nonce="f" * 64,
        erase=erase_unknown,
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
    payload = json.loads(output.read_text())
    rendered = json.dumps(payload)
    assert payload["trust_status"] == "external_signature_requires_core_policy_verification"
    assert "governed lesson" not in rendered
    assert "tenant-evidence" not in rendered
    assert str(tmp_path) not in rendered


async def test_external_evidence_fails_closed_when_erasure_does_not_change_core_delta(tmp_path):
    storage = _CoreBackedErasureStorage(_snapshot())
    feature = await _feature(storage, tmp_path)

    async def no_op_erase() -> None:
        return None

    with pytest.raises(ExternalReleaseEvidenceError, match="did not invalidate"):
        await ParametricSelfExternalEvidenceRunner(_identity()).run(
            feature,
            scratch_dir=tmp_path / "fresh-drill",
            trusted_scratch_root=tmp_path,
            run_nonce="a" * 64,
            erase=no_op_erase,
        )
    # Failed observation fails closed: leaving a candidate marked served after
    # a purported erasure would be a more dangerous state than quarantining it.
    assert feature._active_adapter_path is None
    assert not (tmp_path / "fresh-drill").exists()


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
    envelope = await hook.observe(prepared)

    assert storage.erased is True
    assert all(record.passed for record in envelope.records)
    assert feature._active_adapter_path is None
    with pytest.raises(ExternalReleaseEvidenceError, match="already consumed"):
        await hook.observe(prepared)


async def test_external_evidence_cli_requires_a_kite_factory_and_runs_two_phases(tmp_path, monkeypatch):
    storage = _CoreBackedErasureStorage(_snapshot())
    feature = await _feature(storage, tmp_path)
    feature.agent.is_test_instance = True
    module_name = "kestrel_feature_parametric_self._test_kite_factory"
    module = types.ModuleType(module_name)

    async def make_feature():
        return feature

    module.make_feature = make_feature
    monkeypatch.setitem(sys.modules, module_name, module)
    key_file = tmp_path / "external-ci.key"
    key_file.write_text("08" * 32, encoding="ascii")
    key_file.chmod(0o600)
    output = tmp_path / "external-evidence.json"
    args = Namespace(
        feature_factory=f"{module_name}:make_feature",
        signing_key_file=key_file,
        issuer_id="parametric_self_ci",
        key_id="release_evidence_key",
        run_nonce="2" * 64,
        scratch_dir=tmp_path / "cli-drill",
        trusted_scratch_root=tmp_path,
        output=output,
    )

    envelope = await release_evidence_module._run_cli(args)

    assert storage.erased is True
    assert output.exists()
    assert json.loads(output.read_text())["run_nonce"] == envelope.run_nonce


async def test_external_evidence_scratch_tree_is_private_while_plaintext_is_live(tmp_path):
    storage = _CoreBackedErasureStorage(_snapshot())
    feature = await _feature(storage, tmp_path)
    trusted_root = tmp_path / "trusted-scratch"
    trusted_root.mkdir(mode=0o700)
    scratch_dir = trusted_root / "drill"

    async def erase() -> None:
        for directory in (scratch_dir, scratch_dir / "candidate", scratch_dir / "corpus"):
            metadata = directory.stat()
            assert metadata.st_uid == os.geteuid()
            assert metadata.st_mode & 0o777 == 0o700
        for path in scratch_dir.rglob("*"):
            if path.is_file():
                assert path.stat().st_mode & 0o777 == 0o600
        storage.erased = True

    await ParametricSelfExternalEvidenceRunner(_identity()).run(
        feature,
        scratch_dir=scratch_dir,
        trusted_scratch_root=trusted_root,
        run_nonce="b" * 64,
        erase=erase,
    )
    assert not scratch_dir.exists()


async def test_external_evidence_refuses_unsafe_or_reused_scratch_paths(tmp_path):
    storage = _CoreBackedErasureStorage(_snapshot())
    feature = await _feature(storage, tmp_path)
    trusted_root = tmp_path / "trusted-scratch"
    trusted_root.mkdir(mode=0o700)
    unsafe_root = tmp_path / "unsafe-scratch"
    unsafe_root.mkdir(mode=0o755)

    with pytest.raises(ExternalReleaseEvidenceError, match="not private"):
        await ParametricSelfExternalEvidenceRunner(_identity()).run(
            feature,
            scratch_dir=unsafe_root / "drill",
            trusted_scratch_root=unsafe_root,
            run_nonce="c" * 64,
        )
    with pytest.raises(ExternalReleaseEvidenceError, match="not trusted"):
        await ParametricSelfExternalEvidenceRunner(_identity()).run(
            feature,
            scratch_dir=trusted_root / ".." / "escaped",
            trusted_scratch_root=trusted_root,
            run_nonce="c" * 64,
        )
    linked = trusted_root / "linked"
    linked.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ExternalReleaseEvidenceError, match="scratch tree"):
        await ParametricSelfExternalEvidenceRunner(_identity()).run(
            feature,
            scratch_dir=linked / "drill",
            trusted_scratch_root=trusted_root,
            run_nonce="c" * 64,
        )
    existing = trusted_root / "existing"
    existing.mkdir(mode=0o700)
    with pytest.raises(ExternalReleaseEvidenceError, match="must be fresh"):
        await ParametricSelfExternalEvidenceRunner(_identity()).run(
            feature,
            scratch_dir=existing,
            trusted_scratch_root=trusted_root,
            run_nonce="c" * 64,
        )


async def test_external_evidence_detects_scratch_tree_replacement_race(tmp_path, monkeypatch):
    storage = _CoreBackedErasureStorage(_snapshot())
    feature = await _feature(storage, tmp_path)
    scratch_dir = tmp_path / "race-drill"
    original = release_evidence_module.build_corpus

    def replace_corpus(*args, **kwargs):
        result = original(*args, **kwargs)
        corpus = Path(args[1])
        corpus.rename(corpus.with_name("replaced-corpus"))
        corpus.mkdir(mode=0o700)
        return result

    monkeypatch.setattr(release_evidence_module, "build_corpus", replace_corpus)
    with pytest.raises(ExternalReleaseEvidenceError, match="changed during execution"):
        await ParametricSelfExternalEvidenceRunner(_identity()).run(
            feature,
            scratch_dir=scratch_dir,
            trusted_scratch_root=tmp_path,
            run_nonce="d" * 64,
        )
    # The cleanup guard refuses to delete a path whose identity changed.
    assert scratch_dir.exists()


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


async def test_external_evidence_refuses_non_external_signer(tmp_path):
    storage = _CoreBackedErasureStorage(_snapshot())
    feature = await _feature(storage, tmp_path)
    wrong_identity = CatalogSigningIdentity(
        issuer_id="core_ci",
        key_id="wrong_source",
        private_key=Ed25519PrivateKey.from_private_bytes(b"\x09" * 32),
    )
    with pytest.raises(ExternalReleaseEvidenceError, match="external_ci"):
        ParametricSelfExternalEvidenceRunner(wrong_identity)
    assert feature._active_adapter_path is None


async def test_external_evidence_binds_unique_freshness_to_every_signed_record(tmp_path):
    first_storage = _CoreBackedErasureStorage(_snapshot())
    second_storage = _CoreBackedErasureStorage(_snapshot())
    first = await _feature(first_storage, tmp_path / "one")
    second = await _feature(second_storage, tmp_path / "two")
    runner = ParametricSelfExternalEvidenceRunner(_identity())

    async def erase_first():
        first_storage.erased = True

    async def erase_second():
        second_storage.erased = True

    ledger = ExternalFreshnessLedger(
        tmp_path / "freshness-ledger.sqlite", trusted_root=tmp_path
    )
    left = await runner.run(
        first,
        scratch_dir=tmp_path / "drill-one",
        trusted_scratch_root=tmp_path,
        run_nonce=ledger.issue_challenge(),
        erase=erase_first,
    )
    right = await runner.run(
        second,
        scratch_dir=tmp_path / "drill-two",
        trusted_scratch_root=tmp_path,
        run_nonce=ledger.issue_challenge(),
        erase=erase_second,
    )
    assert left.run_nonce != right.run_nonce
    assert left.report.freshness_receipt != right.report.freshness_receipt
    assert {record.artifact.artifact_digest for record in left.records}.isdisjoint(
        {record.artifact.artifact_digest for record in right.records}
    )


def test_external_evidence_uses_the_immutable_core_contract_digest() -> None:
    assert len(CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST) == 64
    assert set(CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST) <= set("0123456789abcdef")


async def test_external_evidence_default_path_uses_real_core_storage_privacy_and_erasure(tmp_path):
    """No fake corpus/delta: core creates the fact and its physical tombstone."""
    from kestrel_sovereign.knowledge import InferenceProfile
    from kestrel_sovereign.privacy import PrivacyMode
    from kestrel_sovereign.storage.async_assertion_store import (
        _issue_assertion_tenant_capability,
    )
    from kestrel_sovereign.storage.async_storage import AsyncStorage
    from kestrel_sovereign.storage.privacy_wrapper import PrivacyEnforcingStorage

    tenant = "did:kestrel:release-evidence:e2e"
    raw = AsyncStorage(
        str(tmp_path / "state.db"),
        agent_id=tenant,
        _assertion_tenant_capability=_issue_assertion_tenant_capability(tenant),
    )
    await raw.initialize()
    try:
        governed = PrivacyEnforcingStorage(raw, PrivacyMode.NORMAL)
        saved = await governed.save_explicit_fact(
            subject="user", predicate="preferred_deploy_region", value="private e2e value",
            confidence=0.9, invocation_id="release-evidence-e2e",
        )
        assert saved.saved
        profile = InferenceProfile(
            OntologyRef(
                "http://www.w3.org/2000/01/rdf-schema#", "1.0.0",
                "e362812917fddab7cfab3dc35553ad292725e8f264e05f376077340e91034db5",
                "semantic-kb-v1",
            ), "1.0.0",
        )
        await raw.run_semantic_maintenance(profile)
        assertion = (await raw.assertion_inference_inputs())[0]
        capabilities = await raw.semantic_maintenance_capability_versions(profile)
        policy = GovernedCorpusPolicy(
            policy_id="release-evidence-e2e", policy_version="1",
            accepted_epistemic_states=(EpistemicState.REPORTED,),
            accepted_visibility=(Visibility.PRIVATE,),
            accepted_privacy_classifications=("normal",),
            accepted_consent_references=("policy:privacy:normal-v1",),
            accepted_grounding_classes=("explicit-tool-invocation",),
            accepted_source_kinds=("agent_tool_invocation",),
            accepted_ontology_pins=(assertion.ontology_version,),
            accepted_semantic_capability_versions=tuple(capabilities.items()),
        )
        agent = SimpleNamespace(
            storage=raw, storage_path=None, parametric_self_work_dir=str(tmp_path / "work"),
            parametric_self_governed_corpus_policy=policy, semantic_inference_profile=profile,
            is_test_instance=False, agent_id=tenant, sleep_hooks=[],
        )
        feature = ParametricSelfFeature(agent=agent)
        await feature.initialize()
        feature._governed_corpus_policy = policy
        ledger = ExternalFreshnessLedger(
            tmp_path / "real-verifier-freshness.sqlite", trusted_root=tmp_path
        )
        envelope = await ParametricSelfExternalEvidenceRunner(_identity()).run(
            feature,
            scratch_dir=tmp_path / "fresh-real-drill",
            trusted_scratch_root=tmp_path,
            run_nonce=ledger.issue_challenge(),
        )
        assert all(record.passed for record in envelope.records)
        # Physical erasure removes the canonical row and records only the
        # blinded operation shell; it is not a lifecycle "deleted" revision.
        assert await raw.get_assertion(saved.assertion_id, include_inactive=True) is None
        assert await raw.db.fetchval(
            "SELECT COUNT(*) FROM semantic_assertions WHERE assertion_id = ?",
            (saved.assertion_id,),
        ) == 0
        assert await raw.db.fetchval(
            "SELECT COUNT(*) FROM semantic_assertion_erased_operation_tombstones"
        ) == 1
        assert feature._active_adapter_path is None
        assert not (tmp_path / "fresh-real-drill").exists()
    finally:
        await raw.close()
