"""External #2753 evidence must exercise the real core value contracts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from kestrel_feature_parametric_self import ParametricSelfFeature
from kestrel_feature_parametric_self.release_evidence import (
    CORE_RELEASE_EVIDENCE_COMMIT,
    EXTERNAL_GATE_IDS,
    ExternalReleaseEvidenceError,
    ParametricSelfExternalEvidenceRunner,
)
from kestrel_sovereign.knowledge.assertion import (
    Assertion,
    DirectLineage,
    EpistemicState,
    IRI,
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
from kestrel_sovereign.knowledge.release_evidence_execution import CatalogSigningIdentity
from kestrel_sovereign.knowledge.release_evidence_models import (
    ExecutionSource,
    TrustedExecutionPolicy,
)
from kestrel_sovereign.knowledge.shacl_validation import (
    ValidationState,
    ValidationWriteAction,
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
        asserted_at=datetime(2026, 7, 31, tzinfo=timezone.utc),
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
        datetime(2026, 7, 31, tzinfo=timezone.utc),
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

    async def erase() -> None:
        storage.erased = True

    envelope = await ParametricSelfExternalEvidenceRunner(identity).run(
        feature, scratch_dir=tmp_path / "fresh-drill", erase=erase
    )

    assert envelope.core_release_evidence_commit == CORE_RELEASE_EVIDENCE_COMMIT
    assert tuple(record.gate_id for record in envelope.records) == EXTERNAL_GATE_IDS
    assert all(record.passed for record in envelope.records)
    assert all(record.observation == {"erased_count": 1, "remaining_count": 0} for record in envelope.records)
    assert feature._active_adapter_path is None
    assert feature._adapter_lineage
    assert {lineage["state"] for lineage in feature._adapter_lineage.values()} == {"invalid"}
    assert not (tmp_path / "fresh-drill" / "corpus").exists()

    policy = TrustedExecutionPolicy((identity.trusted_key(("external_ci",)),))
    evidence = apply_evidence_records(
        release_evidence_template(), envelope.records, trust_policy=policy
    )
    attached = attach_external_capability_report(evidence, envelope.report)
    assert attached.external_capabilities == (envelope.report,)

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
            feature, scratch_dir=tmp_path / "fresh-drill", erase=no_op_erase
        )
    assert feature._active_adapter_path is not None
    assert not (tmp_path / "fresh-drill" / "corpus").exists()


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
