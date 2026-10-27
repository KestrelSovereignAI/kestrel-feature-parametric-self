"""Reflection examples reach training only through the governed corpus (#32).

The host runs core's real ``GovernedAssertionCorpusService`` over core-shaped
assertion storage, so consent, source and exclusion rules are applied by core
itself. Its snapshot method accepts exactly what core's ``AsyncStorage``
accepts and registers each snapshot through core's
``GovernedArtifactRegistration``. The trainer keeps the corpus it was handed,
which is what reaches the adapter's weights.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from kestrel_sovereign.features.training.types import TrainingState, TrainingStatus
from kestrel_sovereign.knowledge import (
    IRI,
    XSD_STRING,
    Assertion,
    CorpusCheckpoint,
    CorpusValidationStatus,
    DirectLineage,
    EpistemicState,
    GovernedArtifactKind,
    GovernedArtifactLineage,
    GovernedArtifactRegistration,
    GovernedAssertionCorpusService,
    GovernedCorpusPolicy,
    Literal,
    OntologyRef,
    SemanticMaintenanceTrainingReadiness,
    SourceOccurrence,
    ValidationState,
    ValidationWriteAction,
    Visibility,
)
from kestrel_sovereign.storage.async_storage import AsyncStorage

from kestrel_feature_parametric_self import REFLECTION_SOURCE_KIND, ParametricSelfFeature, build_corpus
from kestrel_feature_parametric_self.corpus import GovernedCorpusRequiredError

# Session and saved-item exclusions ship in a core release after this
# package's floor; until the floor reaches it, these tests run only against
# such a core, and a reflection is never admitted on an older one.
CORE_CAN_EXCLUDE = callable(getattr(GovernedCorpusPolicy, "excludes", None))
needs_corpus_exclusions = pytest.mark.skipif(
    not CORE_CAN_EXCLUDE,
    reason="installed core predates GovernedCorpusPolicy exclusions (kestrel-sovereign#3849)",
)

TENANT = "did:example:meridian"
ONTOLOGY = OntologyRef("https://example.test/ontology", "1.0.0", "sha256:test", "semantic-kb-v1")
CAPABILITIES = {"semantic_maintenance": "v3"}
CONSENT = "consent:approved-batch-1"
FACT_SOURCE_KIND = "agent_tool_invocation"
EXAM_SESSION = "dcd6f767-829d-4a7d-b8c7-d7a722e5de13"
EVERYDAY_SESSION = "5b0c1e2a-3f4d-4e6a-8b7c-9d0e1f2a3b4c"
EXAM_LESSON = "The held-out exam answer is forty-two"
EVERYDAY_LESSON = "Shorter replies land better"
FACT = "tea over coffee"
# Core takes the consumer's Ed25519 public key as raw lowercase hex.
ARTIFACT_CONSUMER = {
    "consumer_id": "parametric-self:meridian",
    "consumer_key_id": "consumer-key:meridian",
    "consumer_public_key": "ab" * 32,
    "retention_seconds": 300.0,
}


def _assertion(revision: str, lesson: str) -> Assertion:
    return Assertion(
        tenant_id=TENANT,
        owning_agent_id=TENANT,
        subject=IRI("urn:kestrel:agent:did:example:meridian:self"),
        predicate=IRI("https://example.test/learned"),
        object=Literal(lesson, XSD_STRING),
        revision_id=revision,
        confidence=Decimal("1"),
        confidence_method="operator",
        confidence_basis="operator-attested",
        epistemic_state=EpistemicState.ASSERTED,
        asserted_at="2026-10-25T12:00:00Z",
        ontology_version=ONTOLOGY,
        lineage=DirectLineage((f"source:{revision}",)),
        privacy_classification="normal",
        release_policy_reference=CONSENT,
        visibility=Visibility.PRIVATE,
    )


def _source(assertion: Assertion, source_kind: str, locator: str) -> SourceOccurrence:
    return SourceOccurrence(
        source_occurrence_id=assertion.lineage.source_occurrence_ids[0],
        source_kind=source_kind,
        locator=locator,
        received_at="2026-10-25T12:00:00Z",
        content_digest="sha256:source",
    )


def _reflection_locator(insight_id: str, *session_ids: str) -> str:
    """A reflection source naming the sessions its insight was derived from."""
    segments = [f"reflection:{insight_id}"]
    if session_ids:
        from kestrel_sovereign.knowledge import SourceReferenceKind, source_reference_segment

        segments += [source_reference_segment(SourceReferenceKind.SESSION, item) for item in session_ids]
    return "#".join(segments)


def _fact() -> tuple[Assertion, SourceOccurrence]:
    fact = _assertion("fact", FACT)
    return fact, _source(fact, FACT_SOURCE_KIND, "POST:/api/agent/invoke#tool:memory_agency.save_fact#op-1")


def _reflection(revision: str, lesson: str, *session_ids: str) -> tuple[Assertion, SourceOccurrence]:
    insight = _assertion(revision, lesson)
    return insight, _source(insight, REFLECTION_SOURCE_KIND, _reflection_locator(revision, *session_ids))


def _policy(**exclusions) -> GovernedCorpusPolicy:
    return GovernedCorpusPolicy(
        policy_id="meridian-training",
        policy_version="1",
        accepted_epistemic_states=(EpistemicState.ASSERTED,),
        accepted_visibility=(Visibility.PRIVATE,),
        accepted_privacy_classifications=("normal",),
        accepted_consent_references=(CONSENT,),
        accepted_grounding_classes=("operator-attested",),
        accepted_source_kinds=(FACT_SOURCE_KIND, REFLECTION_SOURCE_KIND),
        accepted_ontology_pins=(ONTOLOGY,),
        accepted_semantic_capability_versions=tuple(CAPABILITIES.items()),
        **exclusions,
    )


class _Host:
    """Agent storage: graph nodes plus core's governed corpus over its assertions."""

    def __init__(self, *items: tuple[Assertion, SourceOccurrence]) -> None:
        self.assertions = [assertion for assertion, _ in items]
        self.sources = {assertion.revision_id: [source] for assertion, source in items}
        self.validations = {
            assertion.assertion_id: CorpusValidationStatus(
                ValidationState.CONFORMS, ValidationWriteAction.ACCEPT,
                "kestrel-assertion-shapes", "1.0.0", "1.0.0",
            )
            for assertion in self.assertions
        }
        self.checkpoint = CorpusCheckpoint(TENANT, 7, "event:7")
        self.nodes = {}
        self.registrations = []

    # Core's GovernedCorpusStorage protocol, as core's own corpus tests host it.
    async def assertion_event_checkpoint(self):
        return self.checkpoint

    async def assertion_changes_after(self, checkpoint, *, limit=100):
        return []

    async def assertion_inference_inputs(self, query=None):
        values = self.assertions
        if query is not None and query.assertion_ids:
            values = [item for item in values if item.assertion_id in query.assertion_ids]
        return list(values[: query.limit if query is not None else None])

    async def list_assertion_revision_sources_batch(self, revision_ids):
        return {revision_id: list(self.sources.get(revision_id, ())) for revision_id in revision_ids}

    async def get_derivation_inputs(self, revision_id):
        return []

    async def assertion_validation_statuses(self, assertions):
        return {
            item.assertion_id: self.validations[item.assertion_id]
            for item in assertions
            if item.assertion_id in self.validations
        }

    async def semantic_maintenance_training_readiness(self, *args, **kwargs):
        return SemanticMaintenanceTrainingReadiness(True, None)

    async def semantic_maintenance_capability_versions(self, *args, **kwargs):
        return dict(CAPABILITIES)

    async def governed_assertion_corpus_snapshot(self, **kwargs):
        inspect.signature(AsyncStorage.governed_assertion_corpus_snapshot).bind(self, **kwargs)
        snapshot = await GovernedAssertionCorpusService(self).snapshot(
            policy=kwargs["policy"], inference_profile=kwargs["inference_profile"],
        )
        self.registrations.append(GovernedArtifactRegistration(
            artifact_id=kwargs["artifact_id"],
            kind=GovernedArtifactKind.CORPUS_MANIFEST,
            tenant_id=snapshot.tenant_id,
            consumer_id=kwargs["consumer_id"],
            consumer_key_id=kwargs["consumer_key_id"],
            consumer_public_key=kwargs["consumer_public_key"],
            checkpoint_generation=snapshot.checkpoint.generation,
            policy_pin=kwargs["policy"].digest,
            capability_pins={
                name: hashlib.sha256(f"{name}:{version}".encode()).hexdigest()
                for name, version in snapshot.capability_versions.items()
            },
            lineage=tuple(
                GovernedArtifactLineage(item.assertion.assertion_id, item.assertion.revision_id)
                for item in snapshot.examples
            ),
            retention_expires_at=(
                datetime.now(timezone.utc) + timedelta(seconds=kwargs["retention_seconds"])
            ).isoformat(),
            artifact_digest=snapshot.snapshot_hash,
        ))
        return snapshot

    async def add_node(self, node):
        self.nodes[node.node_id] = copy.deepcopy(node)

    async def get_node(self, node_id):
        return copy.deepcopy(self.nodes.get(node_id))


class _CorpusKeepingTrainer:
    """Completes at once, keeping the corpus text it was handed."""

    def __init__(self) -> None:
        self.corpus: str | None = None

    def is_available(self) -> bool:
        return True

    async def start_training(self, agent_id, config):
        data = Path(config.data_dir)
        self.corpus = "".join((data / name).read_text() for name in ("train.jsonl", "valid.jsonl"))
        return TrainingStatus(job_id="job-1", state=TrainingState.COMPLETED, progress=1.0)

    async def get_status(self, job_id):
        return TrainingStatus(job_id=job_id, state=TrainingState.COMPLETED, progress=1.0)

    def read_training_log(self, job_id):
        return "Iter 100: Val loss 1.2"

    async def cancel(self, job_id):
        return True


async def _feature(host: _Host, tmp_path: Path, policy: GovernedCorpusPolicy) -> ParametricSelfFeature:
    agent = MagicMock()
    agent.storage = host
    agent.storage_path = str(tmp_path / "kestrel_prime.db")
    agent.parametric_self_work_dir = None
    agent.parametric_self_governed_corpus_policy = None
    agent.parametric_self_governed_artifact_consumer = dict(ARTIFACT_CONSUMER)
    agent.semantic_inference_profile = None
    agent.is_test_instance = False
    agent.agent_id = "meridian"
    agent.sleep_hooks = []
    feature = ParametricSelfFeature(agent=agent)
    await feature.initialize()
    # The operator's policy arrives as its mapping, as persisted config does.
    await feature.set_config({"governed_corpus_policy": policy.to_mapping()})
    feature._adapter = _CorpusKeepingTrainer()
    return feature


async def _train(feature: ParametricSelfFeature) -> tuple[str, dict]:
    """Run the feature's training cycle; return the trained corpus and its manifest.

    Promotion re-verifies the candidate's manifest, so a governed reflection's
    session lineage must satisfy the same check that refuses a raw one.
    """
    outcome = await feature._run_training_cycle_locked(trigger="manual")
    assert outcome["trained"] is True and outcome["promoted"] is True, outcome
    manifest_path = Path(feature._resolve_work_dir()) / "candidates"
    (candidate,) = manifest_path.iterdir()
    manifest = json.loads((candidate / "corpus_manifest.json").read_text())
    return feature._adapter.corpus, manifest


def _reflection_db(path: Path) -> None:
    """Reflection's own table, holding an insight from the exam session."""
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE reflection_insights (id TEXT, agent_id TEXT, session_id TEXT, type TEXT, "
        "title TEXT, description TEXT, suggested_action TEXT)"
    )
    con.executemany(
        "INSERT INTO reflection_insights VALUES (?,?,?,?,?,?,?)",
        [
            ("1", "meridian", EXAM_SESSION, "failure", "Exam", EXAM_LESSON, "Remember it."),
            ("2", "meridian", EVERYDAY_SESSION, "success", "Brevity", EVERYDAY_LESSON, ""),
        ],
    )
    con.commit()
    con.close()


async def test_raw_reflection_rows_never_reach_training(tmp_path):
    _reflection_db(tmp_path / "kestrel_prime.db")
    feature = await _feature(_Host(_fact()), tmp_path, _policy())

    corpus, manifest = await _train(feature)

    assert FACT in corpus
    assert EXAM_LESSON not in corpus
    assert EVERYDAY_LESSON not in corpus
    assert [item["source"] for item in manifest["examples"]] == ["governed_assertion"]


@needs_corpus_exclusions
async def test_reflection_from_excluded_session_never_reaches_training(tmp_path):
    host = _Host(
        _fact(),
        _reflection("exam-reflection", EXAM_LESSON, EXAM_SESSION),
        _reflection("everyday-reflection", EVERYDAY_LESSON, EVERYDAY_SESSION),
    )
    unexcluded = await _feature(host, tmp_path / "unexcluded", _policy())
    held_out = await _feature(
        host, tmp_path / "held-out", _policy(excluded_session_ids=(EXAM_SESSION,)),
    )

    unexcluded_corpus, _ = await _train(unexcluded)
    corpus, manifest = await _train(held_out)

    # Without the exclusion the reflection path admits the exam lesson.
    assert EXAM_LESSON in unexcluded_corpus
    assert EXAM_LESSON not in corpus
    assert EVERYDAY_LESSON in corpus and FACT in corpus
    reflections = [item for item in manifest["examples"] if item["source"] == "reflection"]
    assert [item["lineage"]["session_ids"] for item in reflections] == [[EVERYDAY_SESSION]]
    assert manifest["counts"]["reflection"] == 1
    assert manifest["policy_digest"] == _policy(excluded_session_ids=(EXAM_SESSION,)).digest


@needs_corpus_exclusions
async def test_reflection_that_names_no_session_is_left_out(tmp_path):
    host = _Host(_fact(), _reflection("unsourced-reflection", EVERYDAY_LESSON))
    feature = await _feature(host, tmp_path, _policy())

    corpus, manifest = await _train(feature)

    assert EVERYDAY_LESSON not in corpus
    assert FACT in corpus
    assert manifest["counts"]["reflection"] == 0
    assert manifest["counts"]["reflection_unresolved"] == 1


async def test_reflection_is_left_out_by_a_core_that_cannot_exclude_sessions(tmp_path):
    # The reflection names its session wherever the installed core can say so.
    sessions = (EVERYDAY_SESSION,) if CORE_CAN_EXCLUDE else ()
    host = _Host(_fact(), _reflection("everyday-reflection", EVERYDAY_LESSON, *sessions))
    snapshot = await GovernedAssertionCorpusService(host).snapshot(
        policy=_policy(), inference_profile=None,
    )
    # A policy from a core before exclusions: pinned by digest, unable to exclude.
    pre_exclusion = replace(snapshot, policy=SimpleNamespace(digest=snapshot.policy.digest))

    stats = build_corpus(
        str(tmp_path / "corpus"), governed_snapshot=pre_exclusion,
        manifest_dir=str(tmp_path / "candidate"),
    )

    corpus = "".join(
        (tmp_path / "corpus" / name).read_text() for name in ("train.jsonl", "valid.jsonl")
    )
    assert EVERYDAY_LESSON not in corpus and FACT in corpus
    assert (stats.from_insights, stats.insights_unresolved, stats.from_facts) == (0, 1, 1)
    assert [revision for _, revision in stats.assertion_lineage] == ["fact"]


@needs_corpus_exclusions
async def test_snapshot_that_admits_an_excluded_reflection_is_refused(tmp_path):
    host = _Host(_reflection("exam-reflection", EXAM_LESSON, EXAM_SESSION))
    unexcluded = await GovernedAssertionCorpusService(host).snapshot(
        policy=_policy(), inference_profile=None,
    )
    # A host that ignored the exclusion its policy names.
    ignored = replace(unexcluded, policy=_policy(excluded_session_ids=(EXAM_SESSION,)))

    with pytest.raises(GovernedCorpusRequiredError, match="contains_excluded_example"):
        build_corpus(
            str(tmp_path / "corpus"), governed_snapshot=ignored,
            manifest_dir=str(tmp_path / "candidate"),
        )
    assert not (tmp_path / "corpus" / "train.jsonl").exists()


@needs_corpus_exclusions
async def test_policy_exclusions_survive_a_restart(tmp_path):
    policy = _policy(excluded_session_ids=(EXAM_SESSION,))
    host = _Host(_fact())
    await _feature(host, tmp_path, policy)

    restarted = ParametricSelfFeature(agent=MagicMock(storage=host))
    await restarted.initialize()
    await restarted._restore_persisted_config()

    assert restarted._governed_corpus_policy == policy
    assert restarted._governed_corpus_policy.digest == policy.digest


def test_policy_mapping_with_an_unknown_field_is_refused():
    mapping = {**_policy().to_mapping(), "excluded_conversation_ids": ["x"]}

    with pytest.raises(ValueError, match="excluded_conversation_ids"):
        ParametricSelfFeature._coerce_policy(mapping)


@pytest.mark.skipif(CORE_CAN_EXCLUDE, reason="this core applies exclusions")
def test_core_without_exclusions_refuses_rather_than_drops_them():
    mapping = {**_policy().to_mapping(), "excluded_session_ids": [EXAM_SESSION]}

    with pytest.raises(ValueError, match="excluded_session_ids"):
        ParametricSelfFeature._coerce_policy(mapping)
