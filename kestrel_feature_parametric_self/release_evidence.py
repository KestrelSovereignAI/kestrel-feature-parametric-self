"""External-CI evidence for the parametric-self erasure release gates.

This module is deliberately an *executor*, not a generic JSON formatter.  It
drives an isolated :class:`ParametricSelfFeature` through the governed-corpus
and adapter-lineage paths, waits for a caller-supplied real erasure action, and
only then signs the three external gate records declared by Kestrel core.

The resulting envelope contains aggregates, immutable catalog bindings, and
opaque digests only.  It contains no assertion text, tenant ID, filesystem
path, command line, or erasure implementation detail.  Core still decides
whether the external-CI signing key is trusted when it assembles a release.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import stat
import subprocess
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from kestrel_sovereign.knowledge.corpus import GovernedCorpusSnapshot
from kestrel_sovereign.knowledge.release_evidence import (
    CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST,
    PARAMETRIC_SELF_CAPABILITY_SOURCE_REVISION,
    PARAMETRIC_SELF_EVIDENCE_REPOSITORY,
    release_gate_specs,
)
from kestrel_sovereign.knowledge.release_evidence_execution import CatalogSigningIdentity
from kestrel_sovereign.knowledge.release_evidence_models import (
    ArtifactReference,
    EvidenceRecord,
    EvidenceState,
    ExecutionSource,
    ExternalCapabilityReport,
    ExternalGateAttestation,
    GateSpec,
    ReleaseEvidenceError,
)

from .corpus import build_corpus

if TYPE_CHECKING:
    from .feature import ParametricSelfFeature


EXTERNAL_GATE_IDS = (
    "external_corpus_consumed",
    "external_candidate_invalidated",
    "external_served_eligibility_rejected",
)
_CAPABILITY_ID = "parametric_self_governed_corpus"
_FRESHNESS_NONCE_BYTES = 32
_FULL_COMMIT_LENGTH = 40


class ExternalReleaseEvidenceError(ValueError):
    """The isolated external erasure drill did not establish a required fact."""


ErasureAction = Callable[[], None | Awaitable[None]]


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_full_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _FULL_COMMIT_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve_clean_evidence_runner_revision() -> str:
    """Return this checkout's immutable runner revision or fail closed.

    The submitted provenance is obtained by the runner itself, rather than
    supplied by a caller.  A dirty or unverifiable checkout cannot produce
    release evidence because it has no precise, reviewable source identity.
    """
    repository = Path(__file__).resolve().parents[1]
    try:
        status = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
        revision = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "--verify", "HEAD^{commit}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise ExternalReleaseEvidenceError(
            "external evidence runner checkout is not clean and verifiable"
        ) from error
    if status.stdout or not _is_full_commit(revision):
        raise ExternalReleaseEvidenceError(
            "external evidence runner checkout is not clean and verifiable"
        )
    return revision


def _lstat_private_directory(path: Path, *, expected_identity: tuple[int, int] | None = None) -> tuple[int, int]:
    """Require an owner-only non-symlink directory and return its identity."""
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ExternalReleaseEvidenceError("external evidence scratch tree is unavailable") from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise ExternalReleaseEvidenceError("external evidence scratch tree is not private")
    identity = (metadata.st_dev, metadata.st_ino)
    if expected_identity is not None and identity != expected_identity:
        raise ExternalReleaseEvidenceError("external evidence scratch tree changed during execution")
    return identity


def _prepare_private_scratch_tree(
    scratch_dir: Path,
    trusted_scratch_root: Path,
) -> tuple[Path, Path, Path, dict[Path, tuple[int, int]]]:
    """Create a fresh, private tree inside a verifier-selected root.

    The caller may select a leaf only; the root is explicit and is verified
    before every creation.  No path component is followed through a symlink.
    """
    scratch_dir = Path(scratch_dir)
    trusted_scratch_root = Path(trusted_scratch_root)
    if (
        not scratch_dir.is_absolute()
        or not trusted_scratch_root.is_absolute()
        or ".." in scratch_dir.parts
        or ".." in trusted_scratch_root.parts
        or trusted_scratch_root.is_symlink()
    ):
        raise ExternalReleaseEvidenceError("external evidence scratch path is not trusted")
    try:
        root = trusted_scratch_root.resolve(strict=True)
        relative = scratch_dir.relative_to(root)
    except (OSError, ValueError) as error:
        raise ExternalReleaseEvidenceError("external evidence scratch path is not trusted") from error
    if not relative.parts or scratch_dir.exists() or scratch_dir.is_symlink():
        raise ExternalReleaseEvidenceError("external evidence scratch directory must be fresh")

    identities: dict[Path, tuple[int, int]] = {root: _lstat_private_directory(root)}
    current = root
    for component in relative.parts[:-1]:
        current = current / component
        identities[current] = _lstat_private_directory(current)
    try:
        os.mkdir(scratch_dir, 0o700)
        os.chmod(scratch_dir, 0o700)
        candidate = scratch_dir / "candidate"
        corpus = scratch_dir / "corpus"
        os.mkdir(candidate, 0o700)
        os.mkdir(corpus, 0o700)
        os.chmod(candidate, 0o700)
        os.chmod(corpus, 0o700)
    except OSError as error:
        raise ExternalReleaseEvidenceError("external evidence scratch tree could not be created") from error
    identities[scratch_dir] = _lstat_private_directory(scratch_dir)
    identities[candidate] = _lstat_private_directory(candidate)
    identities[corpus] = _lstat_private_directory(corpus)
    _verify_private_scratch_tree(scratch_dir, identities)
    return scratch_dir, candidate, corpus, identities


def _verify_private_scratch_tree(
    scratch_dir: Path,
    identities: Mapping[Path, tuple[int, int]],
) -> None:
    """Recheck directories and lock generated files to the private tree."""
    for path, identity in identities.items():
        _lstat_private_directory(path, expected_identity=identity)
    try:
        for parent, directories, files in os.walk(scratch_dir, followlinks=False):
            parent_path = Path(parent)
            _lstat_private_directory(parent_path)
            for name in directories:
                _lstat_private_directory(parent_path / name)
            for name in files:
                child = parent_path / name
                metadata = child.lstat()
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid():
                    raise ExternalReleaseEvidenceError("external evidence scratch tree contains an unsafe file")
                os.chmod(child, 0o600)
                if stat.S_IMODE(child.lstat().st_mode) & 0o077:
                    raise ExternalReleaseEvidenceError("external evidence scratch file is not private")
    except OSError as error:
        raise ExternalReleaseEvidenceError("external evidence scratch tree is unavailable") from error


def _cleanup_private_scratch_tree(
    scratch_dir: Path,
    identities: Mapping[Path, tuple[int, int]],
) -> None:
    """Remove only the exact tree this invocation created.

    If an identity check fails, leave it for the verifier/operator rather than
    risking deletion through a replacement path.
    """
    try:
        _verify_private_scratch_tree(scratch_dir, identities)
    except ExternalReleaseEvidenceError:
        return
    shutil.rmtree(scratch_dir)


def _external_specs() -> dict[str, GateSpec]:
    specs = {spec.gate_id: spec for spec in release_gate_specs()}
    selected = {gate_id: specs.get(gate_id) for gate_id in EXTERNAL_GATE_IDS}
    if set(selected) != set(EXTERNAL_GATE_IDS) or any(
        spec is None for spec in selected.values()
    ):
        raise ExternalReleaseEvidenceError("core external release gate catalog is incomplete")
    result = {gate_id: spec for gate_id, spec in selected.items() if spec is not None}
    if any(
        spec.category != "external_adapter"
        or spec.runner.runner_id != "external_ci"
        or spec.owner != "parametric_self"
        or spec.correlation is None
        for spec in result.values()
    ):
        raise ExternalReleaseEvidenceError("core external release gate catalog does not match the adapter contract")
    return result


@dataclass(frozen=True, slots=True)
class ExternalReleaseEvidenceEnvelope:
    """Content-free, independently signed external-CI submission for core.

    It is intentionally not a readiness claim: every record is signed, but
    only core's operator-owned :class:`TrustedExecutionPolicy` may trust it.
    """

    core_release_evidence_contract_digest: str
    repository: str
    capability_source_revision: str
    evidence_runner_revision: str
    run_nonce: str
    records: tuple[EvidenceRecord, ...]
    report: ExternalCapabilityReport

    def __post_init__(self) -> None:
        if self.core_release_evidence_contract_digest != CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST:
            raise ExternalReleaseEvidenceError("external evidence must bind the current core release contract")
        if self.repository != PARAMETRIC_SELF_EVIDENCE_REPOSITORY:
            raise ExternalReleaseEvidenceError("external evidence repository does not match core contract")
        if self.capability_source_revision != PARAMETRIC_SELF_CAPABILITY_SOURCE_REVISION:
            raise ExternalReleaseEvidenceError("external evidence revision does not match core contract")
        if not _is_full_commit(self.evidence_runner_revision):
            raise ExternalReleaseEvidenceError("external evidence runner revision is invalid")
        if len(self.run_nonce) != _FRESHNESS_NONCE_BYTES * 2 or any(
            character not in "0123456789abcdef" for character in self.run_nonce
        ):
            raise ExternalReleaseEvidenceError("external evidence requires a fresh nonce")
        specs = _external_specs()
        by_gate = {record.gate_id: record for record in self.records}
        if set(by_gate) != set(specs) or len(by_gate) != len(self.records):
            raise ExternalReleaseEvidenceError("external evidence must contain each declared adapter gate once")
        for gate_id, record in by_gate.items():
            spec = specs[gate_id]
            try:
                spec.validate_attestation(record)
            except ReleaseEvidenceError as error:
                raise ExternalReleaseEvidenceError("external record does not bind the current core gate spec") from error
            if (
                record.state is not EvidenceState.PASSED
                or record.execution_attestation is None
                or record.execution_attestation.source is not ExecutionSource.EXTERNAL_CI
                or record.external_run_nonce != self.run_nonce
                or record.external_evidence_runner_revision != self.evidence_runner_revision
            ):
                raise ExternalReleaseEvidenceError(
                    "external evidence records must be externally signed passes bound to the verifier nonce"
                )
        if (
            self.report.capability_id != _CAPABILITY_ID
            or self.report.repository != self.repository
            or self.report.capability_source_revision != self.capability_source_revision
            or self.report.evidence_runner_revision != self.evidence_runner_revision
            or self.report.core_release_evidence_contract_digest
            != self.core_release_evidence_contract_digest
            or self.report.run_nonce != self.run_nonce
        ):
            raise ExternalReleaseEvidenceError("external report identity does not match its envelope")
        report_by_gate = {item.gate_id: item for item in self.report.attestations}
        if set(report_by_gate) != set(by_gate):
            raise ExternalReleaseEvidenceError("external report must contain each signed adapter gate once")
        for gate_id, record in by_gate.items():
            item = report_by_gate.get(gate_id)
            if (
                item is None
                or item.gate_spec_digest != specs[gate_id].digest
                or item.result_digest != record.run_digest
                or item.artifact != record.artifact
                or item.drill != specs[gate_id].correlation
            ):
                raise ExternalReleaseEvidenceError("external report is not bound to its signed record")

    @property
    def trust_status(self) -> str:
        return "external_signature_requires_core_policy_verification"

    def to_mapping(self) -> dict[str, object]:
        return {
            "core_release_evidence_contract_digest": self.core_release_evidence_contract_digest,
            "repository": self.repository,
            "capability_source_revision": self.capability_source_revision,
            "evidence_runner_revision": self.evidence_runner_revision,
            "run_nonce": self.run_nonce,
            "trust_status": self.trust_status,
            "records": [record.to_mapping() for record in self.records],
            "report": self.report.to_mapping(),
        }

    def write(self, output: Path, *, overwrite: bool = False) -> None:
        """Write one aggregate-only artifact without replacing another run."""
        if output.exists() and not overwrite:
            raise ExternalReleaseEvidenceError("refusing to replace existing external evidence envelope")
        if not output.parent.is_dir():
            raise ExternalReleaseEvidenceError("external evidence output parent does not exist")
        output.write_text(_canonical_json(self.to_mapping()) + "\n", encoding="utf-8")


class ParametricSelfExternalEvidenceRunner:
    """Execute the real external corpus → invalidation → serving drill.

    ``erase`` is the isolated environment's actual administrative deletion
    operation.  This runner does not accept a caller-supplied observation,
    status, gate ID, artifact reference, or record: it derives every release
    value from core's fixed catalog and the feature's post-erasure state.
    """

    def __init__(self, signing_identity: CatalogSigningIdentity) -> None:
        if (
            not isinstance(signing_identity, CatalogSigningIdentity)
            or signing_identity.source is not ExecutionSource.EXTERNAL_CI
        ):
            raise ExternalReleaseEvidenceError("external evidence requires an external_ci signing identity")
        self._identity = signing_identity

    async def run(
        self,
        feature: "ParametricSelfFeature",
        *,
        scratch_dir: Path,
        trusted_scratch_root: Path,
        run_nonce: str,
        erase: ErasureAction | None = None,
    ) -> ExternalReleaseEvidenceEnvelope:
        """Run one fresh, correlated drill and return its signed envelope.

        The caller is responsible for supplying an isolated Kite/test agent,
        a verifier-issued one-time nonce, and an erasure action scoped to the
        governed assertion created for this drill.  A no-op, unrelated
        deletion, stale snapshot, untracked candidate, or still-served adapter
        all fail closed.
        """
        if erase is not None and not callable(erase):
            raise ExternalReleaseEvidenceError("external erasure action must be callable")
        if (
            not isinstance(run_nonce, str)
            or len(run_nonce) != _FRESHNESS_NONCE_BYTES * 2
            or any(character not in "0123456789abcdef" for character in run_nonce)
        ):
            raise ExternalReleaseEvidenceError("external evidence requires a verifier-issued nonce")
        specs = _external_specs()
        runner_revision = _resolve_clean_evidence_runner_revision()
        if getattr(feature, "_active_adapter_path", None) is not None:
            raise ExternalReleaseEvidenceError("external drill requires an isolated feature with no served adapter")
        scratch_dir, candidate, corpus_dir, scratch_identities = _prepare_private_scratch_tree(
            Path(scratch_dir), Path(trusted_scratch_root)
        )
        try:
            snapshot, problem = await feature._request_governed_snapshot()
            if problem or not isinstance(snapshot, GovernedCorpusSnapshot):
                raise ExternalReleaseEvidenceError("core governed corpus snapshot is unavailable or invalid")
            if not snapshot.examples:
                raise ExternalReleaseEvidenceError("external erasure drill requires a non-empty governed corpus")
            _verify_private_scratch_tree(scratch_dir, scratch_identities)
            stats = build_corpus(
                None,
                str(corpus_dir),
                governed_snapshot=snapshot,
                manifest_dir=str(candidate),
            )
            _verify_private_scratch_tree(scratch_dir, scratch_identities)
            if stats.from_facts <= 0 or not stats.assertion_lineage:
                raise ExternalReleaseEvidenceError("core governed corpus did not contribute an erasure-tracked example")
            manifest, manifest_problem = feature._manifest_lineage(str(candidate))
            receipt = feature._manifest_receipt_stamp(manifest or {}) if manifest else None
            if manifest_problem or receipt is None:
                raise ExternalReleaseEvidenceError("candidate lineage receipt could not be established")

            candidate_path = str(candidate)
            feature._adapter_lineage[candidate_path] = {**receipt, "state": "candidate"}
            feature._active_adapter_path = candidate_path
            feature._live_corpus_snapshot = snapshot

            # The default path is a real, scoped core physical erasure for the exact
            # assertion that produced this snapshot.  ``erase`` remains only
            # for an operator-owned erasure coordinator with a wider physical
            # surface (vectors/exports): it must still produce this core
            # tombstone or the lineage verifier below rejects the run.
            result = (
                self._erase_snapshot_assertion(feature, snapshot, run_nonce)
                if erase is None
                else erase()
            )
            if inspect.isawaitable(result):
                result = await result
            if result is not None:
                raise ExternalReleaseEvidenceError("external erasure action must not supply a result claim")

            invalidation_reason = await feature._verify_adapter_lineage(candidate_path)
            lineage = feature._adapter_lineage.get(candidate_path, {})
            if not invalidation_reason or lineage.get("state") != "invalid":
                raise ExternalReleaseEvidenceError("erasure did not invalidate the governed corpus candidate")
            if feature._active_adapter_path is not None:
                raise ExternalReleaseEvidenceError("erasure did not reject served adapter eligibility")

            # All three gates are proven by one core-correlated drill.  The
            # aggregate deliberately reveals neither the assertion count nor its
            # identity: positive/zero is sufficient to bind the gate schema.
            observations: Mapping[str, Mapping[str, object]] = {
                "external_corpus_consumed": {"erased_count": 1, "remaining_count": 0},
                "external_candidate_invalidated": {"erased_count": 1, "remaining_count": 0},
                "external_served_eligibility_rejected": {"erased_count": 1, "remaining_count": 0},
            }
            records = tuple(
                self._record(
                    specs[gate_id], observations[gate_id], run_nonce, runner_revision
                )
                for gate_id in EXTERNAL_GATE_IDS
            )
            attestations: list[ExternalGateAttestation] = []
            for record in records:
                artifact = record.artifact
                drill = specs[record.gate_id].correlation
                if record.run_digest is None or artifact is None or drill is None:
                    raise ExternalReleaseEvidenceError("signed external record is missing a core binding")
                attestations.append(
                    ExternalGateAttestation(
                        gate_id=record.gate_id,
                        gate_spec_digest=specs[record.gate_id].digest,
                        result_digest=record.run_digest,
                        artifact=artifact,
                        drill=drill,
                    )
                )
            report = ExternalCapabilityReport.attest(
                capability_id=_CAPABILITY_ID,
                repository=PARAMETRIC_SELF_EVIDENCE_REPOSITORY,
                capability_source_revision=PARAMETRIC_SELF_CAPABILITY_SOURCE_REVISION,
                evidence_runner_revision=runner_revision,
                core_release_evidence_contract_digest=CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST,
                attestations=tuple(attestations),
                run_nonce=run_nonce,
            )
            return ExternalReleaseEvidenceEnvelope(
                core_release_evidence_contract_digest=CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST,
                repository=PARAMETRIC_SELF_EVIDENCE_REPOSITORY,
                capability_source_revision=PARAMETRIC_SELF_CAPABILITY_SOURCE_REVISION,
                evidence_runner_revision=runner_revision,
                run_nonce=run_nonce,
                records=records,
                report=report,
            )
        finally:
            # The candidate manifest is lineage-sensitive too.  This exact
            # fresh tree exists solely for the drill, so delete *all* of it on
            # success, exceptions, and task cancellation.
            _cleanup_private_scratch_tree(scratch_dir, scratch_identities)

    @staticmethod
    async def _erase_snapshot_assertion(
        feature: "ParametricSelfFeature",
        snapshot: GovernedCorpusSnapshot,
        run_nonce: str,
    ) -> None:
        storage = getattr(getattr(feature, "agent", None), "storage", None)
        erase = getattr(storage, "erase_assertion", None)
        example = snapshot.examples[0] if snapshot.examples else None
        assertion = getattr(example, "assertion", None)
        if not callable(erase) or assertion is None:
            raise ExternalReleaseEvidenceError("core physical erasure capability is unavailable")
        try:
            await erase(
                assertion.assertion_id,
                operation_id=f"parametric-self-release-erasure:{run_nonce}",
            )
        except Exception as error:
            raise ExternalReleaseEvidenceError("core physical erasure action failed") from error

    def _record(
        self,
        spec: GateSpec,
        observation: Mapping[str, object],
        run_nonce: str,
        evidence_runner_revision: str,
    ) -> EvidenceRecord:
        # The artifact is a digest of only catalog-bound aggregate fields; it
        # cannot be used to smuggle an assertion, tenant, filesystem path, or
        # caller-controlled log into core's release report.
        artifact_digest = _digest(
            {
                "core_release_evidence_contract_digest": CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST,
                "gate_id": spec.gate_id,
                "gate_spec_digest": spec.digest,
                "observation": dict(observation),
                "run_nonce": run_nonce,
            }
        )
        artifact = ArtifactReference(f"ci://sha256/{artifact_digest}", artifact_digest)
        _, run_digest = EvidenceRecord._bound_run_digest(
            spec,
            observation,
            artifact,
            state=EvidenceState.PASSED,
            external_run_nonce=run_nonce,
            external_evidence_runner_revision=evidence_runner_revision,
        )
        return EvidenceRecord._from_trusted_execution(
            spec,
            observation,
            artifact,
            state=EvidenceState.PASSED,
            execution_attestation=self._identity.sign(
                kind="evidence_record", spec=spec, run_digest=run_digest
            ),
            external_run_nonce=run_nonce,
            external_evidence_runner_revision=evidence_runner_revision,
        )


__all__ = [
    "CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST",
    "EXTERNAL_GATE_IDS",
    "ExternalReleaseEvidenceEnvelope",
    "ExternalReleaseEvidenceError",
    "ParametricSelfExternalEvidenceRunner",
]
