"""External-CI evidence for the parametric-self erasure release gates.

This module is deliberately an *executor*, not a generic JSON formatter.  It
drives isolated SQLite and core-created disposable PostgreSQL
:class:`ParametricSelfFeature` instances through the governed-corpus and
adapter-lineage paths. It signs the four declared external records only after
both independent physical-erasure observations and cleanup complete.

The resulting envelope contains aggregates, immutable catalog bindings, and
opaque digests only.  It contains no assertion text, tenant ID, filesystem
path, command line, or erasure implementation detail.  Core still decides
whether the external-CI signing key is trusted when it assembles a release.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import inspect
import json
import os
import secrets
import shutil
import stat
import subprocess
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

from kestrel_sovereign.knowledge.corpus import GovernedCorpusSnapshot
from kestrel_sovereign.knowledge.release_evidence import (
    CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST,
    PARAMETRIC_SELF_CAPABILITY_SOURCE_REVISION,
    PARAMETRIC_SELF_EVIDENCE_REPOSITORY,
    release_gate_specs,
)
from kestrel_sovereign.knowledge.release_evidence_execution import (
    CatalogSigningIdentity,
)
from kestrel_sovereign.knowledge.release_evidence_postgres import (
    DisposablePostgresDatabase,
)
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
    "erasure_served_adapter_eligibility",
    "external_corpus_consumed",
    "external_candidate_invalidated",
    "external_served_eligibility_rejected",
)
_EXTERNAL_CAPABILITY_GATE_IDS = EXTERNAL_GATE_IDS[1:]
_CAPABILITY_ID = "parametric_self_governed_corpus"
_FRESHNESS_NONCE_BYTES = 32
_FULL_COMMIT_LENGTH = 40


class ExternalReleaseEvidenceError(ValueError):
    """The isolated external erasure drill did not establish a required fact."""


FeatureFactory = Callable[
    ["KiteErasureBackend"], "ParametricSelfFeature | Awaitable[ParametricSelfFeature]"
]


@dataclass(frozen=True, slots=True)
class KiteErasurePreparation:
    """Opaque capability for one prepared Kite erasure drill.

    The assertion identity, governed snapshot, candidate location, and feature
    stay in the hook's private state. The capability only correlates the two
    explicitly ordered phases and carries no user content or tenant identity.
    """

    _drill_id: str
    run_nonce: str
    evidence_runner_revision: str


@dataclass(slots=True)
class _PreparedKiteDrill:
    """Private, server-owned state for a prepared external evidence drill."""

    feature: ParametricSelfFeature
    snapshot: GovernedCorpusSnapshot
    scratch_dir: Path
    scratch_identities: Mapping[Path, tuple[int, int]]
    candidate_path: str
    artifact_consumer: Mapping[str, object]
    run_nonce: str
    evidence_runner_revision: str


@dataclass(frozen=True, slots=True)
class _BackendDrillObservation:
    """Private result from one independently owned physical erasure drill."""

    backend: str
    observation: Mapping[str, object]
    identity_digest: str
    drill_digest: str
    run_nonce: str
    evidence_runner_revision: str


@dataclass(slots=True)
class KiteErasureBackend:
    """One runner-owned storage authority for an isolated evidence backend.

    A feature factory receives this typed capability rather than a database
    path or DSN.  It can open exactly one storage instance for the generated
    test identity.  The runner later verifies that the returned feature uses
    that exact instance, so a factory cannot substitute an ambient SQLite file
    or caller-selected PostgreSQL database.
    """

    backend: str
    trusted_scratch_root: Path
    scratch_dir: Path
    disposable_postgres: DisposablePostgresDatabase | None = None
    _storage: object | None = None
    _sqlite_state_path: Path | None = None
    _sqlite_state_identity: tuple[int, int] | None = None
    _closed: bool = False

    def __post_init__(self) -> None:
        if self.backend == "sqlite" and self.disposable_postgres is None:
            return
        if self.backend == "postgres" and isinstance(
            self.disposable_postgres, DisposablePostgresDatabase
        ):
            return
        raise ExternalReleaseEvidenceError(
            "external evidence backend must be isolated SQLite or core-created PostgreSQL"
        )

    @property
    def agent_id(self) -> str:
        """The generated test identity the factory must use for this backend."""
        return f"did:kestrel:parametric-self-release:{self.backend}:{_digest(str(self.scratch_dir))[:24]}"

    async def open_storage(self) -> object:
        """Create the only storage instance accepted for this backend drill."""
        if self._closed or self._storage is not None:
            raise ExternalReleaseEvidenceError("external evidence backend storage is unavailable")
        from kestrel_sovereign.storage.async_assertion_store import (
            _issue_assertion_tenant_capability,
        )
        from kestrel_sovereign.storage.async_storage import AsyncStorage

        assertion_capability = _issue_assertion_tenant_capability(self.agent_id)

        if self.backend == "sqlite":
            root_identity = _lstat_private_directory(self.trusted_scratch_root)
            state_path = self.trusted_scratch_root / (
                f".parametric-self-release-{secrets.token_hex(16)}.sqlite"
            )
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(state_path, flags, 0o600)
                os.close(descriptor)
                metadata = state_path.lstat()
            except OSError as error:
                raise ExternalReleaseEvidenceError(
                    "external evidence SQLite state could not be created"
                ) from error
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_ISLNK(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) & 0o077
            ):
                raise ExternalReleaseEvidenceError("external evidence SQLite state is not private")
            _lstat_private_directory(self.trusted_scratch_root, expected_identity=root_identity)
            self._sqlite_state_path = state_path
            self._sqlite_state_identity = (metadata.st_dev, metadata.st_ino)
            storage = AsyncStorage(
                db_path=str(state_path),
                backend="sqlite",
                agent_id=self.agent_id,
                _assertion_tenant_capability=assertion_capability,
            )
        else:
            assert self.disposable_postgres is not None
            storage = AsyncStorage(
                backend="postgres",
                dsn=self.disposable_postgres.dsn,
                agent_id=self.agent_id,
                _assertion_tenant_capability=assertion_capability,
            )
        self._storage = storage
        try:
            await storage.initialize()
        except BaseException:
            await self.close()
            raise
        return storage

    def validate_feature(self, feature: ParametricSelfFeature) -> None:
        storage = getattr(getattr(feature, "agent", None), "storage", None)
        if self._storage is None or storage is not self._storage:
            raise ExternalReleaseEvidenceError(
                "external evidence feature did not use the runner-owned backend"
            )
        if getattr(storage, "backend_type", None) != self.backend:
            raise ExternalReleaseEvidenceError(
                "external evidence feature backend does not match its isolated drill"
            )
        if getattr(feature.agent, "agent_id", None) != self.agent_id:
            raise ExternalReleaseEvidenceError(
                "external evidence feature identity does not match its isolated drill"
            )
        if getattr(feature.agent, "is_test_instance", False) is not True:
            raise ExternalReleaseEvidenceError(
                "external evidence requires an isolated Kite ParametricSelfFeature"
            )

    async def close(self) -> None:
        """Close the owned storage and remove the exact SQLite state file."""
        if self._closed:
            return
        self._closed = True
        storage, self._storage = self._storage, None
        close = getattr(storage, "close", None)
        try:
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await result
        finally:
            path, identity = self._sqlite_state_path, self._sqlite_state_identity
            if path is not None and identity is not None:
                try:
                    metadata = path.lstat()
                except OSError as error:
                    raise ExternalReleaseEvidenceError(
                        "external evidence SQLite state cleanup could not be verified"
                    ) from error
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or stat.S_ISLNK(metadata.st_mode)
                    or (metadata.st_dev, metadata.st_ino) != identity
                ):
                    raise ExternalReleaseEvidenceError(
                        "external evidence SQLite state changed before cleanup"
                    )
                try:
                    path.unlink()
                except OSError as error:
                    raise ExternalReleaseEvidenceError(
                        "external evidence SQLite state cleanup failed"
                    ) from error
                if path.exists():
                    raise ExternalReleaseEvidenceError(
                        "external evidence SQLite state cleanup could not be verified"
                    )


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
    selected = tuple(
        spec for spec in release_gate_specs() if spec.category == "external_adapter"
    )
    if tuple(spec.gate_id for spec in selected) != EXTERNAL_GATE_IDS:
        raise ExternalReleaseEvidenceError("core external release gate catalog is incomplete")
    result = {spec.gate_id: spec for spec in selected}
    if any(
        spec.category != "external_adapter"
        or spec.runner.runner_id != "external_ci"
        or spec.owner != "parametric_self"
        or spec.correlation is None
        or spec.environment.backend != "dual_backend"
        or spec.environment.mode != "external_adapter"
        for spec in result.values()
    ):
        raise ExternalReleaseEvidenceError("core external release gate catalog does not match the adapter contract")
    correlations = {
        (spec.correlation.drill_id, spec.correlation.drill_digest)
        for spec in result.values()
        if spec.correlation is not None
    }
    if len(correlations) != 1:
        raise ExternalReleaseEvidenceError("core external gates do not share one erasure drill")
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
        if tuple(record.gate_id for record in self.records) != EXTERNAL_GATE_IDS:
            raise ExternalReleaseEvidenceError(
                "external evidence must contain each declared adapter gate once in order"
            )
        by_gate = {record.gate_id: record for record in self.records}
        if len(by_gate) != len(self.records):
            raise ExternalReleaseEvidenceError("external evidence must contain each declared adapter gate once")
        artifacts: set[tuple[str, str]] = set()
        correlations: set[tuple[str, str]] = set()
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
            if record.artifact is None:
                raise ExternalReleaseEvidenceError("external record is missing its signed artifact")
            artifacts.add((record.artifact.artifact_ref, record.artifact.artifact_digest))
            assert spec.correlation is not None
            correlations.add((spec.correlation.drill_id, spec.correlation.drill_digest))
        if len(artifacts) != len(self.records):
            raise ExternalReleaseEvidenceError("external evidence records must retain distinct signed artifacts")
        if len(correlations) != 1:
            raise ExternalReleaseEvidenceError("external evidence records must share one erasure drill")
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
        if tuple(item.gate_id for item in self.report.attestations) != _EXTERNAL_CAPABILITY_GATE_IDS:
            raise ExternalReleaseEvidenceError(
                "external report must attest exactly the three external capability stages in order"
            )
        report_by_gate = {item.gate_id: item for item in self.report.attestations}
        if len(report_by_gate) != len(self.report.attestations):
            raise ExternalReleaseEvidenceError("external report must contain each capability gate once")
        for gate_id in _EXTERNAL_CAPABILITY_GATE_IDS:
            record = by_gate[gate_id]
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
        """Atomically publish one complete envelope to a fresh private path."""
        if overwrite:
            raise ExternalReleaseEvidenceError(
                "external evidence envelopes are immutable and cannot be replaced"
            )
        target, parent_identity = _validate_cli_output(Path(output))
        _write_cli_envelope_atomic(self, target, parent_identity)


class ParametricSelfKiteErasureHook:
    """Two-phase, server-owned Kite hook for the external erasure drill.

    ``prepare`` creates and verifies a real governed candidate *before* any
    deletion.  The isolated agent's server then calls
    ``erase_prepared_assertion`` (or performs its own scoped physical erase),
    and ``observe`` proves that exact candidate became invalid and can no
    longer be served.  A preparation is an opaque one-shot capability; callers
    never receive the assertion or snapshot needed to substitute a different
    deletion.
    """

    def __init__(self, runner: ParametricSelfExternalEvidenceRunner) -> None:
        self._runner = runner
        self._prepared: dict[str, _PreparedKiteDrill] = {}

    async def prepare(
        self,
        feature: ParametricSelfFeature,
        *,
        scratch_dir: Path,
        trusted_scratch_root: Path,
        run_nonce: str,
    ) -> KiteErasurePreparation:
        """Establish candidate and served eligibility before physical erasure."""
        if (
            not isinstance(run_nonce, str)
            or len(run_nonce) != _FRESHNESS_NONCE_BYTES * 2
            or any(character not in "0123456789abcdef" for character in run_nonce)
        ):
            raise ExternalReleaseEvidenceError("external evidence requires a verifier-issued nonce")
        if getattr(feature, "_active_adapter_path", None) is not None:
            raise ExternalReleaseEvidenceError(
                "external drill requires an isolated feature with no served adapter"
            )
        runner_revision = _resolve_clean_evidence_runner_revision()
        scratch, candidate, corpus_dir, identities = _prepare_private_scratch_tree(
            Path(scratch_dir), Path(trusted_scratch_root)
        )
        candidate_path: str | None = None
        artifact_consumer: Mapping[str, object] = MappingProxyType(
            {
                "consumer_id": self._runner._identity.issuer_id,
                "consumer_key_id": self._runner._identity.key_id,
                "consumer_public_key": self._runner._identity.public_key,
                "retention_seconds": 300.0,
            }
        )
        try:
            snapshot, problem = await feature._request_governed_snapshot(
                artifact_consumer=artifact_consumer
            )
            if problem or not isinstance(snapshot, GovernedCorpusSnapshot):
                raise ExternalReleaseEvidenceError(
                    "core governed corpus snapshot is unavailable or invalid"
                )
            if not snapshot.examples:
                raise ExternalReleaseEvidenceError(
                    "external erasure drill requires a non-empty governed corpus"
                )
            _verify_private_scratch_tree(scratch, identities)
            stats = build_corpus(
                None,
                str(corpus_dir),
                governed_snapshot=snapshot,
                manifest_dir=str(candidate),
            )
            _verify_private_scratch_tree(scratch, identities)
            if stats.from_facts <= 0 or not stats.assertion_lineage:
                raise ExternalReleaseEvidenceError(
                    "core governed corpus did not contribute an erasure-tracked example"
                )
            manifest, manifest_problem = feature._manifest_lineage(str(candidate))
            receipt = feature._manifest_receipt_stamp(manifest or {}) if manifest else None
            if manifest_problem or receipt is None:
                raise ExternalReleaseEvidenceError("candidate lineage receipt could not be established")

            candidate_path = str(candidate)
            feature._adapter_lineage[candidate_path] = {**receipt, "state": "candidate"}
            feature._active_adapter_path = candidate_path
            feature._live_corpus_snapshot = snapshot

            # This first verification is deliberately before the erasure. It
            # proves that this exact candidate was both candidate-eligible and
            # serve-eligible using the same server-owned snapshot.  The
            # verifier consumes an in-memory delta base, so restore the exact
            # snapshot for the post-erasure observation below.
            eligibility_problem = await feature._verify_adapter_lineage(
                candidate_path,
                before_promotion=True,
                artifact_consumer=artifact_consumer,
            )
            if eligibility_problem or feature._active_adapter_path != candidate_path:
                raise ExternalReleaseEvidenceError(
                    "candidate was not eligible for serving before erasure"
                )
            feature._adapter_lineage[candidate_path].update(
                {
                    "state": "served",
                    "candidate_eligibility": "accepted",
                    "served_eligibility": "accepted",
                }
            )
            feature._live_corpus_snapshot = snapshot
            drill_id = secrets.token_hex(32)
            self._prepared[drill_id] = _PreparedKiteDrill(
                feature=feature,
                snapshot=snapshot,
                scratch_dir=scratch,
                scratch_identities=identities,
                candidate_path=candidate_path,
                artifact_consumer=artifact_consumer,
                run_nonce=run_nonce,
                evidence_runner_revision=runner_revision,
            )
            return KiteErasurePreparation(drill_id, run_nonce, runner_revision)
        except BaseException:
            if candidate_path is not None:
                # A failed pre-erase eligibility check must not leave an
                # active pointer to a candidate whose private manifest is
                # about to be removed.
                await feature._quarantine_adapter(
                    candidate_path, "external erasure drill preparation failed"
                )
            _cleanup_private_scratch_tree(scratch, identities)
            raise

    def _state(self, preparation: KiteErasurePreparation) -> _PreparedKiteDrill:
        if not isinstance(preparation, KiteErasurePreparation):
            raise ExternalReleaseEvidenceError("external erasure observation requires a prepared Kite drill")
        state = self._prepared.get(preparation._drill_id)
        if (
            state is None
            or preparation.run_nonce != state.run_nonce
            or preparation.evidence_runner_revision != state.evidence_runner_revision
        ):
            raise ExternalReleaseEvidenceError("external erasure preparation is unknown or already consumed")
        return state

    async def erase_prepared_assertion(self, preparation: KiteErasurePreparation) -> None:
        """Invoke the server's scoped physical erase for this prepared drill."""
        state = self._state(preparation)
        await self._runner._erase_snapshot_assertion(
            state.feature, state.snapshot, state.run_nonce
        )

    async def observe(
        self, preparation: KiteErasurePreparation, *, backend: str
    ) -> _BackendDrillObservation:
        """Verify one backend's post-erasure state without signing anything.

        The runner deliberately withholds signatures until it has independently
        observed this same drill on both required storage engines and verified
        their cleanup.  A single backend can therefore never become an
        externally consumable release claim.
        """
        state = self._state(preparation)
        try:
            # Re-establish the exact pre-erase base held by the hook, rather
            # than accepting a caller-mutated feature cache or a later fresh
            # snapshot as the observation's lineage source.
            state.feature._live_corpus_snapshot = state.snapshot
            invalidation_reason = await state.feature._verify_adapter_lineage(
                state.candidate_path,
                artifact_consumer=state.artifact_consumer,
            )
            lineage = state.feature._adapter_lineage.get(state.candidate_path, {})
            if not invalidation_reason or lineage.get("state") != "invalid":
                await state.feature._quarantine_adapter(
                    state.candidate_path,
                    "external erasure observation did not establish invalidation",
                )
                raise ExternalReleaseEvidenceError(
                    "erasure did not invalidate the governed corpus candidate"
                )
            if state.feature._active_adapter_path is not None:
                raise ExternalReleaseEvidenceError(
                    "erasure did not reject served adapter eligibility"
                )
            specs = _external_specs()
            common_drill = next(iter(specs.values())).correlation
            if common_drill is None:
                raise ExternalReleaseEvidenceError("core external erasure drill is unavailable")
            observation: Mapping[str, object] = MappingProxyType(
                {"erased_count": 1, "remaining_count": 0}
            )
            return _BackendDrillObservation(
                backend=backend,
                observation=observation,
                identity_digest=_digest(
                    {
                        "capability_id": _CAPABILITY_ID,
                        "repository": PARAMETRIC_SELF_EVIDENCE_REPOSITORY,
                        "capability_source_revision": PARAMETRIC_SELF_CAPABILITY_SOURCE_REVISION,
                        "core_release_evidence_contract_digest": CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST,
                    }
                ),
                drill_digest=_digest(
                    {
                        "drill_id": common_drill.drill_id,
                        "drill_digest": common_drill.drill_digest,
                        "schema": tuple(observation),
                    }
                ),
                run_nonce=state.run_nonce,
                evidence_runner_revision=state.evidence_runner_revision,
            )
        finally:
            self._prepared.pop(preparation._drill_id, None)
            _cleanup_private_scratch_tree(state.scratch_dir, state.scratch_identities)

    async def abort(self, preparation: KiteErasurePreparation) -> None:
        """Fail closed and clear a prepared drill that will not be observed."""
        state = self._state(preparation)
        self._prepared.pop(preparation._drill_id, None)
        try:
            await state.feature._quarantine_adapter(
                state.candidate_path, "external erasure drill aborted before observation"
            )
        finally:
            _cleanup_private_scratch_tree(state.scratch_dir, state.scratch_identities)


class ParametricSelfExternalEvidenceRunner:
    """Execute one real erasure drill on each required isolated backend.

    The only factory input is a runner-owned :class:`KiteErasureBackend`.
    PostgreSQL arrives from core's disposable-database authority, never from a
    caller DSN.  No record is signed until SQLite and PostgreSQL each complete
    prepare → physical erase → observe, their observations agree, and both
    database lifecycles have been cleaned up.
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
        feature_factory: FeatureFactory,
        *,
        scratch_dir: Path,
        trusted_scratch_root: Path,
        run_nonce: str,
    ) -> ExternalReleaseEvidenceEnvelope:
        """Run the sealed dual-backend drill and produce its one envelope."""
        if not callable(feature_factory):
            raise ExternalReleaseEvidenceError(
                "external evidence requires a dual-backend Kite feature factory"
            )
        if (
            not isinstance(run_nonce, str)
            or len(run_nonce) != _FRESHNESS_NONCE_BYTES * 2
            or any(character not in "0123456789abcdef" for character in run_nonce)
        ):
            raise ExternalReleaseEvidenceError("external evidence requires a verifier-issued nonce")
        scratch_dir, trusted_scratch_root = Path(scratch_dir), Path(trusted_scratch_root)
        if scratch_dir.parent != trusted_scratch_root:
            raise ExternalReleaseEvidenceError(
                "external evidence dual-backend scratch directory must be directly inside its trusted root"
            )
        # Fail before the SQLite feature can create any governed material if
        # the core PostgreSQL authority is unavailable or refuses to create a
        # fresh disposable database.
        async with await DisposablePostgresDatabase.create() as database:
            sqlite_backend = KiteErasureBackend(
                "sqlite", trusted_scratch_root, scratch_dir.with_name(f"{scratch_dir.name}-sqlite")
            )
            postgres_backend = KiteErasureBackend(
                "postgres",
                trusted_scratch_root,
                scratch_dir.with_name(f"{scratch_dir.name}-postgres"),
                database,
            )
            try:
                sqlite = await self._run_backend(
                    feature_factory, sqlite_backend, run_nonce=run_nonce
                )
                postgres = await self._run_backend(
                    feature_factory, postgres_backend, run_nonce=run_nonce
                )
            finally:
                # ``_run_backend`` closes its own storage. The explicit calls
                # are idempotent and cover factories that fail before a hook.
                await sqlite_backend.close()
                await postgres_backend.close()
        observation, drill_semantics_digest, runner_revision = self._agree_backends(
            sqlite, postgres, run_nonce=run_nonce
        )
        return self._seal_envelope(
            observation,
            run_nonce=run_nonce,
            evidence_runner_revision=runner_revision,
            drill_semantics_digest=drill_semantics_digest,
        )

    async def _run_backend(
        self,
        feature_factory: FeatureFactory,
        backend: KiteErasureBackend,
        *,
        run_nonce: str,
    ) -> _BackendDrillObservation:
        """Drive one typed backend through the only permitted phase ordering."""
        hook = ParametricSelfKiteErasureHook(self)
        preparation: KiteErasurePreparation | None = None
        try:
            feature = feature_factory(backend)
            if inspect.isawaitable(feature):
                feature = await feature
            from .feature import ParametricSelfFeature

            if not isinstance(feature, ParametricSelfFeature):
                raise ExternalReleaseEvidenceError(
                    "Kite feature factory did not return a ParametricSelfFeature"
                )
            backend.validate_feature(feature)
            preparation = await hook.prepare(
                feature,
                scratch_dir=backend.scratch_dir,
                trusted_scratch_root=backend.trusted_scratch_root,
                run_nonce=run_nonce,
            )
            await hook.erase_prepared_assertion(preparation)
            return await hook.observe(preparation, backend=backend.backend)
        except BaseException:
            if preparation is not None:
                try:
                    await hook.abort(preparation)
                except ExternalReleaseEvidenceError:
                    pass
            raise
        finally:
            await backend.close()

    @staticmethod
    def _agree_backends(
        sqlite: _BackendDrillObservation,
        postgres: _BackendDrillObservation,
        *,
        run_nonce: str,
    ) -> tuple[Mapping[str, object], str, str]:
        """Require matching semantic observations from the two named engines."""
        observations = (sqlite, postgres)
        if tuple(item.backend for item in observations) != ("sqlite", "postgres"):
            raise ExternalReleaseEvidenceError(
                "external evidence did not observe the required isolated backend identities"
            )
        if any(item.run_nonce != run_nonce for item in observations):
            raise ExternalReleaseEvidenceError("external evidence backend observations use mixed nonces")
        revisions = {item.evidence_runner_revision for item in observations}
        if len(revisions) != 1 or not _is_full_commit(next(iter(revisions), "")):
            raise ExternalReleaseEvidenceError(
                "external evidence backend observations use mixed runner revisions"
            )
        identity_digests = {item.identity_digest for item in observations}
        drill_digests = {item.drill_digest for item in observations}
        schemas = {tuple(item.observation) for item in observations}
        if len(identity_digests) != 1 or len(drill_digests) != 1 or len(schemas) != 1:
            raise ExternalReleaseEvidenceError(
                "external evidence backend observations disagree on identity, drill, or schema"
            )
        if any(
            set(item.observation) != {"erased_count", "remaining_count"}
            or type(item.observation["erased_count"]) is not int
            or type(item.observation["remaining_count"]) is not int
            or item.observation["erased_count"] <= 0
            or item.observation["remaining_count"] != 0
            for item in observations
        ):
            raise ExternalReleaseEvidenceError(
                "external evidence backend erasure observations are not passing"
            )
        # Each independently observed backend has the same fixed semantic
        # shape. The external contract sees only their safe aggregate.
        return (
            MappingProxyType(
                {
                    "erased_count": sum(item.observation["erased_count"] for item in observations),
                    "remaining_count": 0,
                }
            ),
            next(iter(drill_digests)),
            next(iter(revisions)),
        )

    def _seal_envelope(
        self,
        observation: Mapping[str, object],
        *,
        run_nonce: str,
        evidence_runner_revision: str,
        drill_semantics_digest: str,
    ) -> ExternalReleaseEvidenceEnvelope:
        """Sign the declared contract only after both private drills succeed."""
        specs = _external_specs()
        records = tuple(
            self._record(
                specs[gate_id],
                observation,
                run_nonce,
                evidence_runner_revision,
                drill_semantics_digest,
            )
            for gate_id in EXTERNAL_GATE_IDS
        )
        by_gate = {record.gate_id: record for record in records}
        attestations: list[ExternalGateAttestation] = []
        for gate_id in _EXTERNAL_CAPABILITY_GATE_IDS:
            record = by_gate[gate_id]
            artifact = record.artifact
            drill = specs[gate_id].correlation
            if record.run_digest is None or artifact is None or drill is None:
                raise ExternalReleaseEvidenceError(
                    "signed external record is missing a core binding"
                )
            attestations.append(
                ExternalGateAttestation(
                    gate_id=gate_id,
                    gate_spec_digest=specs[gate_id].digest,
                    result_digest=record.run_digest,
                    artifact=artifact,
                    drill=drill,
                )
            )
        report = ExternalCapabilityReport.attest(
            capability_id=_CAPABILITY_ID,
            repository=PARAMETRIC_SELF_EVIDENCE_REPOSITORY,
            capability_source_revision=PARAMETRIC_SELF_CAPABILITY_SOURCE_REVISION,
            evidence_runner_revision=evidence_runner_revision,
            core_release_evidence_contract_digest=CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST,
            attestations=tuple(attestations),
            run_nonce=run_nonce,
        )
        return ExternalReleaseEvidenceEnvelope(
            core_release_evidence_contract_digest=CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST,
            repository=PARAMETRIC_SELF_EVIDENCE_REPOSITORY,
            capability_source_revision=PARAMETRIC_SELF_CAPABILITY_SOURCE_REVISION,
            evidence_runner_revision=evidence_runner_revision,
            run_nonce=run_nonce,
            records=records,
            report=report,
        )

    @staticmethod
    async def _erase_snapshot_assertion(
        feature: ParametricSelfFeature,
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
        drill_semantics_digest: str,
    ) -> EvidenceRecord:
        # The artifact is a digest of only catalog-bound aggregate fields; it
        # cannot be used to smuggle an assertion, tenant, filesystem path, or
        # caller-controlled log into core's release report.
        artifact_digest = _digest(
            {
                "core_release_evidence_contract_digest": CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST,
                "repository": PARAMETRIC_SELF_EVIDENCE_REPOSITORY,
                "capability_source_revision": PARAMETRIC_SELF_CAPABILITY_SOURCE_REVISION,
                "gate_id": spec.gate_id,
                "gate_spec_digest": spec.digest,
                "observation": dict(observation),
                "run_nonce": run_nonce,
                "drill_semantics_digest": drill_semantics_digest,
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


def _load_private_signing_key(path: Path) -> object:
    """Load a CI signing seed only from an owner-private regular file."""
    try:
        metadata = path.lstat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise ExternalReleaseEvidenceError("external signing key file is not private")
        raw = path.read_bytes().strip()
    except OSError as error:
        raise ExternalReleaseEvidenceError("external signing key file is unavailable") from error
    try:
        seed = bytes.fromhex(raw.decode("ascii")) if len(raw) == 64 else raw
        if len(seed) != 32:
            raise ValueError("wrong seed size")
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        return Ed25519PrivateKey.from_private_bytes(seed)
    except (UnicodeDecodeError, ValueError) as error:
        raise ExternalReleaseEvidenceError("external signing key file is invalid") from error


def _load_kite_feature_factory(factory_reference: str) -> FeatureFactory:
    """Load the one-argument isolated-backend factory used by the CLI.

    The factory receives the runner-owned typed backend capability only. In
    particular, the command line never accepts an agent path, backend name,
    SQLite path, PostgreSQL DSN, assertion ID, or erasure callback.
    """
    module_name, separator, attribute = factory_reference.partition(":")
    if not module_name or not separator or not attribute or "." not in module_name:
        raise ExternalReleaseEvidenceError(
            "Kite feature factory must be a fully-qualified module:callable reference"
        )
    try:
        factory = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as error:
        raise ExternalReleaseEvidenceError("Kite feature factory is unavailable") from error
    if not callable(factory):
        raise ExternalReleaseEvidenceError("Kite feature factory is not callable")
    try:
        signature = inspect.signature(factory)
        parameters = tuple(signature.parameters.values())
        if (
            len(parameters) != 1
            or parameters[0].kind
            not in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        ):
            raise TypeError("factory signature is not one typed backend capability")
        signature.bind(object())
    except (TypeError, ValueError) as error:
        raise ExternalReleaseEvidenceError(
            "Kite feature factory must accept exactly the runner-owned backend capability"
        ) from error
    return factory


def _validate_cli_output(output: Path) -> tuple[Path, tuple[int, int]]:
    """Validate the output target before the drill can erase anything."""
    output = Path(output)
    if not output.is_absolute() or output.name in {"", ".", ".."}:
        raise ExternalReleaseEvidenceError(
            "external evidence output must be a fresh absolute path"
        )
    try:
        parent_identity = _lstat_private_directory(output.parent)
    except ExternalReleaseEvidenceError as error:
        raise ExternalReleaseEvidenceError(
            "external evidence output parent is unavailable or not private"
        ) from error
    try:
        output.lstat()
    except FileNotFoundError:
        return output, parent_identity
    except OSError as error:
        raise ExternalReleaseEvidenceError(
            "external evidence output target is unavailable"
        ) from error
    raise ExternalReleaseEvidenceError(
        "external evidence output must be a fresh absolute path"
    )


def _write_cli_envelope_atomic(
    envelope: ExternalReleaseEvidenceEnvelope,
    output: Path,
    parent_identity: tuple[int, int],
) -> None:
    """Publish a complete envelope without following or replacing a path.

    The private parent is opened and identity-pinned. A complete, fsynced 0600
    temporary inode is hard-linked into the fresh destination name, which is
    an atomic no-replace operation, then the temporary name is removed.
    """
    _lstat_private_directory(output.parent, expected_identity=parent_identity)
    parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    temp_name = f".{output.name}.tmp-{secrets.token_hex(16)}"
    parent_fd: int | None = None
    temp_created = False
    target_created = False
    try:
        parent_fd = os.open(output.parent, parent_flags)
        parent_metadata = os.fstat(parent_fd)
        if (
            (parent_metadata.st_dev, parent_metadata.st_ino) != parent_identity
            or parent_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(parent_metadata.st_mode) & 0o077
        ):
            raise ExternalReleaseEvidenceError(
                "external evidence output parent changed before publication"
            )
        try:
            os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ExternalReleaseEvidenceError(
                "external evidence output target is no longer fresh"
            )

        open_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        file_fd = os.open(temp_name, open_flags, 0o600, dir_fd=parent_fd)
        temp_created = True
        with os.fdopen(file_fd, "w", encoding="utf-8") as stream:
            stream.write(_canonical_json(envelope.to_mapping()) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        _lstat_private_directory(output.parent, expected_identity=parent_identity)
        os.link(
            temp_name,
            output.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        target_created = True
        os.unlink(temp_name, dir_fd=parent_fd)
        temp_created = False
        os.fsync(parent_fd)
        _lstat_private_directory(output.parent, expected_identity=parent_identity)
    except ExternalReleaseEvidenceError:
        raise
    except OSError as error:
        raise ExternalReleaseEvidenceError(
            "external evidence output could not be published safely"
        ) from error
    finally:
        if parent_fd is not None:
            if target_created:
                try:
                    # Keep a successfully-published target unless the parent
                    # path changed after publication; then it is not the path
                    # the verifier approved.
                    _lstat_private_directory(
                        output.parent, expected_identity=parent_identity
                    )
                except ExternalReleaseEvidenceError:
                    try:
                        os.unlink(output.name, dir_fd=parent_fd)
                    except OSError:
                        pass
            if temp_created:
                try:
                    os.unlink(temp_name, dir_fd=parent_fd)
                except OSError:
                    pass
            os.close(parent_fd)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parametric-self-release-evidence",
        description=(
            "Run the isolated Kite prepare → physical erase → observe evidence drill. "
            "The feature factory must create a test agent, never a production agent."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--feature-factory", required=True)
    parser.add_argument("--signing-key-file", type=Path, required=True)
    parser.add_argument("--issuer-id", required=True)
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--run-nonce", required=True)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--trusted-scratch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


async def _run_cli(args: argparse.Namespace) -> ExternalReleaseEvidenceEnvelope:
    # Validate all output invariants before loading the agent, preparing a
    # candidate, or invoking the irreversible physical erasure.
    output, output_parent_identity = _validate_cli_output(args.output)
    feature_factory = _load_kite_feature_factory(args.feature_factory)
    identity = CatalogSigningIdentity(
        issuer_id=args.issuer_id,
        key_id=args.key_id,
        private_key=_load_private_signing_key(args.signing_key_file),
        source=ExecutionSource.EXTERNAL_CI,
    )
    runner = ParametricSelfExternalEvidenceRunner(identity)
    envelope = await runner.run(
        feature_factory,
        scratch_dir=args.scratch_dir,
        trusted_scratch_root=args.trusted_scratch_root,
        run_nonce=args.run_nonce,
    )
    _write_cli_envelope_atomic(envelope, output, output_parent_identity)
    return envelope


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for externally operated, isolated Kite evidence runs."""
    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    try:
        asyncio.run(_run_cli(args))
    except ExternalReleaseEvidenceError as error:
        parser.exit(2, f"external release evidence refused: {error}\n")
    # Do not print the output path or live-agent identity; callers supplied
    # those and the resulting envelope is the CI artifact.
    print("external release evidence completed; core policy verification required")
    return 0


__all__ = [
    "CORE_RELEASE_EVIDENCE_CONTRACT_DIGEST",
    "EXTERNAL_GATE_IDS",
    "ExternalReleaseEvidenceEnvelope",
    "ExternalReleaseEvidenceError",
    "KiteErasureBackend",
    "KiteErasurePreparation",
    "ParametricSelfExternalEvidenceRunner",
    "ParametricSelfKiteErasureHook",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - entrypoint is tested via main().
    raise SystemExit(main())
