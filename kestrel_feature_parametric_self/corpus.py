"""Build a transient MLX corpus from a governed assertion snapshot.

Every example arrives in a host-provided governed assertion snapshot: this
module never opens a database, receives a database handle, or reconstructs
assertion eligibility itself.  Reflections remain this feature's distinct
symbolic-self source, but only as governed assertions read from a reflection
source (``REFLECTION_SOURCE_KIND``), so they carry the same consent, source and
exclusion decisions as every other example.  A reflection is admitted only
when its provenance names the session(s) it was derived from and the snapshot's
policy can exclude sessions; otherwise it is left out, never admitted by
default.  The durable sidecar records lineage hashes and identifiers, never
prompt/answer text.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REFLECTION_SOURCE_KIND = "agent_reflection"
CORPUS_POLICY_VERSION = "parametric-self-corpus-v2"
MANIFEST_FILENAME = "corpus_manifest.json"


class GovernedCorpusRequiredError(ValueError):
    """A training caller attempted to build without the host corpus boundary."""


@dataclass(frozen=True)
class CorpusStats:
    """Content-free result of a deterministic corpus build."""

    total: int
    train: int
    valid: int
    from_insights: int
    from_facts: int
    insights_unresolved: int
    out_dir: str
    manifest_path: str
    manifest_hash: str
    semantic_checkpoint_generation: int
    semantic_checkpoint_id: str | None
    snapshot_hash: str
    policy_digest: str
    assertion_lineage: tuple[tuple[str, str], ...]


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _term_value(value: object) -> str:
    """Render the public assertion value without depending on concrete term classes."""
    for attribute in ("lexical_form", "value"):
        candidate = getattr(value, attribute, None)
        if isinstance(candidate, str) and candidate:
            return candidate
    return str(value)


def _human_term(value: object) -> str:
    raw = _term_value(value)
    if "#" in raw:
        raw = raw.rsplit("#", 1)[1]
    elif "/" in raw:
        raw = raw.rstrip("/").rsplit("/", 1)[-1]
    return raw.replace("_", " ").strip()


def _reflection_sessions(sources: tuple[Any, ...]) -> tuple[str, ...] | None:
    """The sessions a reflection was derived from, or None if one source names none.

    Called only under a policy that can exclude sessions; every core with such
    a policy also names sessions in source provenance.
    """
    from kestrel_sovereign.knowledge import SourceReferenceKind

    sessions: set[str] = set()
    for source in sources:
        if getattr(source, "source_kind", None) != REFLECTION_SOURCE_KIND:
            continue
        named = {
            identifier
            for kind, identifier in getattr(source, "references", ())
            if kind is SourceReferenceKind.SESSION
        }
        if not named:
            return None
        sessions |= named
    return tuple(sorted(sessions))


def _governed_examples(snapshot: Any) -> tuple[list[dict[str, Any]], int]:
    """Translate host-approved examples; return them with the unresolved-reflection count."""
    if snapshot is None or not getattr(snapshot, "verified", False):
        raise GovernedCorpusRequiredError("governed_corpus_snapshot_required")
    excludes = getattr(getattr(snapshot, "policy", None), "excludes", None)
    examples: list[dict[str, Any]] = []
    unresolved_reflections = 0
    for item in tuple(getattr(snapshot, "examples", ())):
        decision = getattr(item, "decision", None)
        if not bool(getattr(decision, "included", False)):
            raise GovernedCorpusRequiredError("governed_corpus_contains_ineligible_example")
        assertion = getattr(item, "assertion", None)
        assertion_id = getattr(assertion, "assertion_id", None)
        revision_id = getattr(assertion, "revision_id", None)
        content_hash = getattr(item, "content_hash", None)
        if not all(isinstance(value, str) and value for value in (assertion_id, revision_id, content_hash)):
            raise GovernedCorpusRequiredError("governed_corpus_example_lineage_invalid")
        subject = _human_term(getattr(assertion, "subject", ""))
        predicate = _human_term(getattr(assertion, "predicate", ""))
        value = _term_value(getattr(assertion, "object", "")).strip()
        if not subject or not predicate or not value:
            raise GovernedCorpusRequiredError("governed_corpus_example_content_invalid")
        sources = tuple(getattr(item, "source_occurrences", ()))
        if callable(excludes) and excludes(assertion, sources):
            raise GovernedCorpusRequiredError("governed_corpus_contains_excluded_example")
        source_ids = tuple(
            sorted(
                str(getattr(source, "source_occurrence_id"))
                for source in sources
                if getattr(source, "source_occurrence_id", None)
            )
        )
        lineage: dict[str, Any] = {
            "assertion_id": assertion_id,
            "revision_id": revision_id,
            "content_hash": content_hash,
            "source_occurrence_ids": source_ids,
            "eligibility": getattr(getattr(decision, "reason", None), "value", "included"),
        }
        source_label = "governed_assertion"
        if any(getattr(source, "source_kind", None) == REFLECTION_SOURCE_KIND for source in sources):
            # Without a policy that can exclude sessions, a reflection cannot
            # be shown to be clear of an excluded one.
            sessions = _reflection_sessions(sources) if callable(excludes) else None
            if sessions is None:
                unresolved_reflections += 1
                continue
            source_label = "reflection"
            lineage["session_ids"] = sessions
        stable_identity = {
            "source": source_label,
            "assertion_id": assertion_id,
            "revision_id": revision_id,
            "content_hash": content_hash,
            "policy_version": CORPUS_POLICY_VERSION,
        }
        examples.append(
            {
                "example_id": _digest(stable_identity),
                "source": source_label,
                "lineage": lineage,
                "messages": [
                    {"role": "user", "content": f"What do you know about {subject}'s {predicate}?"},
                    {"role": "assistant", "content": value},
                ],
            }
        )
    return examples, unresolved_reflections


def _split_examples(examples: Iterable[dict[str, Any]], valid_every: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(examples, key=lambda item: item["example_id"])
    if valid_every <= 0:
        return ordered, []
    train: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for example in ordered:
        bucket = int(example["example_id"].rsplit(":", 1)[-1], 16) % valid_every
        (valid if bucket == 0 else train).append(example)
    # MLX requires a training row.  Preserve deterministic assignment even for
    # a one-row corpus by moving the lexicographically first held-out row.
    if not train and valid:
        train.append(valid.pop(0))
    return train, valid


def _write_jsonl(path: Path, examples: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps({"messages": example["messages"]}, ensure_ascii=False) + "\n")


def corpus_manifest_problem(manifest: dict[str, Any]) -> str | None:
    """Why a manifest cannot authorize its adapter under this corpus policy, if it can't.

    The manifest is the only record of what an adapter's weights absorbed, so
    an adapter is accepted only if its manifest was built under the current
    corpus policy and every example is a governed assertion revision.  A
    reflection must also name the session(s) it was derived from.  An adapter
    built under an earlier policy may hold reflection rows read straight from
    reflection's table, with no consent, source or exclusion decision; it must
    be rebuilt, never grandfathered.
    """
    if manifest.get("corpus_policy_version") != CORPUS_POLICY_VERSION:
        return "candidate corpus policy superseded; rebuild required"
    examples = manifest.get("examples")
    if not isinstance(examples, list):
        return "candidate manifest governed evidence is malformed"
    for example in examples:
        lineage = example.get("lineage") if isinstance(example, dict) else None
        if (
            not isinstance(lineage, dict)
            or example.get("source") not in ("governed_assertion", "reflection")
            or not isinstance(lineage.get("assertion_id"), str)
            or not isinstance(lineage.get("revision_id"), str)
        ):
            return "candidate manifest example lineage missing; rebuild required"
        if example["source"] == "reflection":
            sessions = lineage.get("session_ids")
            if (
                not isinstance(sessions, list)
                or not sessions
                or not all(isinstance(item, str) and item for item in sessions)
            ):
                return "candidate manifest reflection session lineage missing; rebuild required"
    return None


def _write_immutable_manifest(path: Path, manifest: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_hash = _digest(manifest)
    payload = dict(manifest, manifest_hash=manifest_hash)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(_canonical_json(payload) + "\n")
    except FileExistsError as error:
        raise GovernedCorpusRequiredError("candidate_manifest_already_exists") from error
    try:
        os.chmod(path, 0o444)
    except OSError:
        pass
    return manifest_hash


def build_corpus(
    out_dir: str,
    *,
    governed_snapshot: Any,
    manifest_dir: str,
    valid_every: int = 10,
) -> CorpusStats:
    """Build transient JSONL plus an immutable, content-free lineage sidecar.

    ``governed_snapshot`` must be the public capability result captured by the
    host after semantic maintenance.  It is mandatory even if this particular
    agent currently has zero accepted assertions: capability absence is never
    interpreted as permission to use an ungoverned source.
    """
    if valid_every < 0:
        raise ValueError("valid_every must be non-negative")
    all_examples, unresolved_reflections = _governed_examples(governed_snapshot)
    checkpoint = getattr(governed_snapshot, "checkpoint", None)
    policy = getattr(governed_snapshot, "policy", None)
    policy_digest = getattr(policy, "digest", None)
    snapshot_hash = getattr(governed_snapshot, "snapshot_hash", None)
    generation = getattr(checkpoint, "generation", None)
    checkpoint_tenant = getattr(checkpoint, "tenant_id", None)
    snapshot_tenant = getattr(governed_snapshot, "tenant_id", None)
    if (
        not isinstance(policy_digest, str)
        or not isinstance(snapshot_hash, str)
        or type(generation) is not int
        or not isinstance(checkpoint_tenant, str)
        or not checkpoint_tenant
        or checkpoint_tenant != snapshot_tenant
    ):
        raise GovernedCorpusRequiredError("governed_corpus_snapshot_metadata_invalid")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reflections = sum(1 for example in all_examples if example["source"] == "reflection")
    train, valid = _split_examples(all_examples, valid_every)
    split_by_id = {example["example_id"]: "train" for example in train}
    split_by_id.update({example["example_id"]: "valid" for example in valid})
    manifest_examples = [
        {
            "example_id": example["example_id"],
            "source": example["source"],
            "lineage": example["lineage"],
            "split": split_by_id[example["example_id"]],
        }
        for example in sorted(all_examples, key=lambda item: item["example_id"])
    ]
    manifest = {
        "schema_version": 1,
        "corpus_policy_version": CORPUS_POLICY_VERSION,
        "policy_digest": policy_digest,
        "snapshot_hash": snapshot_hash,
        "semantic_checkpoint": {
            "tenant_id": checkpoint_tenant,
            "generation": generation,
            "event_id": getattr(checkpoint, "latest_event_id", None),
        },
        "capability_versions": dict(sorted(dict(getattr(governed_snapshot, "capability_versions", {})).items())),
        "counts": {
            "total": len(all_examples), "train": len(train), "valid": len(valid),
            "reflection": reflections,
            "governed_assertion": len(all_examples) - reflections,
            "reflection_unresolved": unresolved_reflections,
        },
        "examples": manifest_examples,
    }
    manifest_path = Path(manifest_dir) / MANIFEST_FILENAME
    manifest_hash = _write_immutable_manifest(manifest_path, manifest)
    _write_jsonl(out / "train.jsonl", train)
    _write_jsonl(out / "valid.jsonl", valid)
    assertion_lineage = tuple(
        (example["lineage"]["assertion_id"], example["lineage"]["revision_id"])
        for example in manifest_examples
    )
    return CorpusStats(
        total=len(all_examples), train=len(train), valid=len(valid),
        from_insights=reflections, from_facts=len(all_examples) - reflections,
        insights_unresolved=unresolved_reflections, out_dir=str(out),
        manifest_path=str(manifest_path), manifest_hash=manifest_hash,
        semantic_checkpoint_generation=generation,
        semantic_checkpoint_id=getattr(checkpoint, "latest_event_id", None),
        snapshot_hash=snapshot_hash, policy_digest=policy_digest,
        assertion_lineage=assertion_lineage,
    )
