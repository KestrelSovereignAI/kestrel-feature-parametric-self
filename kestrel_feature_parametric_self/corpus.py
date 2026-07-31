"""Build a transient MLX corpus from reflections and governed assertions.

Reflections remain this feature's distinct symbolic-self input.  Factual
examples arrive only in a host-provided governed assertion snapshot: this
module never opens a fact table, receives a database handle, or reconstructs
assertion eligibility itself.  The durable sidecar records lineage hashes and
identifiers, never prompt/answer text.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


EXTERNALLY_GROUNDED_TYPES = ("failure", "success", "improvement")
CORPUS_POLICY_VERSION = "parametric-self-corpus-v1"
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


def _reflection_examples(db_path: str | None, grounded_only: bool) -> list[dict[str, Any]]:
    """Read reflection's own table only; no factual data is read from SQLite."""
    if not db_path or not Path(db_path).is_file():
        return []
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT id, type, title, description, suggested_action FROM reflection_insights "
            "WHERE title IS NOT NULL AND length(trim(title)) > 0"
        ).fetchall()
    finally:
        con.close()
    examples: list[dict[str, Any]] = []
    for insight_id, insight_type, title, description, suggested_action in rows:
        if grounded_only and insight_type not in EXTERNALLY_GROUNDED_TYPES:
            continue
        description = (description or "").strip()
        suggested_action = (suggested_action or "").strip()
        if not description and not suggested_action:
            continue
        answer = description
        if suggested_action:
            answer = f"{answer}\n\nWhat to do: {suggested_action}".strip()
        stable_identity = {
            "source": "reflection",
            "reflection_id": str(insight_id),
            "type": str(insight_type),
            "title": title.strip(),
            "content_hash": _digest([description, suggested_action]),
            "policy_version": CORPUS_POLICY_VERSION,
        }
        examples.append(
            {
                "example_id": _digest(stable_identity),
                "source": "reflection",
                "lineage": {"reflection_id": str(insight_id), "content_hash": stable_identity["content_hash"]},
                "messages": [
                    {"role": "user", "content": f"From your reflections, what did you learn about: {title.strip()}?"},
                    {"role": "assistant", "content": answer},
                ],
            }
        )
    return examples


def _governed_examples(snapshot: Any) -> list[dict[str, Any]]:
    """Translate immutable host-approved examples without second-guessing policy."""
    if snapshot is None or not getattr(snapshot, "verified", False):
        raise GovernedCorpusRequiredError("governed_corpus_snapshot_required")
    examples: list[dict[str, Any]] = []
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
        source_ids = tuple(
            sorted(
                str(getattr(source, "source_occurrence_id"))
                for source in tuple(getattr(item, "source_occurrences", ()))
                if getattr(source, "source_occurrence_id", None)
            )
        )
        stable_identity = {
            "source": "governed_assertion",
            "assertion_id": assertion_id,
            "revision_id": revision_id,
            "content_hash": content_hash,
            "policy_version": CORPUS_POLICY_VERSION,
        }
        examples.append(
            {
                "example_id": _digest(stable_identity),
                "source": "governed_assertion",
                "lineage": {
                    "assertion_id": assertion_id,
                    "revision_id": revision_id,
                    "content_hash": content_hash,
                    "source_occurrence_ids": source_ids,
                    "eligibility": getattr(getattr(decision, "reason", None), "value", "included"),
                },
                "messages": [
                    {"role": "user", "content": f"What do you know about {subject}'s {predicate}?"},
                    {"role": "assistant", "content": value},
                ],
            }
        )
    return examples


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
    db_path: str | None,
    out_dir: str,
    *,
    governed_snapshot: Any,
    manifest_dir: str,
    grounded_only: bool = True,
    valid_every: int = 10,
) -> CorpusStats:
    """Build transient JSONL plus an immutable, content-free lineage sidecar.

    ``governed_snapshot`` must be the public capability result captured by the
    host after semantic maintenance.  It is mandatory even if this particular
    agent currently has zero accepted assertions: capability absence is never
    interpreted as permission to use an ungoverned fact source.
    """
    if valid_every < 0:
        raise ValueError("valid_every must be non-negative")
    governed = _governed_examples(governed_snapshot)
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
    reflections = _reflection_examples(db_path, grounded_only)
    all_examples = [
        example for example in reflections + governed
        if example["messages"][1]["content"]
    ]
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
            "reflection": len(reflections), "governed_assertion": len(governed),
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
        if example["source"] == "governed_assertion"
    )
    return CorpusStats(
        total=len(all_examples), train=len(train), valid=len(valid),
        from_insights=len(reflections), from_facts=len(governed), out_dir=str(out),
        manifest_path=str(manifest_path), manifest_hash=manifest_hash,
        semantic_checkpoint_generation=generation,
        semantic_checkpoint_id=getattr(checkpoint, "latest_event_id", None),
        snapshot_hash=snapshot_hash, policy_digest=policy_digest,
        assertion_lineage=assertion_lineage,
    )
