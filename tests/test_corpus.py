"""Corpus builder tests: every example arrives in the host's governed snapshot.

Reflection-derived examples are covered against core in
``test_reflection_governance.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kestrel_feature_parametric_self import build_corpus
from kestrel_feature_parametric_self.corpus import GovernedCorpusRequiredError


def _snapshot(*, values=("she/her",), generation=7):
    examples = []
    for index, value in enumerate(values):
        assertion = SimpleNamespace(
            assertion_id=f"assertion:{index}", revision_id=f"revision:{index}",
            subject=SimpleNamespace(value="https://example.test/Meridian"),
            predicate=SimpleNamespace(value="https://example.test/pronouns_when_choice_needed"),
            object=SimpleNamespace(lexical_form=value),
        )
        examples.append(SimpleNamespace(
            assertion=assertion, content_hash=f"sha256:content-{index}",
            source_occurrences=(SimpleNamespace(source_occurrence_id=f"source:{index}"),),
            decision=SimpleNamespace(included=True, reason=SimpleNamespace(value="included")),
        ))
    return SimpleNamespace(
        verified=True, examples=tuple(examples), snapshot_hash="sha256:snapshot",
        policy=SimpleNamespace(digest="sha256:policy"),
        tenant_id="tenant:test",
        checkpoint=SimpleNamespace(tenant_id="tenant:test", generation=generation, latest_event_id="event:7"),
        capability_versions={"semantic_maintenance": "1"},
    )


def _build(tmp_path, **kwargs):
    return build_corpus(
        str(tmp_path / "corpus"), governed_snapshot=kwargs.pop("snapshot", _snapshot()),
        manifest_dir=str(tmp_path / "candidate"), **kwargs,
    )


def test_facts_arrive_from_host_snapshot(tmp_path):
    stats = _build(tmp_path)
    assert stats.from_insights == 0
    assert stats.from_facts == 1
    assert stats.assertion_lineage == (("assertion:0", "revision:0"),)
    rows = (tmp_path / "corpus" / "train.jsonl").read_text()
    assert "she/her" in rows
    assert "she/her" not in Path(stats.manifest_path).read_text()


def test_manifest_and_split_are_deterministic_across_shuffled_snapshot(tmp_path):
    first = _build(tmp_path / "one", snapshot=_snapshot(values=("a", "b", "c")), valid_every=2)
    second = _build(tmp_path / "two", snapshot=_snapshot(values=("c", "a", "b")), valid_every=2)
    left = json.loads(Path(first.manifest_path).read_text())
    right = json.loads(Path(second.manifest_path).read_text())
    assert first.manifest_hash == second.manifest_hash
    assert left["examples"] == right["examples"]


def test_manifest_is_immutable_and_missing_snapshot_is_refused(tmp_path):
    with pytest.raises(GovernedCorpusRequiredError, match="snapshot_required"):
        build_corpus(str(tmp_path / "out"), governed_snapshot=None, manifest_dir=str(tmp_path / "candidate"))
    stats = _build(tmp_path)
    with pytest.raises(GovernedCorpusRequiredError, match="already_exists"):
        _build(tmp_path)
    assert Path(stats.manifest_path).is_file()
