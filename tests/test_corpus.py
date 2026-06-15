"""Tests for the reflection-derived corpus builder (Linux-friendly, no MLX)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from kestrel_feature_parametric_self import build_corpus


def _fixture_db(path: Path) -> str:
    """A minimal cognition DB with the reflection + learned_fact shapes."""
    db = str(path / "cognition.db")
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE reflection_insights (
            id TEXT PRIMARY KEY, agent_id TEXT, session_id TEXT, type TEXT,
            title TEXT NOT NULL, description TEXT, evidence TEXT,
            confidence REAL, actionable INTEGER, suggested_action TEXT,
            created_at TIMESTAMP
        );
        CREATE TABLE graph_nodes (
            node_id TEXT PRIMARY KEY, node_type TEXT NOT NULL, label TEXT NOT NULL,
            properties TEXT
        );
        """
    )
    con.executemany(
        "INSERT INTO reflection_insights (id, type, title, description, suggested_action) "
        "VALUES (?,?,?,?,?)",
        [
            ("1", "failure", "Excessive verbosity", "User asked for shorter replies.", "Be concise."),
            ("2", "success", "Closed an issue end to end", "Verified the PR and merged.", ""),
            ("3", "anomaly", "Self-musing", "A free-floating thought.", ""),  # not grounded
        ],
    )
    con.execute(
        "INSERT INTO graph_nodes (node_id, node_type, label, properties) VALUES (?,?,?,?)",
        (
            "n1",
            "learned_fact",
            "Pronouns: she/her",
            json.dumps({"subject": "Meridian", "predicate": "pronouns_when_choice_needed", "value": "she/her"}),
        ),
    )
    con.commit()
    con.close()
    return db


def test_grounded_only_excludes_self_musing(tmp_path):
    db = _fixture_db(tmp_path)
    stats = build_corpus(db, str(tmp_path / "out"), grounded_only=True)
    # 2 grounded insights (failure, success) + 1 fact; the 'anomaly' is dropped.
    assert stats.from_insights == 2
    assert stats.from_facts == 1
    assert stats.total == 3


def test_includes_all_insights_when_not_grounded_only(tmp_path):
    db = _fixture_db(tmp_path)
    stats = build_corpus(db, str(tmp_path / "out"), grounded_only=False)
    assert stats.from_insights == 3  # anomaly now included


def test_writes_valid_chat_jsonl(tmp_path):
    db = _fixture_db(tmp_path)
    out = tmp_path / "out"
    build_corpus(db, str(out), grounded_only=True, valid_every=0)
    lines = (out / "train.jsonl").read_text().strip().splitlines()
    assert lines
    for line in lines:
        ex = json.loads(line)
        roles = [m["role"] for m in ex["messages"]]
        assert roles == ["user", "assistant"]
        assert ex["messages"][1]["content"]  # non-empty answer
    # the learned fact is recoverable in the corpus
    assert any("she/her" in line for line in lines)
