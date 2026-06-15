"""Build an MLX-LoRA chat corpus from the agent's own sleep-cycle output.

Source: the agent's cognition DB, specifically the **plaintext** layers the
sleep cycle produces — ``reflection_insights`` (self-improvement reflections)
and ``learned_fact`` graph nodes. ``conversation_history`` is Fernet-encrypted
at rest and is intentionally NOT read here (decryption must happen inside the
agent trust boundary; the reflection layer is on-thesis anyway — see
docs/TWO_BRAIN_ARCHITECTURE.md §5.1, §6).

Anti-collapse (§5.1): the corpus is biased toward *externally grounded*
material — reflections of type ``failure``/``success`` (which come from real
user corrections and outcomes) and durable ``learned_fact`` nodes — rather than
free-floating self-talk.

Output: ``train.jsonl`` / ``valid.jsonl`` in mlx_lm chat format:
    {"messages": [{"role": "user", "content": Q}, {"role": "assistant", "content": A}]}

This module is pure Python (sqlite3 + json) so it is fully testable off
Apple-Silicon; it imports no MLX.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Reflection insight types that derive from external signal (user corrections,
# observed outcomes) rather than pure introspection. Used to bias the corpus.
EXTERNALLY_GROUNDED_TYPES = ("failure", "success", "improvement")


@dataclass
class CorpusStats:
    """Result of a corpus build."""

    total: int
    train: int
    valid: int
    from_insights: int
    from_facts: int
    out_dir: str


def _insight_examples(con: sqlite3.Connection, grounded_only: bool) -> List[Dict]:
    rows = con.execute(
        "SELECT type, title, description, suggested_action FROM reflection_insights "
        "WHERE title IS NOT NULL AND length(trim(title)) > 0"
    ).fetchall()
    examples: List[Dict] = []
    for itype, title, description, suggested_action in rows:
        if grounded_only and (itype not in EXTERNALLY_GROUNDED_TYPES):
            continue
        description = (description or "").strip()
        suggested_action = (suggested_action or "").strip()
        if not description and not suggested_action:
            continue
        answer = description
        if suggested_action:
            answer = f"{answer}\n\nWhat to do: {suggested_action}".strip()
        examples.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"From your reflections, what did you learn about: {title.strip()}?",
                    },
                    {"role": "assistant", "content": answer},
                ]
            }
        )
    return examples


def _fact_examples(con: sqlite3.Connection) -> List[Dict]:
    rows = con.execute(
        "SELECT label, properties FROM graph_nodes WHERE node_type = 'learned_fact'"
    ).fetchall()
    examples: List[Dict] = []
    for label, properties in rows:
        try:
            props = json.loads(properties or "{}")
        except (json.JSONDecodeError, TypeError):
            props = {}
        subject = (props.get("subject") or "").strip()
        predicate = (props.get("predicate") or "").replace("_", " ").strip()
        value = str(props.get("value") or label or "").strip()
        if not value:
            continue
        question = f"What do you know about {subject}'s {predicate}?".replace("  ", " ").strip()
        examples.append(
            {
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": value},
                ]
            }
        )
    return examples


def build_corpus(
    db_path: str,
    out_dir: str,
    *,
    grounded_only: bool = True,
    valid_every: int = 10,
) -> CorpusStats:
    """Build train/valid JSONL from the agent's cognition DB.

    Args:
        db_path: Path to the agent's SQLite cognition DB (read-only).
        out_dir: Directory to write ``train.jsonl`` / ``valid.jsonl``.
        grounded_only: If True (default), keep only externally-grounded insight
            types (anti-collapse). Facts are always included (durable + grounded).
        valid_every: Deterministic 1-in-N holdout for the validation split.

    Returns:
        CorpusStats with counts and the output directory.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        insight_examples = _insight_examples(con, grounded_only)
        fact_examples = _fact_examples(con)
    finally:
        con.close()

    examples = insight_examples + fact_examples
    examples = [e for e in examples if e["messages"][1]["content"]]

    # Deterministic split (no RNG — stable across runs).
    valid = examples[::valid_every] if valid_every > 0 else []
    valid_ids = set(id(e) for e in valid)
    train = [e for e in examples if id(e) not in valid_ids]

    for name, rows in (("train", train), ("valid", valid)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for example in rows:
                fh.write(json.dumps(example, ensure_ascii=False) + "\n")

    return CorpusStats(
        total=len(examples),
        train=len(train),
        valid=len(valid),
        from_insights=len(insight_examples),
        from_facts=len(fact_examples),
        out_dir=str(out),
    )
