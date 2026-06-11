"""Candidate loading and text construction shared across the pipeline.

embed.py, rank.py and the sandbox app must build *identical* candidate
documents, otherwise precomputed embeddings silently disagree with live ones.
Keeping the builder here is what guarantees that.
"""

from __future__ import annotations

import datetime as dt
import gzip
import io
import json
from pathlib import Path
from typing import Iterator

from . import config


def iter_candidates(path: str | Path) -> Iterator[dict]:
    """Stream candidate records from a .jsonl or .jsonl.gz file.

    Streaming keeps peak memory flat (~a few hundred MB) instead of holding
    the full 465 MB pool as Python objects.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_candidates_blob(raw: bytes) -> list[dict]:
    """Parse an uploaded blob that may be JSON array, JSONL, or gzipped JSONL.

    Used by the sandbox app, which accepts small samples in any of the
    bundle's formats.
    """
    if raw[:2] == b"\x1f\x8b":  # gzip magic
        raw = gzip.decompress(raw)
    text = raw.decode("utf-8").strip()
    if text.startswith("["):
        data = json.load(io.StringIO(text))
        if not isinstance(data, list):
            raise ValueError("Expected a JSON array of candidate objects.")
        return data
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value[:10])
    except ValueError:
        return None


def months_between(start: dt.date, end: dt.date) -> float:
    return (end.year - start.year) * 12 + (end.month - start.month) + (end.day - start.day) / 30.0


def career_span_years(candidate: dict) -> float:
    """Observable career span: earliest start date to reference date."""
    starts = [parse_date(j.get("start_date")) for j in candidate.get("career_history", [])]
    starts = [s for s in starts if s]
    if not starts:
        return 0.0
    return max(0.0, months_between(min(starts), config.REFERENCE_DATE) / 12.0)


def build_candidate_document(candidate: dict) -> str:
    """Flatten a candidate into the text that gets embedded.

    Deliberately favours *narrative* fields (summary, role descriptions) over
    the skills list: the skills list is exactly where keyword stuffers live,
    and the JD warns the dataset is seeded with them. Descriptions are where
    plain-language strong candidates describe what they actually built.
    """
    profile = candidate.get("profile", {})
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        f"Current role: {profile.get('current_title', '')} at "
        f"{profile.get('current_company', '')} ({profile.get('current_industry', '')}).",
    ]

    history = candidate.get("career_history", [])
    # Most recent first; the schema does not guarantee order, so sort.
    history = sorted(
        history,
        key=lambda j: parse_date(j.get("start_date")) or dt.date.min,
        reverse=True,
    )
    for job in history[: config.EMBED_CAREER_ENTRIES]:
        desc = (job.get("description") or "")[: config.EMBED_CAREER_CHARS]
        parts.append(f"{job.get('title', '')} at {job.get('company', '')}: {desc}")

    skills = candidate.get("skills", [])
    # Only carry skills with real usage time into the embedding -- this is a
    # cheap pre-filter against stuffed skill lists.
    used = [s["name"] for s in skills if s.get("duration_months", 0) >= 6]
    if used:
        parts.append("Skills used in practice: " + ", ".join(used[:20]))

    return "\n".join(p for p in parts if p)


def load_job_description(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")
