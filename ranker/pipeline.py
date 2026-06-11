"""Pipeline orchestration: combine semantic, structural, behavioral and
integrity signals into a final score, select the top-K, and emit the
submission CSV. Shared by rank.py (full 100K run) and app.py (sandbox).
"""

from __future__ import annotations

import csv
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from . import config
from .behavioral import BehavioralResult, behavioral_multiplier
from .honeypot import IntegrityReport, check_integrity
from .reasoning import build_reasoning
from .structural import StructuralResult, structural_score


@dataclass
class ScoredCandidate:
    candidate_id: str
    final: float
    semantic: float
    structural: StructuralResult
    behavioral: BehavioralResult
    integrity: IntegrityReport
    record: dict

    def sort_key(self) -> tuple:
        # Spec section 3: ties broken by candidate_id ascending.
        return (-self.final, self.candidate_id)


def score_candidate(record: dict, semantic_norm: float) -> ScoredCandidate:
    structural = structural_score(record)
    behavioral = behavioral_multiplier(record)
    integrity = check_integrity(record)

    base = config.W_SEMANTIC * semantic_norm + config.W_STRUCTURAL * structural.score
    final = base * behavioral.multiplier * integrity.multiplier

    return ScoredCandidate(
        candidate_id=record.get("candidate_id", ""),
        final=final,
        semantic=semantic_norm,
        structural=structural,
        behavioral=behavioral,
        integrity=integrity,
        record=record,
    )


def select_top(candidates: Iterator[dict],
               semantic_lookup,
               top_k: int = 100,
               pool_factor: int = 3) -> list[ScoredCandidate]:
    """Stream candidates, keeping only a small heap of the best.

    `semantic_lookup(candidate_id) -> float | None` returns the [0, 1]
    normalised semantic similarity. Records missing from the embedding
    artifacts are skipped with a warning rather than crashing a 5-minute
    reproduction run.

    A heap of top_k * pool_factor full records is retained (rather than
    exactly top_k) so the final exact sort with the candidate_id tiebreak has
    slack around equal scores at the cutoff.
    """
    heap: list[tuple[float, str, ScoredCandidate]] = []
    limit = top_k * pool_factor
    skipped = 0

    for record in candidates:
        cid = record.get("candidate_id", "")
        sem = semantic_lookup(cid)
        if sem is None:
            skipped += 1
            continue
        scored = score_candidate(record, sem)
        # heapq is a min-heap; invert candidate_id ordering so that, on equal
        # scores, the lexicographically *larger* id is evicted first.
        entry = (scored.final, _InvertedStr(cid), scored)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)

    if skipped:
        print(f"[warn] {skipped} candidates had no precomputed embedding and were skipped")

    finalists = sorted((e[2] for e in heap), key=ScoredCandidate.sort_key)
    return finalists[:top_k]


class _InvertedStr(str):
    """String with reversed comparison, for min-heap tiebreak ordering."""
    def __lt__(self, other):  # type: ignore[override]
        return str.__gt__(self, other)
    def __gt__(self, other):  # type: ignore[override]
        return str.__lt__(self, other)


def write_submission(ranked: list[ScoredCandidate], out_path: str | Path) -> None:
    """Write the spec-compliant CSV: header, exactly len(ranked) rows,
    non-increasing scores, UTF-8.

    The reported score is the 4-decimal rounding of the model score. Rounding
    can collapse two near-equal scores into a tie, and the validator requires
    ties to be ordered by candidate_id ascending -- so the final sort happens
    on the *rounded* score, which is the number the spec actually checks.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    decorated = sorted(
        ((min(round(sc.final, 4), 1.0), sc) for sc in ranked),
        key=lambda pair: (-pair[0], pair[1].candidate_id),
    )

    rows = []
    for i, (score, sc) in enumerate(decorated, start=1):
        rows.append({
            "candidate_id": sc.candidate_id,
            "rank": i,
            "score": f"{score:.4f}",
            "reasoning": build_reasoning(sc.record, i, sc.structural, sc.behavioral),
        })

    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(rows)
