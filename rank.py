#!/usr/bin/env python3
"""Produce the top-100 submission CSV from precomputed embeddings.

This is the RANKING step that must satisfy the Stage 3 compute constraints:
<= 5 minutes wall-clock, <= 16 GB RAM, CPU only, no network. It performs no
model inference and makes no external calls -- it streams candidates.jsonl,
joins each record to its precomputed embedding, applies the structural /
behavioral / integrity scoring, and writes the CSV.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./team_xxx.csv
    python rank.py --candidates ./candidates.jsonl.gz \
                   --artifacts ./artifacts --out ./team_xxx.csv --top-k 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ranker.loading import iter_candidates
from ranker.pipeline import select_top, write_submission


def load_semantic_lookup(artifacts: Path):
    """Build candidate_id -> normalised semantic similarity in [0, 1].

    Similarities are computed once as a single matrix-vector product (both
    sides are L2-normalised, so the dot product is cosine similarity) and
    min-max normalised over the pool so the semantic component shares the
    [0, 1] scale of the structural component.
    """
    embeddings = np.load(artifacts / "candidate_embeddings.npy")
    jd = np.load(artifacts / "jd_embedding.npy")
    ids = json.loads((artifacts / "candidate_ids.json").read_text(encoding="utf-8"))
    if len(ids) != embeddings.shape[0]:
        raise SystemExit(
            f"Artifact mismatch: {len(ids)} ids vs {embeddings.shape[0]} embedding rows. "
            "Re-run embed.py."
        )

    sims = embeddings @ jd
    lo, hi = float(sims.min()), float(sims.max())
    span = (hi - lo) or 1.0
    norm = (sims - lo) / span
    index = {cid: float(norm[i]) for i, cid in enumerate(ids)}
    return index.get


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="candidates.jsonl or .jsonl.gz")
    parser.add_argument("--artifacts", default="artifacts", help="directory written by embed.py")
    parser.add_argument("--out", required=True, help="output CSV path (name it <participant_id>.csv)")
    parser.add_argument("--top-k", type=int, default=100)
    args = parser.parse_args()

    started = time.time()
    artifacts = Path(args.artifacts)
    for required in ("candidate_embeddings.npy", "jd_embedding.npy", "candidate_ids.json"):
        if not (artifacts / required).exists():
            raise SystemExit(
                f"Missing artifact {artifacts / required}. "
                "Run the precompute step first: python embed.py --candidates <file>"
            )

    semantic_lookup = load_semantic_lookup(artifacts)
    print(f"Embedding artifacts loaded in {time.time()-started:.1f}s")

    ranked = select_top(iter_candidates(args.candidates), semantic_lookup, top_k=args.top_k)
    if len(ranked) < args.top_k:
        print(f"[warn] only {len(ranked)} candidates available for top-{args.top_k}")

    write_submission(ranked, args.out)

    elapsed = time.time() - started
    flagged = sum(1 for sc in ranked if sc.integrity.is_suspect)
    print(f"Wrote {len(ranked)} rows -> {args.out}")
    print(f"Integrity-flagged candidates in output: {flagged} (should be 0)")
    print(f"Total ranking time: {elapsed:.1f}s (budget: 300s)")
    if elapsed > 300:
        print("[warn] over the 5-minute Stage 3 budget — investigate before submitting")


if __name__ == "__main__":
    main()
