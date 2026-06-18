#!/usr/bin/env python3
"""Pre-compute candidate and JD embeddings.

This is the documented PRE-COMPUTATION step (submission_spec.md section 10.3
allows pre-computation outside the 5-minute ranking budget). It runs once,
takes ~30-60 minutes on a laptop CPU for the full 100K pool, and writes three
artifacts that rank.py consumes:

    artifacts/candidate_embeddings.npy   float32 [N, 384], L2-normalised
    artifacts/candidate_ids.json         list[str], row-aligned with the matrix
    artifacts/jd_embedding.npy           float32 [384], L2-normalised

Usage:
    python embed.py --candidates ./candidates.jsonl
    python embed.py --candidates ./candidates.jsonl.gz --out-dir ./artifacts
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ranker import config
from ranker.loading import build_candidate_document, iter_candidates, load_job_description


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="candidates.jsonl or .jsonl.gz")
    parser.add_argument("--jd", default="data/job_description.txt")
    parser.add_argument("--out-dir", default="artifacts")
    args = parser.parse_args()

    # Imported here so `--help` works without the model stack installed.
    import torch
    from sentence_transformers import SentenceTransformer

    # Prevent PyTorch from oversubscribing threads on CPU (particularly on Windows)
    # which causes thread thrashing and massive context switching overhead.
    # Limiting PyTorch to 1 CPU thread keeps execution deterministic and highly performant.
    torch.set_num_threads(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {config.EMBEDDING_MODEL} (CPU)...")
    model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")

    jd_text = load_job_description(args.jd)
    bge_query_prefix = "Represent this sentence for searching relevant passages: "
    jd_embedding = model.encode(bge_query_prefix + jd_text, normalize_embeddings=True)
    np.save(out_dir / "jd_embedding.npy", jd_embedding.astype(np.float32))

    print("Model loaded successfully! Starting candidate processing...")

    ids: list[str] = []
    chunks: list[np.ndarray] = []
    batch: list[str] = []
    started = time.time()

    def flush() -> None:
        if batch:
            chunks.append(
                model.encode(
                    batch,
                    batch_size=config.EMBED_BATCH_SIZE,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).astype(np.float32)
            )
            batch.clear()

    for n, candidate in enumerate(iter_candidates(args.candidates), start=1):
        ids.append(candidate["candidate_id"])
        batch.append(build_candidate_document(candidate))
        if len(batch) >= config.EMBED_BATCH_SIZE:
            flush()
        if n % 100 == 0:
            elapsed = time.time() - started
            print(f"  {n:>6} candidates embedded  ({elapsed/60:.1f} min, {n/elapsed:.0f}/s)")
    flush()

    matrix = np.vstack(chunks) if chunks else np.zeros((0, config.EMBEDDING_DIM), np.float32)
    np.save(out_dir / "candidate_embeddings.npy", matrix.astype(np.float32))
    (out_dir / "candidate_ids.json").write_text(json.dumps(ids), encoding="utf-8")

    print(f"Done: {len(ids)} embeddings -> {out_dir}/ in {(time.time()-started)/60:.1f} min")


if __name__ == "__main__":
    main()
