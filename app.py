"""Sandbox app for the Redrob India Runs Data & AI Challenge (Track 1).

Satisfies submission_spec.md section 10.5: a hosted environment where the
ranking system runs end-to-end on a small candidate sample (<= 100) and
produces a ranked CSV. Embeddings for the sample are computed live (a 100-
candidate sample embeds in seconds on CPU); the full 100K run uses the
precomputed artifacts exactly as documented in the README.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ranker import config
from ranker.loading import build_candidate_document, load_candidates_blob, load_job_description
from ranker.pipeline import select_top, write_submission

MAX_SAMPLE = 100
JD_PATH = Path("data/job_description.txt")

st.set_page_config(page_title="redrob-ranker sandbox", page_icon="🎯", layout="wide")
st.title("redrob-ranker — sandbox")
st.caption(
    "Intelligent Candidate Discovery (India Runs, Track 1). Upload a candidate "
    f"sample (≤{MAX_SAMPLE}, JSON / JSONL / JSONL.GZ) and get the ranked CSV. "
    "Scoring is identical to the full 100K pipeline; sample embeddings are "
    "computed live on CPU."
)


@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")


@st.cache_data
def get_jd_text() -> str:
    return load_job_description(JD_PATH)


uploaded = st.file_uploader(
    "Candidate sample", type=["json", "jsonl", "gz"],
    help="Use sample_candidates.json from the hackathon bundle, or any slice of candidates.jsonl",
)

top_k = st.slider("Rows to rank", min_value=5, max_value=MAX_SAMPLE, value=25)

if uploaded is not None:
    try:
        candidates = load_candidates_blob(uploaded.getvalue())
    except (ValueError, json.JSONDecodeError) as exc:
        st.error(f"Could not parse the file: {exc}")
        st.stop()

    if len(candidates) > MAX_SAMPLE:
        st.warning(f"{len(candidates)} candidates uploaded; sandbox caps at {MAX_SAMPLE}. Truncating.")
        candidates = candidates[:MAX_SAMPLE]
    st.write(f"Parsed **{len(candidates)}** candidates.")

    if st.button("Run ranking", type="primary"):
        model = get_model()
        with st.spinner("Embedding sample + JD..."):
            bge_query_prefix = "Represent this sentence for searching relevant passages: "
            jd_vec = model.encode(bge_query_prefix + get_jd_text(), normalize_embeddings=True)
            docs = [build_candidate_document(c) for c in candidates]
            vecs = model.encode(docs, normalize_embeddings=True, show_progress_bar=False)
            sims = np.asarray(vecs) @ np.asarray(jd_vec)
            lo, hi = float(sims.min()), float(sims.max())
            span = (hi - lo) or 1.0
            lookup = {
                c["candidate_id"]: float((s - lo) / span)
                for c, s in zip(candidates, sims)
            }

        with st.spinner("Scoring..."):
            ranked = select_top(iter(candidates), lookup.get, top_k=min(top_k, len(candidates)))
            # write_submission expects a path; reuse its CSV logic via a temp file.
            out_path = Path("sandbox_output.csv")
            write_submission(ranked, out_path)
            csv_text = out_path.read_text(encoding="utf-8")

        df = pd.read_csv(io.StringIO(csv_text))
        st.subheader("Ranked output")
        st.dataframe(df, use_container_width=True, hide_index=True)

        flagged = sum(1 for sc in ranked if sc.integrity.is_suspect)
        col1, col2, col3 = st.columns(3)
        col1.metric("Candidates scored", len(candidates))
        col2.metric("Rows ranked", len(ranked))
        col3.metric("Integrity-flagged in output", flagged)

        st.download_button(
            "Download CSV", csv_text, file_name="sandbox_ranked.csv", mime="text/csv"
        )

        with st.expander("Why these ranks? (per-candidate breakdown)"):
            for sc in ranked[:10]:
                st.markdown(
                    f"**{sc.candidate_id}** — final `{sc.final:.4f}` "
                    f"(semantic `{sc.semantic:.2f}`, structural `{sc.structural.score:.2f}`, "
                    f"behavioral ×`{sc.behavioral.multiplier:.2f}`, "
                    f"integrity ×`{sc.integrity.multiplier:.2f}`)"
                )
                if sc.structural.penalties:
                    st.markdown(f"&nbsp;&nbsp;penalties: `{', '.join(sc.structural.penalties)}`")
else:
    st.info("Upload a candidate file to begin. `sample_candidates.json` from the bundle works as-is.")
