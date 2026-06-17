# redrob-ranker

Candidate ranking system for the Redrob India Runs **Data & AI Challenge**
(Track 1 — Intelligent Candidate Discovery). Ranks a 100,000-candidate pool
against the Senior AI Engineer job description and produces the top-100
submission CSV with per-candidate reasoning.

## Reproduce

```bash
pip install -r requirements.txt

# 1. One-time pre-computation (~45 min on a laptop CPU): embeds every
#    candidate and the JD, writes artifacts/.
python embed.py --candidates ./candidates.jsonl

# 2. Ranking step — the single command that produces the submission CSV.
#    Runs in well under 5 minutes, CPU only, no network, needs only numpy.
python rank.py --candidates ./candidates.jsonl --out ./team_xxx.csv
```

Both commands accept `.jsonl` or `.jsonl.gz`. Validate before uploading:

```bash
python scripts/validate_submission.py ./team_xxx.csv
```

The split matters for the compute constraints: pre-computation is allowed
outside the 5-minute budget (submission_spec.md §10.3), while `rank.py` —
the step reproduced at Stage 3 — performs no model inference and no network
calls. It streams the candidate file, joins each record to its precomputed
embedding, scores, and writes the CSV.

## Scoring model

```
final = (0.40 × semantic + 0.60 × structural)
        × behavioral_multiplier
        × integrity_multiplier
```

**Semantic** — cosine similarity between the JD embedding and a candidate
document built from *narrative* fields (summary, headline, recent role
descriptions), embedded with `BAAI/bge-small-en-v1.5`. Skills enter the document
only if used ≥ 6 months, so a stuffed skill list contributes nothing here.
Similarities are min-max normalised over the pool. This component is what
surfaces the "plain-language" strong candidates the JD warns about — people
who never write "RAG" but describe building a recommendation system.

**Structural (dominant, 0.60)** — the JD is rule-heavy, and rules an
embedding cannot see decide this dataset. Six weighted components
(title/domain fit 0.30, career evidence of shipped retrieval/ranking systems
0.30, experience band 0.15, trust-weighted JD-skill coverage 0.15,
education tier 0.05, logistics 0.05), then explicit penalty multipliers for
every disqualifier the JD names: consulting-only careers, research-only
careers, title-chasers, CV/speech/robotics-only specialists, leadership roles
with stale hands-on work, and keyword stuffers (an AI skill list attached to
a non-technical career is cut to 5%). Skill trust is weighted by endorsements
and `duration_months`, so self-reported "expert" claims with no usage time
score ~0.

**Behavioral (multiplier, 0.30–1.15)** — activity recency, recruiter response
rate, open-to-work, interview completion, verification. Multiplicative on
purpose: enthusiasm can't rescue a bad skills fit, but a ghost can't be
rescued by a perfect profile — exactly the asymmetry the JD's hackathon note
asks for ("a perfect-on-paper candidate who hasn't logged in for 6 months …
is not actually available").

**Integrity (multiplier)** — internal-consistency checks against honeypots:
claimed experience exceeding the observable career span, stated role
durations contradicting their own dates, roles ending before they start, and
3+ "expert" skills with zero months of use. Two or more hard flags ⇒ ×0.02
(effectively excluded); one ⇒ ×0.30. The checks were calibrated on the
bundle's sample: each hard check has a 0% false-positive rate there, while
signals that *look* impossible but occur in 4–26% of real profiles (salary
min > max, signup after last activity) are treated as generator noise — far
too common to mark ~80 honeypots in 100K — and demoted or ignored. The
calibration is documented inline in `ranker/honeypot.py`.

**No retrieval pre-filter (deliberate)** — a common pattern is to shortlist
~500 candidates by cosine similarity and score only those. We score all 100K:
structural scoring is cheap enough to fit the budget, and a semantic
pre-filter would drop exactly the plain-language candidates whose narrative
embedding is mediocre but whose career structure is excellent — the trap the
JD warns about. Two-stage retrieval is the right call when scoring is
expensive; here it is not.

**Determinism** — all recency math is anchored to a pinned
`REFERENCE_DATE` (2026-06-01), reasoning patterns are a pure function of
rank, and ties are broken by `candidate_id` ascending on the *rounded* score
the spec checks. Two runs produce byte-identical CSVs.

## Weight rationale

`W_SEMANTIC = 0.40` / `W_STRUCTURAL = 0.60` are principled defaults, not empirically tuned.
No labeled ground-truth ranking was available for calibration. Grid search over
(W_SEM, W_STR) would optimise against our own assumptions, not ground truth.
The JD is unusually rule-heavy — it names explicit disqualifiers, not relative
preferences — so structural rules dominate correctly. The 0.40 semantic weight
is enough to surface plain-language candidates whose narrative matches the JD
without the fashionable vocabulary (the "plain-language Tier 5" case the JD
explicitly warns about).

## Reasoning column

Generated at write time from computed profile facts only — no free text, so
nothing can be claimed that isn't in the record. Sentence patterns rotate
deterministically with rank (any 30 consecutive ranks get structurally
distinct text), concerns are stated whenever they exist (long notice periods,
short stints, inactivity), and tone follows rank: 1–10 strong, 11–40
positive-with-caveat, 41–75 balanced, 76–100 explicitly hedged.

## Repository layout

```
embed.py                  pre-computation: candidate + JD embeddings
rank.py                   ranking step (≤5 min, numpy only) → submission CSV
app.py                    Streamlit sandbox (≤100-candidate sample, live demo)
ranker/
  config.py               every tunable: weights, lexicons, pinned reference date
  loading.py              streaming JSONL I/O, shared candidate-document builder
  structural.py           JD rules: component scores + disqualifier penalties
  behavioral.py           availability multiplier
  honeypot.py             integrity checks (calibration notes inline)
  reasoning.py            Stage-4-oriented reasoning generation
  pipeline.py             scoring orchestration, top-K selection, CSV writer
tests/test_pipeline.py    15 tests: honeypots, traps, tie-breaks, CSV format
data/                     job description + bundle sample (sandbox input)
scripts/                  the official format validator
artifacts/                embedding artifacts written by embed.py
```

## Artifacts

`embed.py` writes three files to `artifacts/`:

| File | Size | In git? |
|---|---|---|
| `jd_embedding.npy` | 1.5 KB | yes |
| `candidate_ids.json` | ~1.3 MB | yes |
| `candidate_embeddings.npy` | ~290 MB (float32) | **no** — exceeds GitHub's 100 MB file limit |

The embedding matrix is regenerated deterministically by the documented
pre-computation command above (spec §10.3 allows "a script that produces
them" in place of committing artifacts).

## Sandbox

`app.py` is the hosted sandbox (spec §10.5): upload a ≤100-candidate sample
(`data/sample_candidates.json` works as-is), embeddings for the sample are
computed live on CPU in seconds, the identical scoring pipeline runs, and the
ranked CSV is downloadable — with a per-candidate score breakdown for
transparency. Run locally with `streamlit run app.py`.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

The suite targets the failure modes that actually disqualify submissions:
honeypots reaching the top 100, keyword stuffers outranking real engineers,
consulting-only/title-chaser detection, tie-break ordering, and CSV format
compliance.