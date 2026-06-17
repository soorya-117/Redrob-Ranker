# TODO — Manual Steps (everything the repo can't do for you)

Work through these in order. Estimated total: 1–2 days, mostly waiting on
embed.py and HuggingFace deployment friction. Deadline: **June 28, 2026**.
Delete this file before your final git push (or keep it — it's honest
process documentation; your call).

---

## 1. Local setup + verify (~15 min)

```bash
cd redrob-ranker
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements-dev.txt
python -m pytest tests/ -v      # expect: 26 passed
```

If torch install is slow/heavy, the CPU-only wheel is smaller:
`pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu`
(then `pip install -r requirements-dev.txt` for the rest).

## 2. Run the precompute on the full pool (~30–60 min, one time)

Put `candidates.jsonl` (or the .gz) in the repo root, then:

```bash
python embed.py --candidates ./candidates.jsonl
```

Note the printed total time — that number goes into
`submission_metadata.yaml` → `pre_computation_time_minutes`.

## 3. Run + TIME the ranking step (must be < 5 min)

```bash
python rank.py --candidates ./candidates.jsonl --out ./team_xxx.csv
```

- The script prints its own wall-clock time and warns if > 300s.
- It also prints "Integrity-flagged candidates in output" — this should be 0.
- If you're anywhere near 5 minutes, tell me and we'll optimize (it should
  finish in ~1–2 min; JSON parsing is the bottleneck, not scoring).

## 4. Validate format

```bash
python scripts/validate_submission.py ./team_xxx.csv
```

Must print "Submission is valid." Run this before EVERY upload.

## 5. Manually review your top 25 (~30 min — do not skip)

Open the CSV and read every row in the top 25. For each candidate ask:
"Would the JD author actually interview this person?" Look especially for:
- anyone whose title is non-technical (stuffer leak),
- anyone based abroad ranked very high,
- reasoning that claims something you can't find in their profile
  (open `candidates.jsonl` and spot-check 3–4 against the raw record).

If anything looks wrong, bring it back to me before submitting.

## 6. Rename the CSV to your registered participant ID

The filename must be your participant ID, e.g. `team_ab12cd.csv` — NOT
`submission.csv`. Check your registration email/portal for the exact ID.

## 7. Git history (Stage 4 checks this — flat dumps get eliminated)

Initialize and commit in honest, logical increments as you verify each part
works on your machine. Suggested sequence (adapt freely, write your own
messages):

```bash
git init
git add README.md requirements*.txt .gitignore
git commit -m "Project scaffold: requirements, gitignore"

git add ranker/config.py ranker/loading.py ranker/__init__.py
git commit -m "Config with pinned reference date + streaming candidate loader"

git add ranker/honeypot.py
git commit -m "Integrity checks; calibrate against sample (salary inversion is noise, not honeypot)"

git add ranker/structural.py
git commit -m "Structural scoring: JD disqualifiers incl. consulting-only and title-chaser"

git add ranker/behavioral.py
git commit -m "Multiplicative behavioral availability modifier"

git add ranker/reasoning.py ranker/pipeline.py
git commit -m "Rank-aware reasoning generation + top-K selection with id tiebreak"

git add embed.py rank.py tests/ data/ scripts/ artifacts/.gitkeep
git commit -m "Precompute + ranking entrypoints, test suite, bundle reference data"

git add app.py
git commit -m "Streamlit sandbox per spec 10.5"

git add submission_metadata.yaml
git commit -m "Submission metadata"
```

Then create a GitHub repo and push:

```bash
git remote add origin https://github.com/YOUR_USERNAME/redrob-ranker.git
git branch -M main
git push -u origin main
```

Notes:
- `candidates.jsonl` and `artifacts/candidate_embeddings.npy` are gitignored
  (the matrix is ~147 MB, over GitHub's 100 MB limit). That's fine — the spec
  accepts "a script that produces them" (embed.py), and the README documents it.
- As you make further changes (timing fixes, top-25 review tweaks), commit
  them individually. Real iteration in the history is exactly what Stage 4
  wants to see.

## 8. Deploy the sandbox to HuggingFace Spaces (~1–2 hrs incl. friction)

1. Create account at huggingface.co if needed → New Space.
2. Space name: `redrob-ranker` · License: anything · SDK: **Streamlit** ·
   Hardware: CPU basic (free).
3. Upload these files/folders to the Space (web upload or git):
   - `app.py`
   - `ranker/` (whole folder)
   - `data/job_description.txt`
   - `data/sample_candidates.json` (optional, handy for testing)
   - `requirements.txt`
4. The Space builds automatically (~5–10 min; torch is the slow part).
5. Test it: open the Space, upload `data/sample_candidates.json`, click
   Run ranking. You should get a ranked table + CSV download in well under
   a minute.
6. Copy the Space URL into `submission_metadata.yaml` → `sandbox_link`,
   commit, push.

If the build fails on torch size, replace the torch line in the Space's
requirements.txt with the CPU wheel:
`--extra-index-url https://download.pytorch.org/whl/cpu` on the first line,
then `torch==2.4.1`.

## 9. Fill the remaining yaml fields

In `submission_metadata.yaml`: email, phone, team name (your registered
one), GitHub URL, sandbox URL, measured precompute minutes, Python version,
and flip `reproduction_tested: true` AFTER step 3 succeeds. Commit.

On the AI declaration: leave it as written (Claude, declared). Declared use
is explicitly not penalized; what kills people at Stage 5 is a declaration
that contradicts the interview. You used AI heavily and did the engineering
judgment — say exactly that, and be ready to walk through every design
decision (the README's "Scoring model" section is your interview script).

## 10. Submit via the portal

Checklist before upload:
- [ ] CSV named with your participant ID, validator passes
- [ ] GitHub repo public (or you can grant organizer access at Stage 3)
- [ ] Sandbox link loads and runs end-to-end
- [ ] yaml in repo root matches what you type into the portal
- [ ] Pitch deck/PDF if the track page requires it at upload (you said
      you'll build this later — confirm whether it's needed at submission
      time or only for finalists BEFORE the deadline)

Submission strategy: you have 3 slots and only the LAST valid one counts.
Submit once everything above is done (insurance), keep at least one slot
free for a final fix. Do not burn slots on variations — there's no
leaderboard feedback to learn from.

## 11. Before any Stage 5 interview

Re-read `ranker/config.py` top to bottom and be able to answer:
- Why 0.40 semantic / 0.60 structural? (JD is rule-heavy; rules beat vibes here)
- Why multiplicative behavioral? (asymmetry — ghosts can't be rescued)
- Why BGE-Small? (CPU budget; 33M params, 384-dim, fast; quality dominated by structural rules anyway)
- Why is salary min>max NOT a honeypot flag? (26% base rate in the sample —
  measured, documented in honeypot.py)
- Why no top-500 semantic pre-filter before scoring? (scoring all 100K fits
  the budget; a pre-filter risks dropping plain-language strong candidates)
- Why is education weighted at only 0.05? (The JD never explicitly mentions education as a requirement;
  IIT/BITS/IISc adds a small bonus as a founding-team proxy, but it is intentionally a minor signal —
  0.05 weight means a tier-1 degree cannot rescue a weak career or override disqualifiers)
- Why is the embedding matrix not in git? (~290 MB float32 > GitHub's 100 MB limit;
  regenerated by embed.py, which the spec allows)
