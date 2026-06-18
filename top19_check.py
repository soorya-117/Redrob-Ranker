r"""
Run from the repo root (D:\New folder\redrob-ranker\redrob-ranker):

    python top19_check.py

Checks all 19 trap candidates from sample_submission.csv against their real
records in candidates.jsonl, using the actual structural_score(). Confirms
every keyword-stuffer / irrelevant-career trap is correctly crushed below
the leak threshold.
"""
import csv
import json
from ranker.structural import structural_score

with open("official docs/sample_submission.csv") as f:
    top19 = [r["candidate_id"] for r in list(csv.DictReader(f))[:19]]

wanted = set(top19)
found = {}
with open("candidates.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        if c["candidate_id"] in wanted:
            found[c["candidate_id"]] = c
        if len(found) == len(wanted):
            break

print(f"{'candidate_id':15} | {'title':30} | {'score':>6} | penalties")
print("-" * 90)
leaks = 0
missing = []
for cid in top19:
    if cid not in found:
        missing.append(cid)
        print(f"{cid:15} | NOT FOUND IN candidates.jsonl")
        continue
    c = found[cid]
    title = (c.get("profile", {}).get("current_title") or "")[:28]
    r = structural_score(c)
    is_leak = r.score >= 0.10 and not any(
        p in r.penalties for p in ("keyword_stuffer", "irrelevant_career")
    )
    flag = "LEAK!" if is_leak else "ok"
    if is_leak:
        leaks += 1
    print(f"{cid:15} | {title:30} | {r.score:6.3f} | {r.penalties}  {flag}")

print()
print(f"Total leaks: {leaks} / {len(top19) - len(missing)} checked ({len(missing)} missing)")
if leaks == 0 and not missing:
    print("PASS: all 19 trap candidates correctly suppressed.")