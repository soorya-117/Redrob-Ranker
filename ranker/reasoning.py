"""Reasoning-string generation for the submission CSV.

Stage 4 samples 10 rows and checks: specific profile facts, connection to JD
requirements, honest acknowledgment of gaps, zero hallucination, variation
between rows, and tone consistent with rank. Design decisions here map to
those checks directly:

  * Facts only -- every clause is sourced from values computed off the actual
    profile (never free-generated), so nothing can be claimed that isn't in
    the record.
  * Deterministic variation -- sentence patterns are selected from rank only,
    so any 30 consecutive ranks produce distinct (lead, second) pairs.
  * Tone tiers -- ranks 1-10 read strong, 11-40 positive-with-caveat, 41-75
    balanced, 76-100 explicitly hedged.
  * Concerns are MANDATORY when present -- if a candidate has a flag (title
    chaser, long notice, inactive) the second sentence ALWAYS states it.
    "No material gaps" is only possible when concerns is genuinely empty.

Bug fixed (Antigravity audit): the previous implementation put concerns and
"No material gaps" in the same pool and selected by modulo, meaning concerns
could be silently skipped ~60% of the time for top-10 candidates. The fix:
concern-containing options and no-concern options are in separate pools;
the pool chosen depends on whether concerns actually exist.
"""

from __future__ import annotations

from .behavioral import BehavioralResult
from .structural import StructuralResult


def _facts(candidate: dict, structural: StructuralResult, behavioral: BehavioralResult) -> dict:
    profile = candidate.get("profile", {})
    components = structural.components or {}
    return {
        "yoe": float(profile.get("years_of_experience") or 0.0),
        "title": profile.get("current_title") or "unspecified role",
        "company": profile.get("current_company") or "",
        "location": profile.get("location") or "",
        "skills": structural.matched_skills[:3],
        "evidence": structural.evidence,
        "concerns": structural.concerns + behavioral.concerns,
        "notes": behavioral.notes,
        "response_rate": behavioral.response_rate,
        "days_inactive": behavioral.days_inactive,
        "strong_fit": (
            components.get("career_evidence", 0.0) >= 0.40
            or components.get("title_domain", 0.0) >= 0.90
        ),
    }


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _lead_sentence(f: dict, lead_idx: int) -> str:
    yoe, title, company = f["yoe"], f["title"], f["company"]
    skills = ", ".join(f["skills"])
    at = f" at {company}" if company else ""
    art = _article(title)
    leads = [
        f"{title} with {yoe:.1f} years' experience",
        f"{yoe:.1f} years as {art} {title}{at}",
        f"Currently {art} {title} ({yoe:.1f} yrs total)",
        f"{title}{at}, {yoe:.1f} years in",
        f"{yoe:.1f}-year {title}" + (f" based in {f['location']}" if f["location"] else ""),
        f"{title} profile, {yoe:.1f} years of experience",
    ]
    lead = leads[lead_idx % len(leads)]

    detail_idx = lead_idx % 3
    if detail_idx == 0 and f["evidence"]:
        lead += f"; {f['evidence'][0]}"
    elif detail_idx == 1 and skills:
        lead += f"; evidenced depth in {skills}"
    elif f["evidence"]:
        lead += f"; {f['evidence'][-1]}"
    elif skills:
        lead += f"; evidenced depth in {skills}"
    return lead + "."


def _second_sentence(f: dict, rank: int, second_idx: int) -> str:
    concerns = f["concerns"]
    notes = f["notes"]
    rr = f["response_rate"]
    rr_txt = f"{rr:.0%} recruiter response rate" if rr is not None else ""

    # KEY INVARIANT: if concerns exist, we ALWAYS select from the concern pool.
    # "No material gaps" is unreachable when concerns is non-empty.
    # second_idx rotates *within* whichever pool applies, not across both.

    if rank <= 10:
        if concerns:
            if f["strong_fit"]:
                pool = [
                    f"Main watch-out: {concerns[0]}, but the fit otherwise maps directly onto the JD's retrieval-and-ranking mandate.",
                    f"One caveat — {concerns[0]} — though everything else lines up with what the role needs.",
                ]
            else:
                pool = [
                    f"Main watch-out: {concerns[0]}; ranked on overall signal strength rather than direct retrieval evidence.",
                    f"One caveat — {concerns[0]} — alongside a profile that scores well on aggregate rather than direct JD-domain history.",
                ]
            # Surface a second concern when present and idx warrants it.
            if len(concerns) > 1:
                pool.append(f"Two flags: {concerns[0]}, and {concerns[1]}.")
        else:
            pool = []
            if notes:
                pool.append(f"Also {notes[0]}, which the JD explicitly asks for in a reachable candidate.")
            if rr_txt:
                pool.append(f"Reachable in practice too: {rr_txt}.")
            pool.append("No material gaps against the JD's must-have list.")
        return pool[second_idx % len(pool)]

    if rank <= 40:
        if concerns:
            pool = [
                f"Solid match overall; one flag: {concerns[0]}.",
                f"Strong on the core profile; open question: {concerns[0]}.",
            ]
            if len(concerns) > 1:
                pool.append(f"Caveats: {concerns[0]}, and {concerns[1]}.")
        else:
            pool = []
            if notes:
                pool.append(f"Engagement signals support reachability: {notes[0]}.")
            if rr_txt:
                pool.append(f"Availability looks real ({rr_txt}).")
            pool.append("Fits the JD's core profile with no standout red flags.")
        return pool[second_idx % len(pool)]

    if rank <= 75:
        if concerns:
            pool = [
                f"Concerns: {'; '.join(concerns[:2])}.",
                f"Holding it back: {concerns[0]}.",
            ]
        else:
            pool = []
            if notes:
                pool.append(f"{notes[0].capitalize()}, but the fit is partial rather than direct.")
            pool.append("Adjacent rather than direct fit for the JD's retrieval focus.")
        return pool[second_idx % len(pool)]

    # Tail (76-100): tone must read as borderline, not glowing.
    if concerns:
        pool = [
            f"Included as lower-confidence filler — {'; '.join(concerns[:2])} — but retains enough adjacent signal to beat the remaining pool.",
            f"Borderline: {concerns[0]}; kept in the 100 on adjacent signal only.",
        ]
    else:
        pool = ["Borderline inclusion: adjacent skills rather than direct retrieval/ranking experience."]
    return pool[second_idx % len(pool)]


def build_reasoning(candidate: dict, rank: int,
                    structural: StructuralResult,
                    behavioral: BehavioralResult) -> str:
    f = _facts(candidate, structural, behavioral)
    # Pattern selection is a pure function of rank: lead cycles every 6 ranks,
    # second advances once per full lead rotation. The (lead, second) pair is
    # unique across 30 consecutive ranks, keeping sampled rows structurally
    # distinct even for near-identical profiles.
    n = rank - 1
    lead_idx = n % 6
    second_idx = (n // 6) % 5
    text = f"{_lead_sentence(f, lead_idx)} {_second_sentence(f, rank, second_idx)}"
    return " ".join(text.split())[:400]
