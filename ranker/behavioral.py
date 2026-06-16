"""Behavioral availability multiplier.

The JD's hackathon note is explicit: "a perfect-on-paper candidate who hasn't
logged in for 6 months and has a 5% recruiter response rate is, for hiring
purposes, not actually available. Down-weight them appropriately."

This is implemented as a *multiplier* rather than an additive component on
purpose: strong behavioral signals should never compensate for a weak skills
fit (an enthusiastic accountant is still an accountant), but weak behavioral
signals should drag down even a perfect fit. Multiplication gives exactly
that asymmetry; addition does not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import config
from .loading import parse_date


@dataclass
class BehavioralResult:
    multiplier: float = 1.0
    notes: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    days_inactive: int | None = None
    response_rate: float | None = None


def behavioral_multiplier(candidate: dict) -> BehavioralResult:
    result = BehavioralResult()
    signals = candidate.get("redrob_signals", {}) or {}
    m = 1.0

    # -- Recency of activity ----------------------------------------------
    last_active = parse_date(signals.get("last_active_date"))
    if last_active:
        days = (config.REFERENCE_DATE - last_active).days
        result.days_inactive = days
        step = math.exp(-max(0, days) / 90.0)
        m *= step
        if days > 90:
            result.concerns.append(f"inactive on platform for ~{days} days")
        elif days <= 14:
            result.notes.append("active on the platform this fortnight")

    # -- Responsiveness -----------------------------------------------------
    rate = signals.get("recruiter_response_rate")
    if rate is None:
        rate = 0.5
    result.response_rate = rate
    m *= rate
    if rate < 0.2:
        result.concerns.append(f"{rate:.0%} recruiter response rate")
    elif rate >= 0.6:
        result.notes.append(f"{rate:.0%} recruiter response rate")

    # -- Stated availability ------------------------------------------------
    if signals.get("open_to_work_flag"):
        m *= 1.05
        result.notes.append("open to work")
    else:
        m *= 0.90

    # -- Recruiter-revealed preference signals --------------------------------
    saves = signals.get("saved_by_recruiters_30d") or 0
    if saves > 0:
        save_bonus = 1.0 + config.RECRUITER_SAVE_BONUS * min(saves, config.RECRUITER_SAVE_MAX)
        m *= save_bonus
        result.notes.append(f"saved by {saves} recruiter(s) in the last 30 days")

    views = signals.get("profile_views_received_30d") or 0
    if views >= config.PROFILE_VIEWS_THRESHOLD:
        m *= (1.0 + config.PROFILE_VIEWS_BONUS)

    apps = signals.get("applications_submitted_30d") or 0
    if apps > 0:
        m *= (1.0 + config.APP_SUBMITTED_BONUS)
        result.notes.append("actively applying to roles")

    # -- Phase 1.3: 6 Missing Behavioral Signals ------------------------------

    # github_activity_score: JD explicitly values open-source + code quality
    github = signals.get("github_activity_score")
    if github is None:
        github = -1

    if github == -1:
        github_mult = 1.0
    elif github >= 60:
        github_mult = 1.05  # active coder bonus
        result.notes.append(f"active public GitHub (score {github:.0f})")
    elif github < 20:
        github_mult = 0.95  # inactive coder soft penalty
    else:
        github_mult = 1.0

    # interview_completion_rate: strong reliability/availability signal
    icr = signals.get("interview_completion_rate")
    if icr is None:
        icr = 0.5

    if icr < 0.3:
        interview_mult = 0.70  # softer penalty
        result.concerns.append(f"completes only {icr:.0%} of scheduled interviews (ghosting risk)")
    elif icr > 0.8:
        interview_mult = 1.05
    else:
        interview_mult = 1.0

    # offer_acceptance_rate: -1 means no prior offers (new to market = neutral)
    oar = signals.get("offer_acceptance_rate")
    if oar is None:
        oar = -1

    if oar == -1:
        offer_mult = 1.0  # no history = neutral
    elif oar < 0.2:
        offer_mult = 0.92  # serial rejector = availability risk
        result.concerns.append(f"low historical offer acceptance rate ({oar:.0%})")
    else:
        offer_mult = 1.0

    # profile_completeness_score: incomplete profile = lower trust in all signals
    pcs = signals.get("profile_completeness_score")
    if pcs is None:
        pcs = 70

    if pcs < 50:
        completeness_mult = 0.95
        result.concerns.append(f"incomplete profile (score {pcs:.0f})")
    else:
        completeness_mult = 1.0

    # avg_response_time_hours: < 24h = eager, > 168h (1 week) = slow
    rth = signals.get("avg_response_time_hours")
    if rth is None:
        rth = 48

    if rth < 24:
        response_time_mult = 1.02
    elif rth > 168:
        response_time_mult = 0.95
        result.concerns.append(f"slow response time (~{rth:.0f} hours)")
    else:
        response_time_mult = 1.0

    # verified_email + verified_phone: reachability, both verified = tiny bonus
    verified = (signals.get("verified_email", False) and 
                signals.get("verified_phone", False))
    verified_mult = 1.01 if verified else 0.98

    # Compose into final behavioral multiplier
    extra = github_mult * interview_mult * offer_mult * completeness_mult * response_time_mult * verified_mult
    m *= extra

    result.multiplier = max(config.BEHAVIORAL_FLOOR, min(config.BEHAVIORAL_CEILING, m))
    return result
