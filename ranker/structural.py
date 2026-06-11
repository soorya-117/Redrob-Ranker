"""Structural scoring against the explicit rules of the job description.

The JD is unusually explicit: it names the profile it wants, the profiles it
will reject, and the traps planted in the dataset. This module encodes those
rules as deterministic, inspectable features. Every penalty corresponds to a
named section of data/job_description.txt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import config
from .loading import parse_date


def _contains_any(text: str, terms: tuple[str, ...] | frozenset | set) -> bool:
    return any(t in text for t in terms)


def _count_hits(text: str, terms: tuple[str, ...]) -> int:
    return sum(1 for t in terms if t in text)


@dataclass
class StructuralResult:
    score: float = 0.0
    components: dict = field(default_factory=dict)
    penalties: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    matched_skills: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Component scores
# ---------------------------------------------------------------------------

# Rotate evidence phrasing for ML-title matches so no single phrase dominates
# across 100 rows. Rotation index is the hash of candidate_id length modulo
# the pool size — deterministic per-profile without needing rank at this point.
_TITLE_EVIDENCE_TEMPLATES = (
    "current title '{title}' is squarely in the JD's domain",
    "role as '{title}' maps directly onto the JD's search-and-ranking mandate",
    "'{title}' title aligns with the retrieval/ML focus the JD requires",
    "current position as '{title}' fits the target ML/AI profile",
    "title '{title}' places this candidate in the JD's core domain",
)


def _title_domain_score(profile: dict, result: StructuralResult) -> float:
    title = (profile.get("current_title") or "")
    title_lower = title.lower()
    headline = (profile.get("headline") or "").lower()
    combined = f"{title_lower} {headline}"

    if _contains_any(combined, config.ML_TITLE_TERMS):
        # Rotate phrasing deterministically by hashing the title string itself
        # so the same profile always gets the same evidence phrase.
        idx = hash(title) % len(_TITLE_EVIDENCE_TEMPLATES)
        result.evidence.append(_TITLE_EVIDENCE_TEMPLATES[idx].format(title=title))
        return 1.0
    if _contains_any(combined, config.ADJACENT_TITLE_TERMS):
        return 0.55
    if _contains_any(combined, config.ENGINEERING_TITLE_TERMS) and not _contains_any(
        title_lower, config.NON_TECH_TITLE_TERMS
    ):
        return 0.35
    return 0.05


def _career_evidence_score(candidate: dict, result: StructuralResult) -> float:
    """Evidence of having built retrieval/ranking/ML systems in production.

    This is the component that rescues "plain-language Tier 5" candidates:
    someone whose skills section never says RAG but whose role description
    says they built a recommendation system at a product company.

    Tier-1 company prestige bonus: candidates who have worked at top product
    companies (config.TIER_1_COMPANIES) receive a bonus proportional to
    career evidence to signal operational scale experience. The bonus is only
    applied when there is actual ML/retrieval evidence so it amplifies signal
    rather than inflating irrelevant profiles.
    """
    history = candidate.get("career_history", []) or []
    profile = candidate.get("profile", {})
    narrative = " ".join(
        [(profile.get("summary") or ""), (profile.get("headline") or "")]
        + [(j.get("description") or "") for j in history]
    ).lower()

    retrieval_hits = _count_hits(narrative, config.RETRIEVAL_EVIDENCE_TERMS)
    production_hits = _count_hits(narrative, config.PRODUCTION_EVIDENCE_TERMS)
    ml_hits = _count_hits(narrative, config.ML_EVIDENCE_TERMS)

    # Product-company exposure: any role outside services/consulting.
    # Use substring containment (same fix as consulting_only penalty) so that
    # "IT Services & Consulting" correctly counts as a services role here too.
    product_roles = [
        j for j in history
        if not any(ind in (j.get("industry") or "").lower() for ind in config.CONSULTING_INDUSTRIES)
    ]

    # Tier-1 company check: has the candidate worked at a high-signal product
    # company at any point in their career? Matched as a lowercase substring.
    tier1_roles = [
        j for j in history
        if any(t1 in (j.get("company") or "").lower() for t1 in config.TIER_1_COMPANIES)
    ]

    score = 0.0
    if retrieval_hits:
        score += min(0.55, 0.18 * retrieval_hits)
        result.evidence.append("career history describes retrieval/search/ranking work")
    if ml_hits:
        score += min(0.20, 0.07 * ml_hits)
    if production_hits:
        score += min(0.15, 0.05 * production_hits)
    # Product-company bonus only applies when there's actual ML/retrieval
    # evidence — it amplifies signal, not replaces it.
    if product_roles and (retrieval_hits or ml_hits):
        score += 0.10
    # Tier-1 prestige bonus: awarded only when ML/retrieval evidence exists.
    if tier1_roles and (retrieval_hits or ml_hits):
        score += config.TIER_1_COMPANY_BONUS
        company_names = ", ".join(dict.fromkeys(
            j.get("company", "") for j in tier1_roles
        ))[:60]
        result.evidence.append(f"Tier-1 product company experience ({company_names})")
    score = min(1.0, score)

    if retrieval_hits and production_hits:
        result.evidence.append("describes shipping to production, not just prototyping")
    return score


def _experience_band_score(profile: dict, result: StructuralResult) -> float:
    yoe = float(profile.get("years_of_experience") or 0.0)
    if config.EXP_IDEAL_LO <= yoe <= config.EXP_IDEAL_HI:
        return 1.0
    if yoe < config.EXP_HARD_FLOOR:
        result.concerns.append(f"only {yoe:.1f} years of experience for a senior founding-team role")
        return 0.05
    if yoe < config.EXP_IDEAL_LO:  # 3-5y: JD will consider with strong signals
        result.concerns.append(f"{yoe:.1f} years is below the JD's 5-9 band")
        return 0.55
    if yoe <= config.EXP_SOFT_HI:  # 9-11y
        return 0.80
    result.concerns.append(f"{yoe:.1f} years may read as over-band for a hands-on IC role")
    return 0.55


def _skills_trust_score(candidate: dict, result: StructuralResult) -> float:
    """JD-skill coverage, trust-weighted.

    A skill only counts in proportion to evidence it was actually used:
    proficiency alone is self-reported and free to inflate; endorsements and
    duration_months are harder to fake. This is the anti-keyword-stuffing
    weighting -- a stuffed list of zero-duration 'expert' skills scores ~0.
    """
    prof_weight = {"beginner": 0.25, "intermediate": 0.55, "advanced": 0.85, "expert": 1.0}
    assessments = (candidate.get("redrob_signals") or {}).get("skill_assessment_scores") or {}

    total = 0.0
    for skill in candidate.get("skills", []) or []:
        name = (skill.get("name") or "").lower()
        if not any(jd in name or (len(name) >= 3 and name in jd) for jd in config.JD_SKILLS):
            continue
        duration = skill.get("duration_months", 0) or 0
        endorsements = skill.get("endorsements", 0) or 0
        trust = min(1.0, duration / 24.0) * (0.5 + 0.5 * min(1.0, math.log1p(endorsements) / math.log1p(30)))
        weight = prof_weight.get(skill.get("proficiency"), 0.4)
        # Platform assessment, when present, is independent verification.
        assessed = assessments.get(skill.get("name"))
        if assessed is not None:
            trust *= 0.6 + 0.4 * (assessed / 100.0)
        contribution = weight * trust
        if contribution > 0.15:
            result.matched_skills.append(skill.get("name"))
        total += contribution

    return min(1.0, total / 4.0)  # ~4 well-evidenced JD skills saturates


def _logistics_score(candidate: dict, result: StructuralResult) -> float:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {}) or {}
    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    relocate = bool(signals.get("willing_to_relocate"))

    if country and country != "india":
        loc = config.LOCATION_ABROAD
        result.concerns.append(f"based in {profile.get('location')}, {profile.get('country')} — JD doesn't sponsor visas")
    elif _contains_any(location, config.LOCATION_PREFERRED):
        loc = 1.0
        result.evidence.append(f"based in {profile.get('location')} (JD's preferred location)")
    elif _contains_any(location, config.LOCATION_WELCOME):
        loc = 0.90
    elif relocate:
        loc = config.LOCATION_INDIA_RELOCATE
    else:
        loc = config.LOCATION_INDIA_NO_RELOCATE
        result.concerns.append("outside the JD's listed cities and not flagged willing to relocate")

    raw_notice = signals.get("notice_period_days")
    notice = 90 if raw_notice is None else int(raw_notice)
    notice_score = config.NOTICE_LONG
    for max_days, value in config.NOTICE_STEPS:
        if notice <= max_days:
            notice_score = value
            break
    if notice > 60:
        result.concerns.append(f"{notice}-day notice period (JD prefers sub-30)")

    mode = signals.get("preferred_work_mode", "flexible")
    mode_score = 1.0 if mode in ("hybrid", "flexible", "onsite") else 0.75  # remote-only vs hybrid JD

    return 0.55 * loc + 0.30 * notice_score + 0.15 * mode_score


# ---------------------------------------------------------------------------
# JD disqualifier penalties
# ---------------------------------------------------------------------------

def _apply_penalties(candidate: dict, base: float, result: StructuralResult) -> float:
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", []) or []
    title = (profile.get("current_title") or "").lower()
    narrative = " ".join(
        [(profile.get("summary") or "")] + [(j.get("description") or "") for j in history]
    ).lower()

    # Keyword stuffer: JD-perfect skill list attached to a non-technical
    # career. "A candidate who has all the AI keywords listed as skills but
    # whose title is 'Marketing Manager' is not a fit."
    jd_skill_count = sum(
        1 for s in candidate.get("skills", []) or []
        if any(jd in (s.get("name") or "").lower() for jd in config.JD_SKILLS)
    )
    non_tech_title = _contains_any(title, config.NON_TECH_TITLE_TERMS)
    no_real_evidence = _count_hits(narrative, config.RETRIEVAL_EVIDENCE_TERMS) == 0 and \
        _count_hits(narrative, config.ML_EVIDENCE_TERMS) == 0
    if non_tech_title and jd_skill_count >= config.KEYWORD_STUFFER_MIN_JD_SKILLS and no_real_evidence:
        result.penalties.append("keyword_stuffer")
        result.concerns.append(f"AI skill list doesn't match a '{profile.get('current_title')}' career")
        return base * config.PENALTY_KEYWORD_STUFFER

    # Consulting-only career ("entire career" at services firms; prior
    # product experience redeems).
    # Bug fix (Antigravity audit): exact equality check missed "IT Services &
    # Consulting" and "IT Consulting". Use substring containment instead.
    if history:
        services = [
            j for j in history
            if any(ind in (j.get("industry") or "").lower() for ind in config.CONSULTING_INDUSTRIES)
            or _contains_any((j.get("company") or "").lower(), config.CONSULTING_FIRMS)
        ]
        if len(services) == len(history):
            result.penalties.append("consulting_only")
            result.concerns.append("entire career at IT-services/consulting firms (explicit JD disqualifier)")
            base *= config.PENALTY_CONSULTING_ONLY

    # Research-only career, no production deployment.
    if history and all(
        _contains_any((j.get("title") or "").lower(), config.RESEARCH_TITLE_TERMS)
        or (j.get("industry") or "").lower() in config.RESEARCH_INDUSTRIES
        for j in history
    ) and _count_hits(narrative, config.PRODUCTION_EVIDENCE_TERMS) == 0:
        result.penalties.append("research_only")
        result.concerns.append("pure research background with no production deployment signal")
        base *= config.PENALTY_RESEARCH_ONLY

    # Title-chaser: many short stints.
    yoe = float(profile.get("years_of_experience") or 0.0)
    if len(history) >= config.TITLE_CHASER_MIN_ROLES and yoe >= 4:
        tenures = [j.get("duration_months", 0) or 0 for j in history]
        if tenures and sum(tenures) / len(tenures) < config.TITLE_CHASER_MAX_AVG_TENURE_MONTHS:
            result.penalties.append("title_chaser")
            result.concerns.append("frequent short stints — JD wants a 3+ year commitment")
            base *= config.PENALTY_TITLE_CHASER

    # CV/speech/robotics specialist without NLP/IR exposure.
    # Bug fix (Antigravity audit): the original used plain substring matching,
    # so "context" matched "text" and "search" matched "research", falsely
    # injecting NLP hits. Fix: use \b word-boundary regex AND a separate
    # NLP_POSITIVE_TERMS list that excludes short ambiguous tokens ("text",
    # "nlp") that can appear in negating phrases like "no NLP background".
    import re as _re
    skills_text = " ".join((s.get("name") or "").lower() for s in candidate.get("skills", []) or [])
    full_text = f"{narrative} {skills_text}"
    cv_hits = _count_hits(full_text, config.CV_SPEECH_ROBOTICS_TERMS)
    # Only count multi-word or unambiguous NLP/IR terms to avoid false positives.
    nlp_positive_terms = (
        "natural language processing", "natural language", "information retrieval",
        "language model", "large language", "embedding", "embeddings",
        "retrieval", "semantic search", "vector search", "text classification",
        "sentiment analysis", "named entity", "machine translation",
        "question answering", "reading comprehension",
        "ranking", "recommendation", "recommender",
        "word2vec", "bert", "transformer", "lstm",
    )
    nlp_hits = sum(
        1 for t in nlp_positive_terms
        if _re.search(r'\b' + _re.escape(t) + r'\b', full_text)
    )
    if cv_hits >= 3 and nlp_hits == 0:
        result.penalties.append("cv_only")
        result.concerns.append("primary expertise in CV/speech/robotics with no NLP/IR exposure")
        base *= config.PENALTY_CV_ONLY

    # LangChain-only disqualifier: "if your AI experience consists primarily
    # of recent (<12 months) projects using LangChain to call OpenAI — we
    # will probably not move forward, unless you can demonstrate substantial
    # pre-LLM-era ML production experience."
    # Detection: wrapper-vocabulary dominates the skills + narrative AND there
    # is no evidence of pre-LLM ML work. We use only unambiguous, specific
    # pre-LLM terms so that negating phrases like "no retrieval systems" don't
    # accidentally register as positive pre-LLM evidence.
    langchain_terms = ("langchain", "openai api", "gpt-", "chatgpt", "llm api",
                       "llamaindex", "llama index", "llama_index", "openai.chat")
    # Specific enough that they can't appear by accident in a negation phrase.
    pre_llm_terms = (
        "word2vec", "fasttext", "glove embedding",
        "bert", "lstm", "seq2seq", "attention mechanism",
        "xgboost", "scikit-learn", "sklearn", "tensorflow", "pytorch", "keras",
        "a/b test", "ndcg", "mrr", "bm25",
        "elasticsearch", "faiss", "pinecone", "weaviate", "qdrant", "milvus",
        "feature engineering", "gradient boosting", "random forest",
        "collaborative filtering", "matrix factorization",
        "sentence-transformer", "sentence transformer",
    )
    lc_in_narrative = _contains_any(narrative, langchain_terms)
    pre_llm_hits = _count_hits(narrative, pre_llm_terms)
    lc_skill_months = sum(
        s.get("duration_months", 0) or 0
        for s in candidate.get("skills", []) or []
        if any(t in (s.get("name") or "").lower() for t in ("langchain", "openai", "gpt", "chatgpt"))
    )
    non_lc_ml_months = sum(
        s.get("duration_months", 0) or 0
        for s in candidate.get("skills", []) or []
        if (any(jd in (s.get("name") or "").lower() for jd in config.JD_SKILLS)
            and not any(t in (s.get("name") or "").lower()
                        for t in ("langchain", "openai", "gpt", "chatgpt")))
    )
    if (lc_in_narrative
            and lc_skill_months > 0
            and non_lc_ml_months < 12
            and pre_llm_hits == 0):
        result.penalties.append("langchain_only")
        result.concerns.append(
            "ML experience appears limited to recent LLM-wrapper work with no pre-LLM production history"
        )
        base *= 0.35

    # Outside India: the JD takes these "case-by-case", doesn't sponsor
    # visas, and its ideal profile is "located in or willing to relocate to
    # Noida or Pune". Logistics alone (10% weight) under-prices that, so
    # abroad is also a penalty -- softened when relocation is flagged.
    country = (profile.get("country") or "").lower()
    if country and country != "india":
        relocate = bool((candidate.get("redrob_signals") or {}).get("willing_to_relocate"))
        result.penalties.append("abroad")
        base *= 0.80 if relocate else 0.55

    # Stale hands-on: 18+ months in a pure leadership/architecture title with
    # no hands-on verbs in the current role description.
    current = next((j for j in history if j.get("is_current")), None)
    if current and _contains_any((current.get("title") or "").lower(), config.LEADERSHIP_TITLE_TERMS):
        months = current.get("duration_months", 0) or 0
        desc = (current.get("description") or "").lower()
        if months >= config.STALE_HANDS_ON_MONTHS and not _contains_any(desc, config.HANDS_ON_VERBS):
            result.penalties.append("stale_hands_on")
            result.concerns.append("18+ months in a non-coding leadership role — JD says 'this role writes code'")
            base *= config.PENALTY_STALE_HANDS_ON

    # Research-title without production proof: a candidate with 'research' in
    # their current or most-recent title at a product company may genuinely
    # ship ("AI Research Engineer" at Razorpay, Yellow.ai, etc.), but many
    # don't. Apply a proportional penalty unless their role descriptions
    # contain sufficient production-deployment vocabulary. The JD explicitly
    # states: "research without production is a disqualifier."
    # This fires independently of the hard research_only check above — it
    # targets the softer case of one research-flavoured title in an otherwise
    # product-company career that the hard check misses.
    RESEARCH_TITLE_TERMS = ("research engineer", "research scientist",
                            "research analyst", "ai researcher", "ml researcher")
    recent_jobs = sorted(
        history,
        key=lambda j: parse_date(j.get("start_date")) or __import__("datetime").date.min,
        reverse=True,
    )[:2]  # current + one prior
    for job in recent_jobs:
        job_title = (job.get("title") or "").lower()
        if _contains_any(job_title, RESEARCH_TITLE_TERMS):
            # Check production evidence across the full narrative.
            prod_hits = _count_hits(narrative, config.PRODUCTION_EVIDENCE_TERMS)
            if prod_hits < config.RESEARCH_PROD_MIN_HITS:
                if "research_title_no_prod" not in result.penalties:
                    result.penalties.append("research_title_no_prod")
                    result.concerns.append(
                        "research-flavoured title without strong production-deployment evidence"
                    )
                    base *= config.PENALTY_RESEARCH_TITLE_NO_PROD
            break  # only apply once even if multiple research-title roles

    return base


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def structural_score(candidate: dict) -> StructuralResult:
    result = StructuralResult()
    profile = candidate.get("profile", {})

    components = {
        "title_domain": _title_domain_score(profile, result),
        "career_evidence": _career_evidence_score(candidate, result),
        "experience_band": _experience_band_score(profile, result),
        "skills_trust": _skills_trust_score(candidate, result),
        "logistics": _logistics_score(candidate, result),
    }
    base = sum(config.STRUCT_WEIGHTS[k] * v for k, v in components.items())
    result.components = components

    # Hard floor for wholly irrelevant careers (Civil Engineers, Accountants,
    # HR Managers, etc.) that score near zero on both title and career evidence.
    # Without this, a Civil Engineer with 10 YOE and good logistics can
    # outscore a Backend ML engineer who got penalised for being abroad.
    # Threshold: title_domain <= 0.05 (non-tech title) AND career_evidence
    # <= 0.15 (no retrieval/ML narrative; the 0.10 product-roles bonus can
    # push career_evidence to 0.10–0.15 even for wholly irrelevant profiles,
    # so we use 0.15 as the ceiling, not 0.05).
    if components["title_domain"] <= 0.05 and components["career_evidence"] <= 0.15:
        result.penalties.append("irrelevant_career")
        result.concerns.append("no ML/retrieval background found in title or career history")
        base = min(base, 0.08)

    result.score = max(0.0, min(1.0, _apply_penalties(candidate, base, result)))
    return result
