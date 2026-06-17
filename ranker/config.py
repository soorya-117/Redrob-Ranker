"""Central configuration for the redrob-ranker pipeline.

Every tunable lives here so that scoring behaviour is auditable in one place.
All keyword lexicons are derived directly from the released job description
(data/job_description.txt) -- see README "Scoring model" for the mapping.
"""

from __future__ import annotations

import datetime as dt

# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

# All recency math (last_active decay, career spans for open-ended roles) is
# anchored to a fixed date instead of `date.today()`. Without this, re-running
# the pipeline on different days produces different scores, which breaks
# Stage 3 reproduction. Chosen as a date shortly after the dataset snapshot.
REFERENCE_DATE = dt.date(2026, 6, 1)

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

# EMBEDDING MODEL: finalized to BAAI/bge-small-en-v1.5.
# If this changes, pre-computation MUST be re-run across the 465MB candidate pool.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
EMBED_BATCH_SIZE = 256

# Career-history text used for embedding: most recent N roles, each truncated.
EMBED_CAREER_ENTRIES = 3
EMBED_CAREER_CHARS = 300

# ---------------------------------------------------------------------------
# Final score composition
# ---------------------------------------------------------------------------

# final = (W_SEMANTIC * semantic + W_STRUCTURAL * structural)
#         * behavioral_multiplier * integrity_multiplier
#
# Structural dominates because the JD is rule-heavy: it spells out explicit
# disqualifiers (consulting-only, research-only, stale hands-on, ...) that an
# embedding similarity cannot see. Semantic similarity is kept at 0.40 to
# surface "plain-language Tier 5" candidates whose descriptions match the role
# without using the fashionable vocabulary.
W_SEMANTIC = 0.40
W_STRUCTURAL = 0.60

# ---------------------------------------------------------------------------
# Structural sub-weights (sum to 1.0)
# ---------------------------------------------------------------------------

STRUCT_WEIGHTS = {
    "title_domain": 0.30,   # is this person actually an ML/search/IR engineer
    "career_evidence": 0.30, # shipped retrieval/ranking/reco at product companies
    "experience_band": 0.15, # JD band 5-9y, soft edges
    "skills_trust": 0.15,    # JD skills weighted by endorsements + duration
    "education_tier": 0.05,  # IIT/BITS/IISc proxy
    "logistics": 0.05,       # location, notice period, relocation, work mode
}

# Penalty multipliers applied on top of the weighted structural score.
PENALTY_CONSULTING_ONLY = 0.25   # entire career in IT services/consulting
PENALTY_RESEARCH_ONLY = 0.30     # research roles only, no production signal
PENALTY_TITLE_CHASER = 0.60      # serial short-tenure job hopping
PENALTY_CV_ONLY = 0.50           # CV/speech/robotics with no NLP/IR exposure
PENALTY_STALE_HANDS_ON = 0.65    # 18+ months in pure architecture/leadership
PENALTY_KEYWORD_STUFFER = 0.05   # AI skill list attached to a non-tech career
# Soft penalty: research-flavoured title (AI Research Engineer, Research Scientist
# etc.) at a product company whose current/recent role descriptions contain fewer
# than RESEARCH_PROD_MIN_HITS production-deployment hits. The JD is explicit:
# "research without production" is a disqualifier. We can't hard-exclude
# because many "AI Research Engineer" titles at Indian product startups do ship;
# we penalise proportionally and let structural evidence rescue good ones.
PENALTY_RESEARCH_TITLE_NO_PROD = 0.75
RESEARCH_PROD_MIN_HITS = 2  # minimum production_evidence hits to waive the penalty

TITLE_CHASER_MIN_ROLES = 3
TITLE_CHASER_MAX_AVG_TENURE_MONTHS = 20
STALE_HANDS_ON_MONTHS = 18
KEYWORD_STUFFER_MIN_JD_SKILLS = 4

# ---------------------------------------------------------------------------
# Tier-1 company prestige (career evidence bonus)
# ---------------------------------------------------------------------------

# Product/tech companies where shipping at scale is part of the job — not
# consulting firms or IT services. Matched as a substring of career_history
# company name (lowercase). Kept to companies where AI/ML at scale is
# institutionalised so the bonus is a genuine signal, not flattery.
TIER_1_COMPANIES = frozenset({
    # Global top-tier product
    "google", "meta", "microsoft", "apple", "amazon", "netflix", "openai",
    "deepmind", "anthropic", "twitter", "linkedin", "uber", "airbnb",
    "stripe", "salesforce", "nvidia",
    # India tier-1 product (strong ML infra)
    "zomato", "swiggy", "paytm", "phonepe", "razorpay", "cred", "meesho",
    "flipkart", "ola", "byju", "unacademy", "freshworks", "zepto", "blinkit",
    "sarvam", "yellow.ai", "observe.ai", "mad street den",
    "lenskart", "urban company", "policybazaar", "groww", "zerodha",
    "navi", "slice", "healthifyme", "browserstack", "cleartax", "chargebee",
    "postman", "hasura", "setu", "cashfree",
})
TIER_1_COMPANY_BONUS = 0.08  # added to career_evidence component score

# ---------------------------------------------------------------------------
# Recruiter-revealed behavioral signals
# ---------------------------------------------------------------------------

# saved_by_recruiters_30d: how many recruiters bookmarked this profile.
# This is a revealed preference — recruiters already found this person
# interesting, which is direct evidence of market fit for the role type.
RECRUITER_SAVE_MAX = 5        # cap contribution at 5 saves
RECRUITER_SAVE_BONUS = 0.01   # per save (max +0.05 multiplier)

# profile_views_received_30d: passive visibility. Weakly correlated with
# reachability, applied as a very small bonus to avoid over-weighting.
PROFILE_VIEWS_THRESHOLD = 10  # views in 30d to qualify for bonus
PROFILE_VIEWS_BONUS = 0.01

# applications_submitted_30d: candidate is actively job-hunting — positive
# availability signal complementary to open_to_work_flag.
APP_SUBMITTED_BONUS = 0.02  # if applications_submitted_30d > 0

# ---------------------------------------------------------------------------
# Experience band (JD: "5-9 years ... a range, not a requirement")
# ---------------------------------------------------------------------------

EXP_IDEAL_LO = 5.0
EXP_IDEAL_HI = 9.0
EXP_SOFT_LO = 4.0    # JD explicitly considers strong candidates outside band
EXP_SOFT_HI = 11.0
EXP_HARD_FLOOR = 3.0 # below this, "senior / founding team" is implausible

# ---------------------------------------------------------------------------
# Behavioral multiplier bounds
# ---------------------------------------------------------------------------

BEHAVIORAL_FLOOR = 0.30
BEHAVIORAL_CEILING = 1.15

# last_active decay steps: (max_days_inactive, multiplier)
# Deprecated: replaced by continuous exponential decay in behavioral.py
# ACTIVITY_DECAY = [(14, 1.00), (45, 0.95), (90, 0.85), (180, 0.70)]
# ACTIVITY_DECAY_STALE = 0.50  # inactive > 180 days ("not actually available")

# recruiter_response_rate steps: (min_rate, multiplier)
RESPONSE_RATE_STEPS = [(0.60, 1.00), (0.30, 0.90), (0.10, 0.75)]
RESPONSE_RATE_FLOOR = 0.60

# ---------------------------------------------------------------------------
# Integrity / honeypot multipliers
# ---------------------------------------------------------------------------

INTEGRITY_FATAL = 0.02      # >= 2 hard inconsistencies: effectively excluded
INTEGRITY_HARD = 0.10       # exactly 1 hard inconsistency — tightened to keep honeypots out of top-100
INTEGRITY_SOFT_DECAY = 0.90 # per soft inconsistency

# Hard-flag thresholds
HONEYPOT_YOE_SPAN_SLACK_YEARS = 2.0    # claimed yoe vs observable career span
HONEYPOT_DURATION_MISMATCH_MONTHS = 6  # stated duration vs date arithmetic
HONEYPOT_EXPERT_ZERO_DURATION_MIN = 3  # >=3 "expert" skills never actually used

# ---------------------------------------------------------------------------
# Lexicons (lowercase substring matching)
# ---------------------------------------------------------------------------

# Skills the JD names directly or by family ("things you absolutely need" +
# "things we'd like").
JD_SKILLS = {
    "embedding", "embeddings", "sentence-transformers", "sentence transformers",
    "bge", "e5", "openai embeddings",
    "vector", "pinecone", "weaviate", "qdrant", "milvus", "faiss",
    "opensearch", "elasticsearch", "bm25",
    "retrieval", "information retrieval", "semantic search", "hybrid search",
    "ranking", "learning to rank", "ltr", "re-ranking", "reranking",
    "recommendation", "recommender", "recsys",
    "nlp", "natural language processing",
    "llm", "large language model", "fine-tuning", "fine-tuning llms",
    "lora", "qlora", "peft", "rag",
    "python", "pytorch", "transformers", "transformer",
    "ndcg", "mrr", "a/b testing", "ab testing", "xgboost",
}

# Evidence of having *built and shipped* retrieval-class systems -- matched
# against career_history descriptions, headline and summary.
RETRIEVAL_EVIDENCE_TERMS = (
    "retrieval", "ranking", "search", "recommendation", "recommender",
    "embedding", "vector", "semantic", "relevance", "bm25", "elasticsearch",
    "opensearch", "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "information retrieval", "learning to rank", "re-rank", "rerank",
    "two-tower", "ndcg", "personalization", "query understanding",
    "matching engine", "candidate generation",
)

PRODUCTION_EVIDENCE_TERMS = (
    "production", "shipped", "deployed", "launched", "real users", "scale",
    "latency", "a/b", "monitoring", "served", "in prod", "rollout",
)

ML_EVIDENCE_TERMS = (
    "machine learning", "ml model", "ml pipeline", "deep learning", "pytorch",
    "tensorflow", "fine-tun", "llm", "nlp", "feature engineering",
    "model training", "inference",
)

# Title classification
ENGINEERING_TITLE_TERMS = (
    "engineer", "developer", "scientist", "ml ", " ml", "machine learning",
    "ai ", " ai", "data scientist", "sde", "swe", "programmer", "architect",
)

ML_TITLE_TERMS = (
    "machine learning", "ml engineer", "ai engineer", "data scientist",
    "applied scientist", "nlp", "search", "relevance", "recommendation",
    "recommender", "information retrieval", "deep learning", "llm",
)

ADJACENT_TITLE_TERMS = (
    "data engineer", "backend", "software engineer", "full stack",
    "platform engineer", "sde", "swe",
)

# Titles the JD's keyword-stuffer trap pairs with perfect AI skill lists.
NON_TECH_TITLE_TERMS = (
    "marketing", "sales", "hr ", "hr manager", "human resources", "recruiter",
    "accountant", "finance", "operations manager", "customer support",
    "business analyst", "project manager", "product manager",
    "graphic designer", "content writer", "civil engineer",
    "mechanical engineer", "teacher", "consultant - business", "legal",
    "administrative", "office manager",
)

# Leadership-only titles -> "hasn't written production code in 18 months".
LEADERSHIP_TITLE_TERMS = (
    "head of", "director", "vp ", "vice president", "chief", "cto",
    "engineering manager", "delivery manager", "general manager",
    "solution architect", "enterprise architect", "principal architect",
)

HANDS_ON_VERBS = (
    "implemented", "built", "wrote", "coded", "developed", "shipped",
    "debugged", "optimized", "refactored",
)

# Consulting / pure-services detection. The JD names six firms "etc."; the
# industry label in this dataset is the more robust signal, with the name
# list as a backstop.
CONSULTING_INDUSTRIES = ("it services", "consulting", "outsourcing", "bpo")
CONSULTING_FIRMS = (
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mindtree", "ltimindtree", "mphasis",
    "ibm global services", "dxc", "ntt data", "genpact",
)

RESEARCH_TITLE_TERMS = ("research", "researcher", "postdoc", "phd ")
RESEARCH_INDUSTRIES = ("academic", "research", "education", "university")

CV_SPEECH_ROBOTICS_TERMS = (
    "computer vision", "image classification", "object detection", "opencv",
    "speech recognition", "tts", "asr", "robotics", "slam", "autonomous",
    "image segmentation", "video analytics", "face recognition",
)
NLP_IR_TERMS = (
    "nlp", "natural language", "text", "retrieval", "search", "ranking",
    "recommendation", "llm", "language model", "embedding", "information retrieval",
)

# ---------------------------------------------------------------------------
# Logistics (JD "On location, comp, and logistics")
# ---------------------------------------------------------------------------

LOCATION_PREFERRED = ("pune", "noida")
LOCATION_WELCOME = ("hyderabad", "mumbai", "delhi", "gurgaon", "gurugram",
                    "ghaziabad", "faridabad", "new delhi")
# Other Indian cities: fine if willing to relocate (JD: "open to relocation
# candidates from Tier-1 Indian cities", quarterly travel expected).
LOCATION_INDIA_RELOCATE = 0.75
LOCATION_INDIA_NO_RELOCATE = 0.55
LOCATION_ABROAD = 0.20  # "case-by-case, but we don't sponsor work visas"

NOTICE_STEPS = [(30, 1.00), (60, 0.85), (90, 0.70)]
NOTICE_LONG = 0.55  # > 90 days: "the bar gets higher"
