"""redrob-ranker: candidate ranking pipeline for the Redrob India Runs
Data & AI Challenge (Track 1, Intelligent Candidate Discovery)."""

from .pipeline import score_candidate, select_top, write_submission

__all__ = ["score_candidate", "select_top", "write_submission"]
__version__ = "1.0.0"
