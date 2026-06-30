"""judgekit — a reusable LLM-as-judge harness.

Build a rubric (weighted criteria on a scale), score items with a pluggable
judge (deterministic mock, cheap proxy, or a real LLM), and calibrate a judge
against ground-truth labels. Designed as a library: drop it into any eval
pipeline (RAG, summarization, classification) and swap judges without touching
the rubric.

Module map:
  judgekit.types       — Rubric, Criterion, JudgeInput, Verdict, ScoredItem
  judgekit.rubric      — rubric definition + weighted aggregation
  judgekit.judges      — JudgeBackend protocol + MockJudge + LLMJudge
  judgekit.proxy       — cheap no-LLM proxy judges (length, overlap, keyword)
  judgekit.calibration — judge-vs-ground-truth agreement metrics
"""

from .types import Rubric, Criterion, JudgeInput, Verdict, ScoredItem, Scale, JudgeBackend
from .rubric import score_item, score_items, RubricResult
from .judges import MockJudge, LLMJudge, get_judge
from .proxy import LengthJudge, OverlapJudge, KeywordJudge
from .calibration import calibrate, CalibrationReport

__version__ = "0.1.0"
__all__ = [
    "Rubric",
    "Criterion",
    "JudgeInput",
    "Verdict",
    "ScoredItem",
    "Scale",
    "score_item",
    "score_items",
    "RubricResult",
    "JudgeBackend",
    "MockJudge",
    "LLMJudge",
    "get_judge",
    "LengthJudge",
    "OverlapJudge",
    "KeywordJudge",
    "calibrate",
    "CalibrationReport",
]
