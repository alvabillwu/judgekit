"""Multi-judge ensembling — combine several judges' scores into one.

Different judges have different blind spots. An ensemble aggregates N judges'
per-item scores with a chosen strategy (mean, median, majority pass, or
agreement-filtered mean that down-weights items where judges disagree). The
ensemble implements the same `JudgeBackend` protocol, so it can itself be
fed to `score_items` — composable.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Sequence

from .types import Rubric, Criterion, JudgeInput, Verdict, JudgeBackend
from .rubric import score_item, score_items, RubricResult


@dataclass
class EnsembleJudge:
    """A judge that aggregates several sub-judges.

    Each sub-judge scores the item on a criterion; the ensemble combines the
    *normalized* scores per criterion and re-derives a rationale.
    """

    judges: Sequence[JudgeBackend]
    strategy: str = "mean"  # mean | median | majority | agreement
    name: str = "ensemble"

    def __post_init__(self) -> None:
        if not self.judges:
            raise ValueError("an ensemble needs at least one judge")
        if self.strategy not in ("mean", "median", "majority", "agreement"):
            raise ValueError(f"unknown strategy: {self.strategy!r}")

    def judge(self, item: JudgeInput, criterion: Criterion) -> Verdict:
        verdicts = [j.judge(item, criterion) for j in self.judges]
        scores = [v.normalized for v in verdicts]
        norm, rationale = _combine(scores, [v.rationale for v in verdicts], self.strategy)
        # Raw score is reported in the ensemble's own 0..1 scale (continuous).
        return Verdict(
            criterion=criterion.name,
            raw_score=round(norm, 6),
            normalized=round(norm, 6),
            rationale=rationale,
        )


def _combine(
    scores: list[float],
    rationales: list[str],
    strategy: str,
) -> tuple[float, str]:
    if not scores:
        return 0.0, "no judges"
    if strategy == "mean":
        combined = statistics.fmean(scores)
        spread = max(scores) - min(scores)
        return combined, f"mean of {len(scores)} (spread {spread:.2f})"
    if strategy == "median":
        combined = statistics.median(scores)
        spread = max(scores) - min(scores)
        return float(combined), f"median of {len(scores)} (spread {spread:.2f})"
    if strategy == "majority":
        # Fraction of judges passing (>= 0.5) — a 0..1 score.
        passes = sum(1 for s in scores if s >= 0.5)
        combined = passes / len(scores)
        return combined, f"{passes}/{len(scores)} judges pass"
    if strategy == "agreement":
        # Mean but down-weighted by disagreement: score * (1 - stdev).
        if len(scores) == 1:
            return scores[0], "single judge (no disagreement)"
        mean = statistics.fmean(scores)
        stdev = statistics.pstdev(scores)
        adjusted = mean * (1.0 - min(1.0, stdev))
        return adjusted, f"agreement-weighted mean={mean:.2f} stdev={stdev:.2f}"
    raise ValueError(f"unknown strategy: {strategy!r}")


@dataclass
class EnsembleResult:
    """An ensemble run over a batch — per-item aggregated scores + per-judge detail."""

    rubric_name: str
    strategy: str
    judge_names: tuple[str, ...]
    aggregated: RubricResult  # the ensemble's combined scores
    per_judge: tuple[RubricResult, ...] = field(default_factory=tuple)  # each judge's own run

    @property
    def mean_score(self) -> float:
        return self.aggregated.mean_score


def ensemble_score_items(
    rubric: Rubric,
    items: list[JudgeInput],
    judges: Sequence[JudgeBackend],
    strategy: str = "mean",
) -> EnsembleResult:
    """Score items with an ensemble of judges and return aggregated + per-judge results.

    The aggregated scores come from running `EnsembleJudge` (which combines each
    criterion across judges); `per_judge` holds each judge's independent run for
    analysis (e.g. inter-judge agreement).
    """
    ensemble = EnsembleJudge(judges=judges, strategy=strategy)
    aggregated = score_items(rubric, items, ensemble)
    per_judge = tuple(score_items(rubric, items, j) for j in judges)
    return EnsembleResult(
        rubric_name=rubric.name,
        strategy=strategy,
        judge_names=tuple(getattr(j, "name", "judge") for j in judges),
        aggregated=aggregated,
        per_judge=per_judge,
    )


def judge_agreement(per_judge: Sequence[RubricResult]) -> dict:
    """Inter-judge agreement: mean pairwise Pearson-style score correlation.

    Returns per-item agreement (1 - stdev across judges) averaged, plus the
    fraction of items where all judges agree on pass/fail.
    """
    if not per_judge:
        return {"mean_agreement": 0.0, "unanimous_pass_rate": 0.0, "n": 0}
    n_items = len(per_judge[0].scored)
    judges = len(per_judge)
    agreements: list[float] = []
    unanimous = 0
    for i in range(n_items):
        scores = [rj.scored[i].score for rj in per_judge]
        if len(scores) == 1:
            agreements.append(1.0)
        else:
            agreements.append(1.0 - min(1.0, statistics.pstdev(scores)))
        passes = [s >= 0.5 for s in scores]
        if all(passes) or not any(passes):
            unanimous += 1
    return {
        "mean_agreement": round(statistics.fmean(agreements), 4),
        "unanimous_pass_rate": round(unanimous / n_items, 4) if n_items else 0.0,
        "n": n_items,
        "judges": judges,
    }
