"""Rubric scoring — run a judge over every criterion and aggregate.

The rubric is judge-agnostic: `score_item(rubric, item, judge)` asks the judge
to score each criterion, normalizes each score to 0..1, and returns a weighted
aggregate. Swap the judge (mock / proxy / LLM) without touching the rubric.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import Rubric, JudgeInput, Verdict, ScoredItem, JudgeBackend


@dataclass(frozen=True)
class RubricResult:
    """The result of scoring a batch of items against one rubric."""

    rubric_name: str
    scored: tuple[ScoredItem, ...]

    @property
    def mean_score(self) -> float:
        if not self.scored:
            return 0.0
        return sum(s.score for s in self.scored) / len(self.scored)

    @property
    def pass_rate(self) -> float:
        if not self.scored:
            return 0.0
        return sum(1 for s in self.scored if s.passed) / len(self.scored)

    def by_item(self, item_id: str) -> ScoredItem | None:
        for s in self.scored:
            if s.item_id == item_id:
                return s
        return None


def score_item(rubric: Rubric, item: JudgeInput, judge: JudgeBackend) -> ScoredItem:
    """Score one item against every criterion of a rubric."""
    verdicts: list[Verdict] = []
    weighted_sum = 0.0
    total_weight = 0.0
    for criterion in rubric.criteria:
        v = judge.judge(item, criterion)
        verdicts.append(v)
        weighted_sum += v.normalized * criterion.weight
        total_weight += criterion.weight
    aggregate = weighted_sum / total_weight if total_weight else 0.0
    return ScoredItem(item_id=item.id, verdicts=tuple(verdicts), score=aggregate)


def score_items(
    rubric: Rubric,
    items: list[JudgeInput],
    judge: JudgeBackend,
) -> RubricResult:
    """Score a batch of items against a rubric."""
    scored = tuple(score_item(rubric, item, judge) for item in items)
    return RubricResult(rubric_name=rubric.name, scored=scored)
