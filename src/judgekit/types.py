"""Core types — the contracts a judge and a rubric plug into.

A `Rubric` is a set of weighted `Criterion`s, each scored on a `Scale` (a
0..1 float, or a small ordinal like 1..5 that gets normalized). A `JudgeInput`
is the thing being judged (an answer, a summary, a generation) plus optional
context (the question, reference). A `Verdict` is one criterion's score + the
judge's rationale. `ScoredItem` aggregates a full rubric's verdicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Scale:
    """A scoring scale. Continuous 0..1 by default; or ordinal 1..max.

    For an ordinal scale (e.g. 1..5 Likert), `normalize(value)` maps a raw
    score into 0..1 so criteria with different scales can be aggregated.
    """

    max: float = 1.0
    min: float = 0.0
    ordinal: bool = False  # True => integer 1..max Likert scale

    def normalize(self, value: float) -> float:
        if self.ordinal:
            # Map 1..max → 0..1: (v - 1) / (max - 1)
            span = max(1e-9, self.max - 1)
            return max(0.0, min(1.0, (value - 1) / span))
        span = max(1e-9, self.max - self.min)
        return max(0.0, min(1.0, (value - self.min) / span))


@dataclass(frozen=True)
class Criterion:
    """One scored dimension of a rubric."""

    name: str
    description: str
    weight: float = 1.0
    scale: Scale = field(default_factory=Scale)


@dataclass(frozen=True)
class Rubric:
    """A named set of weighted criteria."""

    name: str
    criteria: tuple[Criterion, ...]

    def __post_init__(self) -> None:
        if not self.criteria:
            raise ValueError("a rubric must have at least one criterion")
        if any(c.weight < 0 for c in self.criteria):
            raise ValueError("criterion weights must be non-negative")

    @property
    def total_weight(self) -> float:
        return sum(c.weight for c in self.criteria)


@dataclass(frozen=True)
class JudgeInput:
    """The item being judged, plus optional context."""

    id: str
    content: str
    question: str = ""
    reference: str = ""  # ground-truth / expected answer (optional)
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Verdict:
    """One criterion's verdict on one item."""

    criterion: str
    raw_score: float  # in the criterion's scale
    normalized: float  # 0..1
    rationale: str = ""


@dataclass(frozen=True)
class ScoredItem:
    """A full rubric's verdicts for one item + the weighted aggregate."""

    item_id: str
    verdicts: tuple[Verdict, ...]
    score: float  # weighted aggregate in 0..1

    @property
    def passed(self) -> bool:
        return self.score >= 0.5

    def verdict_for(self, criterion: str) -> Optional[Verdict]:
        for v in self.verdicts:
            if v.criterion == criterion:
                return v
        return None


# ── Judge protocol ──────────────────────────────────────────────────────────


from typing import Protocol, runtime_checkable


@runtime_checkable
class JudgeBackend(Protocol):
    """A backend that scores one item against one criterion.

    Returns a raw score in the criterion's scale plus a rationale.
    """

    name: str

    def judge(self, item: JudgeInput, criterion: Criterion) -> Verdict: ...
