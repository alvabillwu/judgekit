"""Cheap proxy judges — no LLM, no network.

Useful as fast baselines and as smoke-test judges in CI. Each scores a single
proxy signal (length, token overlap with reference, keyword presence) on the
criterion's scale. They implement the same `JudgeBackend` protocol as the real
judges, so you can A/B a proxy against an LLM judge on the same rubric.
"""

from __future__ import annotations

import re

from .types import JudgeInput, Criterion, Verdict
from .judges import _denormalize, _tokenize


def _norm01(x: float) -> float:
    return max(0.0, min(1.0, x))


class LengthJudge:
    """Scores by content length relative to a target.

    Reward peaks at `target` chars and decays for too-short or too-long
    content. A crude but useful fluency/completeness proxy.
    """

    name = "proxy-length"

    def __init__(self, target: int = 500, tolerance: float = 0.5):
        self.target = target
        self.tolerance = tolerance

    def judge(self, item: JudgeInput, criterion: Criterion) -> Verdict:
        n = len(item.content)
        if self.target == 0:
            normalized = 0.0
        else:
            ratio = n / self.target
            # Within tolerance band => 1.0; decays linearly outside it.
            lo = 1.0 - self.tolerance
            hi = 1.0 + self.tolerance
            if lo <= ratio <= hi:
                normalized = 1.0
            elif ratio < lo:
                normalized = _norm01(ratio / max(lo, 1e-9))
            else:
                normalized = _norm01(hi / max(ratio, 1e-9))
        return Verdict(
            criterion=criterion.name,
            raw_score=_denormalize(normalized, criterion.scale),
            normalized=round(normalized, 6),
            rationale=f"length {n} chars vs target {self.target}",
        )


class OverlapJudge:
    """Scores by token overlap between content and reference (or question).

    Identical to the mock judge's core signal, exposed as a named proxy so a
    rubric can explicitly weigh "groundedness" as a cheap proxy.
    """

    name = "proxy-overlap"

    def judge(self, item: JudgeInput, criterion: Criterion) -> Verdict:
        content = _tokenize(item.content)
        basis = _tokenize(item.reference) if item.reference else _tokenize(item.question)
        if not content or not basis:
            normalized = 0.0
        else:
            normalized = len(content & basis) / len(content)
        return Verdict(
            criterion=criterion.name,
            raw_score=_denormalize(normalized, criterion.scale),
            normalized=round(normalized, 6),
            rationale=f"token overlap with {'reference' if item.reference else 'question'}: {normalized:.2f}",
        )


class KeywordJudge:
    """Scores by the fraction of required keywords present in the content."""

    name = "proxy-keyword"

    def __init__(self, keywords: list[str]):
        self.keywords = [k.lower() for k in keywords]

    def judge(self, item: JudgeInput, criterion: Criterion) -> Verdict:
        if not self.keywords:
            normalized = 0.0
        else:
            content_lower = item.content.lower()
            present = sum(1 for k in self.keywords if k in content_lower)
            normalized = present / len(self.keywords)
        return Verdict(
            criterion=criterion.name,
            raw_score=_denormalize(normalized, criterion.scale),
            normalized=round(normalized, 6),
            rationale=f"keywords present: {normalized:.2f}",
        )
