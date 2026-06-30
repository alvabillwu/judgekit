"""Calibration — how well does a judge agree with ground-truth labels?

Given a judge's per-item scores and the ground-truth pass/fail (or score) for
each item, compute agreement metrics. This is how you decide whether a cheap
proxy judge is good enough or whether you need the LLM judge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .rubric import RubricResult


@dataclass
class CalibrationReport:
    """Agreement between a judge's scores and ground-truth labels."""

    n: int = 0
    accuracy: float = 0.0  # fraction where pass/fail matches (threshold 0.5)
    precision: float = 0.0  # of judge-passed, fraction truly passing
    recall: float = 0.0  # of truly passing, fraction judge passed
    f1: float = 0.0
    mean_absolute_error: float = 0.0  # |judge_score - truth_score| avg
    confusion: dict = field(default_factory=dict)  # {tp, fp, tn, fn}

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "mae": round(self.mean_absolute_error, 4),
            "confusion": self.confusion,
        }


def calibrate(
    result: RubricResult,
    ground_truth: dict[str, float],
    threshold: float = 0.5,
) -> CalibrationReport:
    """Calibrate a RubricResult against ground-truth item scores.

    Args:
        result: the judge's scored items.
        ground_truth: {item_id: truth_score_in_0..1}.
        threshold: score >= threshold counts as "pass".
    """
    tp = fp = tn = fn = 0
    abs_errors: list[float] = []
    matched = 0

    for s in result.scored:
        truth = ground_truth.get(s.item_id)
        if truth is None:
            continue
        matched += 1
        abs_errors.append(abs(s.score - truth))
        judge_pass = s.score >= threshold
        truth_pass = truth >= threshold
        if judge_pass and truth_pass:
            tp += 1
        elif judge_pass and not truth_pass:
            fp += 1
        elif not judge_pass and truth_pass:
            fn += 1
        else:
            tn += 1

    n = matched
    if n == 0:
        return CalibrationReport()

    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    mae = sum(abs_errors) / n

    return CalibrationReport(
        n=n,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_absolute_error=mae,
        confusion={"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    )
