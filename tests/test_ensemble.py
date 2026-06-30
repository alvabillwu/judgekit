"""Tests for multi-judge ensembling."""

import pytest

from judgekit import (
    Rubric,
    Criterion,
    JudgeInput,
    MockJudge,
    OverlapJudge,
    KeywordJudge,
    EnsembleJudge,
    ensemble_score_items,
    judge_agreement,
    score_item,
)
from judgekit.ensemble import _combine


def _rubric():
    return Rubric(
        name="r",
        criteria=(Criterion("c", "d", weight=1.0),),
    )


def _items():
    return [
        JudgeInput(id="a", content="the cat sat", reference="the cat", question="cat"),
        JudgeInput(id="b", content="paris france", reference="paris france", question="capital"),
    ]


class TestCombine:
    def test_mean(self):
        v, rat = _combine([0.0, 1.0, 0.5], ["", "", ""], "mean")
        assert v == pytest.approx(0.5)
        assert "mean" in rat

    def test_median(self):
        v, _ = _combine([0.0, 1.0, 0.5], ["", "", ""], "median")
        assert v == pytest.approx(0.5)

    def test_majority_pass(self):
        v, rat = _combine([0.6, 0.7, 0.2], ["", "", ""], "majority")
        assert v == pytest.approx(2 / 3)
        assert "2/3" in rat

    def test_majority_no_pass(self):
        v, _ = _combine([0.1, 0.2, 0.3], ["", "", ""], "majority")
        assert v == 0.0

    def test_agreement_downweights_disagreement(self):
        agree, _ = _combine([1.0, 1.0, 1.0], ["", "", ""], "agreement")
        disagree, _ = _combine([0.0, 1.0, 0.5], ["", "", ""], "agreement")
        # unanimous => no downweight; spread => downweighted
        assert agree > disagree

    def test_single_judge_agreement(self):
        v, rat = _combine([0.7], ["r"], "agreement")
        assert v == pytest.approx(0.7)
        assert "single" in rat

    def test_empty(self):
        assert _combine([], [], "mean") == (0.0, "no judges")

    def test_unknown_strategy(self):
        with pytest.raises(ValueError):
            _combine([0.5], [""], "bogus")


class TestEnsembleJudge:
    def test_is_judge_backend(self):
        e = EnsembleJudge(judges=[MockJudge(), OverlapJudge()])
        # Can be used wherever a JudgeBackend is expected.
        s = score_item(_rubric(), _items()[0], e)
        assert 0.0 <= s.score <= 1.0
        assert s.verdict_for("c") is not None

    def test_empty_judges_raises(self):
        with pytest.raises(ValueError):
            EnsembleJudge(judges=[])

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            EnsembleJudge(judges=[MockJudge()], strategy="bogus")

    def test_mean_strategy_combines(self):
        # MockJudge and OverlapJudge on the same item produce different scores;
        # the ensemble mean should be their average.
        rubric = _rubric()
        item = _items()[0]
        m = MockJudge().judge(item, rubric.criteria[0]).normalized
        o = OverlapJudge().judge(item, rubric.criteria[0]).normalized
        e = EnsembleJudge(judges=[MockJudge(), OverlapJudge()], strategy="mean")
        v = e.judge(item, rubric.criteria[0])
        assert v.normalized == pytest.approx((m + o) / 2)


class TestEnsembleScoreItems:
    def test_returns_aggregated_and_per_judge(self):
        rubric = _rubric()
        items = _items()
        result = ensemble_score_items(rubric, items, [MockJudge(), OverlapJudge()], strategy="mean")
        assert result.strategy == "mean"
        assert len(result.judge_names) == 2
        assert len(result.aggregated.scored) == 2
        assert len(result.per_judge) == 2
        assert len(result.per_judge[0].scored) == 2

    def test_aggregated_mean_in_range(self):
        result = ensemble_score_items(_rubric(), _items(), [MockJudge(), OverlapJudge(), KeywordJudge(["the", "cat"])])
        assert 0.0 <= result.mean_score <= 1.0


class TestJudgeAgreement:
    def test_perfect_agreement_when_same_judge(self):
        rubric = _rubric()
        items = _items()
        r1 = ensemble_score_items(rubric, items, [MockJudge(), MockJudge()]).per_judge
        ag = judge_agreement(r1)
        assert ag["mean_agreement"] == pytest.approx(1.0)
        assert ag["judges"] == 2

    def test_unanimous_pass_rate(self):
        rubric = _rubric()
        items = _items()
        # Both items fully overlap reference => MockJudge passes both.
        r = [MockJudge()]
        from judgekit import score_items

        res = (score_items(rubric, items, MockJudge()),)
        ag = judge_agreement(res)
        assert ag["n"] == 2

    def test_empty(self):
        assert judge_agreement([])["n"] == 0
