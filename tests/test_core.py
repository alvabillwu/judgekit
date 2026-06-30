"""Tests for judgekit core: types, rubric scoring, judges, calibration."""

import pytest

from judgekit import (
    Rubric,
    Criterion,
    Scale,
    JudgeInput,
    MockJudge,
    LengthJudge,
    OverlapJudge,
    KeywordJudge,
    score_item,
    score_items,
    calibrate,
    get_judge,
)
from judgekit.judges import _parse_llm_verdict, _denormalize


def _rubric():
    return Rubric(
        name="qa",
        criteria=(
            Criterion("relevance", "Does it answer the question?", weight=2.0),
            Criterion("fluency", "Is it well-written?", weight=1.0),
        ),
    )


class TestTypes:
    def test_scale_continuous_normalize(self):
        s = Scale(min=0.0, max=1.0)
        assert s.normalize(0.5) == pytest.approx(0.5)
        assert s.normalize(-1) == 0.0
        assert s.normalize(2) == 1.0

    def test_scale_ordinal_normalize(self):
        s = Scale(max=5, ordinal=True)
        assert s.normalize(1) == pytest.approx(0.0)
        assert s.normalize(3) == pytest.approx(0.5)
        assert s.normalize(5) == pytest.approx(1.0)

    def test_rubric_requires_criteria(self):
        with pytest.raises(ValueError):
            Rubric(name="x", criteria=())

    def test_rubric_rejects_negative_weight(self):
        with pytest.raises(ValueError):
            Rubric(name="x", criteria=(Criterion("c", "d", weight=-1),))

    def test_rubric_total_weight(self):
        r = _rubric()
        assert r.total_weight == 3.0


class TestRubricScoring:
    def test_score_item_returns_verdicts_per_criterion(self):
        r = _rubric()
        item = JudgeInput(id="i1", content="Paris is the capital of France.",
                          question="What is the capital of France?",
                          reference="The capital of France is Paris.")
        s = score_item(r, item, MockJudge())
        assert s.item_id == "i1"
        assert len(s.verdicts) == 2
        names = {v.criterion for v in s.verdicts}
        assert names == {"relevance", "fluency"}
        assert 0.0 <= s.score <= 1.0

    def test_weighted_aggregation(self):
        # relevance weight 2, fluency weight 1.
        r = _rubric()
        item = JudgeInput(id="i", content="Paris",
                          question="capital of France", reference="Paris")
        s = score_item(r, item, MockJudge())
        rel = s.verdict_for("relevance")
        flu = s.verdict_for("fluency")
        expected = (rel.normalized * 2 + flu.normalized * 1) / 3
        assert s.score == pytest.approx(expected, rel=1e-6)

    def test_score_items_batch(self):
        r = _rubric()
        items = [
            JudgeInput(id=f"i{i}", content=f"text {i}", question="q", reference="text")
            for i in range(4)
        ]
        result = score_items(r, items, MockJudge())
        assert len(result.scored) == 4
        assert 0.0 <= result.mean_score <= 1.0
        assert 0.0 <= result.pass_rate <= 1.0
        assert result.by_item("i2") is not None
        assert result.by_item("nope") is None

    def test_passed_threshold(self):
        r = Rubric(name="r", criteria=(Criterion("c", "d"),))
        item = JudgeInput(id="i", content="hello world", question="greeting", reference="hello world")
        s = score_item(r, item, MockJudge())
        # full overlap with reference => normalized ~1.0 => passes
        assert s.passed is True


class TestMockJudge:
    def test_deterministic(self):
        r = _rubric()
        item = JudgeInput(id="i", content="a b c", reference="a b c")
        v1 = MockJudge().judge(item, r.criteria[0])
        v2 = MockJudge().judge(item, r.criteria[0])
        assert v1 == v2

    def test_empty_content_scores_zero(self):
        c = Criterion("c", "d")
        v = MockJudge().judge(JudgeInput(id="i", content=""), c)
        assert v.normalized == 0.0

    def test_uses_reference_when_present(self):
        c = Criterion("c", "d")
        v_ref = MockJudge().judge(
            JudgeInput(id="i", content="paris", reference="paris"), c
        )
        v_q = MockJudge().judge(
            JudgeInput(id="i", content="paris", question="capital"), c
        )
        # full overlap with reference => 1.0; partial with question => lower
        assert v_ref.normalized == 1.0
        assert v_q.normalized < 1.0


class TestProxyJudges:
    def test_length_judge_at_target(self):
        c = Criterion("c", "d")
        j = LengthJudge(target=10, tolerance=0.5)
        v = j.judge(JudgeInput(id="i", content="x" * 10), c)
        assert v.normalized == 1.0

    def test_length_judge_too_short(self):
        c = Criterion("c", "d")
        j = LengthJudge(target=100, tolerance=0.2)
        v = j.judge(JudgeInput(id="i", content="short"), c)
        assert v.normalized < 1.0

    def test_overlap_judge(self):
        c = Criterion("c", "d")
        v = OverlapJudge().judge(
            JudgeInput(id="i", content="the cat sat", reference="the cat"), c
        )
        # content tokens {the,cat,sat}, ref {the,cat} => 2/3
        assert v.normalized == pytest.approx(2 / 3)

    def test_keyword_judge(self):
        c = Criterion("c", "d")
        j = KeywordJudge(["python", "async", "await"])
        v_full = j.judge(JudgeInput(id="i", content="python async await"), c)
        v_half = j.judge(JudgeInput(id="i", content="python only"), c)
        assert v_full.normalized == 1.0
        assert v_half.normalized == pytest.approx(1 / 3)


class TestCalibration:
    def _result(self, scores: list[float]):
        from judgekit import RubricResult, ScoredItem
        from judgekit.types import Verdict
        return RubricResult(
            rubric_name="r",
            scored=tuple(
                ScoredItem(item_id=f"i{i}", verdicts=(Verdict("c", s, s),), score=s)
                for i, s in enumerate(scores)
            ),
        )

    def test_perfect_agreement(self):
        result = self._result([0.9, 0.8, 0.1, 0.2])
        truth = {"i0": 1.0, "i1": 0.9, "i2": 0.0, "i3": 0.1}
        rep = calibrate(result, truth)
        assert rep.n == 4
        assert rep.accuracy == 1.0
        assert rep.precision == 1.0
        assert rep.recall == 1.0
        assert rep.f1 == 1.0

    def test_partial_agreement(self):
        result = self._result([0.9, 0.1, 0.9, 0.1])  # judge: pass,fail,pass,fail
        truth = {"i0": 1.0, "i1": 1.0, "i2": 0.0, "i3": 0.0}  # truth: pass,pass,fail,fail
        rep = calibrate(result, truth)
        # judge passed i0 (correct) + i2 (wrong); failed i1 (wrong) + i3 (correct)
        assert rep.confusion == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
        assert rep.accuracy == 0.5

    def test_missing_truth_skipped(self):
        result = self._result([0.9, 0.1])
        rep = calibrate(result, {"i0": 1.0})  # i1 has no truth
        assert rep.n == 1

    def test_mae(self):
        result = self._result([0.9, 0.1])
        rep = calibrate(result, {"i0": 0.5, "i1": 0.5})
        assert rep.mean_absolute_error == pytest.approx(0.4)

    def test_empty(self):
        rep = calibrate(self._result([]), {})
        assert rep.n == 0


class TestLLMJudgeInternals:
    def test_parse_clean_json(self):
        score, rat = _parse_llm_verdict('{"score": 4, "rationale": "good"}')
        assert score == 4.0
        assert rat == "good"

    def test_parse_embedded(self):
        score, _ = _parse_llm_verdict('Sure! {"score": 0.5, "rationale": "ok"} thanks')
        assert score == 0.5

    def test_parse_unparseable(self):
        score, rat = _parse_llm_verdict("no json here")
        assert score == 0.0
        assert "unparseable" in rat

    def test_denormalize_ordinal(self):
        from judgekit.types import Scale

        assert _denormalize(1.0, Scale(max=5, ordinal=True)) == 5
        assert _denormalize(0.0, Scale(max=5, ordinal=True)) == 1


class TestFactory:
    def test_get_mock(self):
        assert isinstance(get_judge("mock"), MockJudge)

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError):
            get_judge("bogus")
