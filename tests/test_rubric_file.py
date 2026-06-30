"""Tests for rubric file loading (JSON + minimal YAML) and the CLI."""

import json
import textwrap
from pathlib import Path

import pytest

from judgekit.rubric_file import load_rubric, rubric_from_dict, parse_yaml
from judgekit.types import Scale


class TestRubricFromDict:
    def test_minimal(self):
        r = rubric_from_dict({"name": "r", "criteria": [{"name": "c", "description": "d"}]})
        assert r.name == "r"
        assert len(r.criteria) == 1
        assert r.criteria[0].name == "c"
        assert r.criteria[0].weight == 1.0

    def test_with_scale_and_weight(self):
        r = rubric_from_dict({
            "name": "r",
            "criteria": [{"name": "c", "description": "d", "weight": 2.5, "scale": {"max": 5, "ordinal": True}}],
        })
        c = r.criteria[0]
        assert c.weight == 2.5
        assert c.scale.max == 5
        assert c.scale.ordinal is True

    def test_empty_criteria_raises(self):
        with pytest.raises(ValueError):
            rubric_from_dict({"name": "r", "criteria": []})

    def test_missing_name_raises(self):
        with pytest.raises(ValueError):
            rubric_from_dict({"name": "r", "criteria": [{"description": "d"}]})


class TestParseYaml:
    def test_simple_map(self):
        d = parse_yaml("name: qa\ncount: 5\nflag: true\n")
        assert d == {"name": "qa", "count": 5, "flag": True}

    def test_nested_map_and_seq(self):
        text = textwrap.dedent("""
            name: qa
            criteria:
              - name: relevance
                description: answers the question
                weight: 2.0
              - name: fluency
                description: well written
        """).strip()
        d = parse_yaml(text)
        assert d["name"] == "qa"
        assert d["criteria"][0]["name"] == "relevance"
        assert d["criteria"][0]["weight"] == 2.0
        assert d["criteria"][1]["name"] == "fluency"

    def test_flow_map_scale(self):
        text = "scale: {max: 5, ordinal: true}\n"
        d = parse_yaml(text)
        assert d["scale"] == {"max": 5, "ordinal": True}

    def test_flow_seq(self):
        d = parse_yaml("items: [a, b, c]\n")
        assert d["items"] == ["a", "b", "c"]

    def test_quoted_strings(self):
        d = parse_yaml('desc: "hello: world"\n')
        assert d["desc"] == "hello: world"

    def test_comments_and_blanks_ignored(self):
        d = parse_yaml("# header\nname: x\n\n# trailing\n")
        assert d == {"name": "x"}

    def test_null_values(self):
        d = parse_yaml("a: null\nb: ~\nc:\n")
        assert d["a"] is None
        assert d["b"] is None

    def test_root_must_be_map(self):
        with pytest.raises(ValueError):
            parse_yaml("- a\n- b\n")


class TestLoadRubric:
    def test_load_json(self, tmp_path: Path):
        p = tmp_path / "r.json"
        p.write_text(json.dumps({
            "name": "qa",
            "criteria": [{"name": "c", "description": "d", "scale": {"max": 5, "ordinal": True}}],
        }))
        r = load_rubric(p)
        assert r.name == "qa"
        assert r.criteria[0].scale.ordinal is True

    def test_load_yaml(self, tmp_path: Path):
        p = tmp_path / "r.yaml"
        p.write_text(textwrap.dedent("""
            name: qa
            criteria:
              - name: relevance
                description: answers the question
                weight: 2.0
              - name: fluency
                description: well written
        """))
        r = load_rubric(p)
        assert r.name == "qa"
        assert len(r.criteria) == 2
        assert r.criteria[0].weight == 2.0

    def test_load_yaml_with_flow_scale(self, tmp_path: Path):
        p = tmp_path / "r.yml"
        p.write_text(textwrap.dedent("""
            name: r
            criteria:
              - name: c
                description: d
                scale: {max: 5, ordinal: true}
        """))
        r = load_rubric(p)
        assert r.criteria[0].scale.max == 5
        assert r.criteria[0].scale.ordinal is True


class TestCLI:
    def _write(self, tmp_path: Path, rubric_yaml: str, items_jsonl: str):
        (tmp_path / "rubric.yaml").write_text(textwrap.dedent(rubric_yaml))
        (tmp_path / "items.jsonl").write_text(items_jsonl)
        return tmp_path / "rubric.yaml", tmp_path / "items.jsonl"

    def test_score_mock(self, tmp_path: Path, capsys):
        from judgekit.cli import main

        rubric, items = self._write(
            tmp_path,
            """
            name: qa
            criteria:
              - name: relevance
                description: answers the question
                weight: 2.0
              - name: groundedness
                description: supported by reference
            """,
            '{"id": "q1", "content": "Paris", "question": "capital of France", "reference": "Paris"}\n',
        )
        main(["score", str(rubric), str(items), "--judge", "mock", "--json"])
        out = json.loads(capsys.readouterr().out)
        assert out["rubric"] == "qa"
        assert out["n"] == 1
        assert 0.0 <= out["mean_score"] <= 1.0
        assert len(out["items"][0]["verdicts"]) == 2

    def test_score_overlap_judge(self, tmp_path: Path, capsys):
        from judgekit.cli import main

        rubric, items = self._write(
            tmp_path,
            """
            name: r
            criteria:
              - name: c
                description: d
            """,
            '{"id": "x", "content": "the cat", "reference": "the cat sat"}\n',
        )
        main(["score", str(rubric), str(items), "--judge", "overlap"])
        out = capsys.readouterr().out
        assert "overlap" in out

    def test_calibrate(self, tmp_path: Path, capsys):
        from judgekit.cli import main

        rubric, items = self._write(
            tmp_path,
            """
            name: r
            criteria:
              - name: c
                description: d
            """,
            '{"id": "a", "content": "the cat sat", "reference": "the cat sat"}\n'
            '{"id": "b", "content": "xyz", "reference": "abc"}\n',
        )
        (tmp_path / "truth.json").write_text(json.dumps({"a": 1.0, "b": 0.0}))
        main(["calibrate", str(rubric), str(items), "--truth", str(tmp_path / "truth.json"), "--judge", "mock", "--json"])
        out = json.loads(capsys.readouterr().out)
        assert out["n"] == 2
        assert 0.0 <= out["accuracy"] <= 1.0

    def test_judges_list(self, tmp_path: Path, capsys):
        from judgekit.cli import main

        main(["judges", "--json"])
        out = json.loads(capsys.readouterr().out)
        names = [j["name"] for j in out["judges"]]
        assert "mock" in names and "llm" in names and "overlap" in names
