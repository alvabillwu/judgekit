"""CLI entry point for judgekit.

  judgekit score <rubric.yaml> <items.jsonl> [--judge mock|llm] [--json]
  judgekit calibrate <rubric.yaml> <items.jsonl> --truth <truth.json> [--judge mock]
  judgekit judges                 # list available judges

items.jsonl: one JSON object per line, each with id, content, and optional
question/reference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from . import __version__
from .rubric_file import load_rubric
from .judges import MockJudge, LLMJudge, get_judge
from .proxy import LengthJudge, OverlapJudge, KeywordJudge
from .rubric import score_items
from .types import JudgeInput
from .calibration import calibrate


def _utf8_stdout():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


def _read_items(path: str) -> list[JudgeInput]:
    items: list[JudgeInput] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        items.append(
            JudgeInput(
                id=str(d.get("id", "")),
                content=str(d.get("content", "")),
                question=str(d.get("question", "")),
                reference=str(d.get("reference", "")),
                metadata=d.get("metadata", {}),
            )
        )
    if not items:
        raise SystemExit(f"no items found in {path}")
    return items


def _resolve_judge(kind: str, **kwargs):
    if kind in ("mock", "llm"):
        return get_judge(kind, **kwargs)
    if kind == "length":
        return LengthJudge(target=int(kwargs.get("target", 500)))
    if kind == "overlap":
        return OverlapJudge()
    if kind == "keyword":
        return KeywordJudge(keywords=kwargs.get("keywords", "").split(","))
    raise SystemExit(f"unknown judge: {kind}")


def cmd_score(args) -> int:
    rubric = load_rubric(args.rubric)
    items = _read_items(args.items)
    kwargs = {}
    if args.judge == "llm":
        kwargs["model"] = args.model
    if args.judge == "length":
        kwargs["target"] = args.target
    if args.judge == "keyword":
        kwargs["keywords"] = args.keywords
    judge = _resolve_judge(args.judge, **kwargs)
    result = score_items(rubric, items, judge)

    if args.json:
        out = {
            "rubric": rubric.name,
            "judge": judge.name,
            "n": len(result.scored),
            "mean_score": round(result.mean_score, 4),
            "pass_rate": round(result.pass_rate, 4),
            "items": [
                {
                    "id": s.item_id,
                    "score": round(s.score, 4),
                    "passed": s.passed,
                    "verdicts": [
                        {"criterion": v.criterion, "normalized": round(v.normalized, 4), "rationale": v.rationale}
                        for v in s.verdicts
                    ],
                }
                for s in result.scored
            ],
        }
        print(json.dumps(out, indent=2))
    else:
        print(f"Rubric: {rubric.name}  Judge: {judge.name}  Items: {len(result.scored)}")
        print(f"  mean: {result.mean_score:.3f}  pass-rate: {result.pass_rate:.1%}")
        print()
        for s in result.scored:
            verdicts = ", ".join(f"{v.criterion}={v.normalized:.2f}" for v in s.verdicts)
            mark = "✓" if s.passed else "✗"
            print(f"  {mark} {s.item_id:<20} {s.score:.3f}  [{verdicts}]")
    return 0


def cmd_calibrate(args) -> int:
    rubric = load_rubric(args.rubric)
    items = _read_items(args.items)
    truth_raw = json.loads(Path(args.truth).read_text(encoding="utf-8"))
    truth = {str(k): float(v) for k, v in truth_raw.items()}
    judge = _resolve_judge(args.judge)
    result = score_items(rubric, items, judge)
    rep = calibrate(result, truth, threshold=args.threshold)
    if args.json:
        print(json.dumps(rep.as_dict(), indent=2))
    else:
        d = rep.as_dict()
        print(f"Calibration  (n={d['n']}, judge={judge.name}, threshold={args.threshold})")
        print(f"  accuracy:  {d['accuracy']:.3f}")
        print(f"  precision: {d['precision']:.3f}  recall: {d['recall']:.3f}  f1: {d['f1']:.3f}")
        print(f"  MAE:       {d['mae']:.3f}")
        print(f"  confusion: {d['confusion']}")
    return 0


def cmd_judges(args) -> int:
    judges = [
        ("mock", "deterministic heuristic (no network)"),
        ("length", "proxy: content length vs target"),
        ("overlap", "proxy: token overlap with reference/question"),
        ("keyword", "proxy: required-keyword presence"),
        ("llm", "LLM judge (optional openai; needs OPENAI_API_KEY)"),
    ]
    if args.json:
        print(json.dumps({"judges": [{"name": n, "description": d} for n, d in judges]}, indent=2))
    else:
        for n, d in judges:
            print(f"  {n:<10} {d}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="judgekit", description="LLM-as-judge harness.")
    parser.add_argument("--version", action="version", version=f"judgekit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score", help="Score items against a rubric")
    p_score.add_argument("rubric", help="path to rubric .yaml/.yml/.json")
    p_score.add_argument("items", help="path to items .jsonl")
    p_score.add_argument("--judge", default="mock", help="mock|length|overlap|keyword|llm (default: mock)")
    p_score.add_argument("--model", default="gpt-4o-mini", help="model for llm judge")
    p_score.add_argument("--target", type=int, default=500, help="target length for length judge")
    p_score.add_argument("--keywords", default="", help="comma-separated keywords for keyword judge")
    p_score.add_argument("--json", action="store_true")
    p_score.set_defaults(func=cmd_score)

    p_cal = sub.add_parser("calibrate", help="Calibrate a judge against ground-truth labels")
    p_cal.add_argument("rubric", help="path to rubric")
    p_cal.add_argument("items", help="path to items .jsonl")
    p_cal.add_argument("--truth", required=True, help="path to truth JSON ({id: score 0..1})")
    p_cal.add_argument("--judge", default="mock")
    p_cal.add_argument("--threshold", type=float, default=0.5)
    p_cal.add_argument("--json", action="store_true")
    p_cal.set_defaults(func=cmd_calibrate)

    p_j = sub.add_parser("judges", help="List available judges")
    p_j.add_argument("--json", action="store_true")
    p_j.set_defaults(func=cmd_judges)

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    _utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    code = args.func(args)
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()
