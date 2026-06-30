"""Judge backends.

  - MockJudge: deterministic heuristic, no network. A real baseline, not a
    random stub — scores by how well the item content overlaps the reference
    (or the question) so identical inputs always give identical verdicts.
  - LLMJudge: optional, lazy-imports `openai`. Asks an LLM to score an item
    against a criterion on the criterion's scale and parses the JSON verdict.

Both implement the `JudgeBackend` protocol, so they're interchangeable.
"""

from __future__ import annotations

import importlib
import json
import os
import re
from typing import Optional

from .types import JudgeInput, Criterion, Verdict, Scale


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[A-Za-z0-9]+", text) if t}


def _overlap_ratio(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a)


class MockJudge:
    """Deterministic heuristic judge.

    For a criterion, scores by content/reference (or content/question) token
    overlap on the criterion's scale. Useful as a no-network baseline and for
    tests where stable, reproducible scores matter.
    """

    name = "mock"

    def judge(self, item: JudgeInput, criterion: Criterion) -> Verdict:
        content_tokens = _tokenize(item.content)
        if item.reference:
            basis = _tokenize(item.reference)
            rationale = "content/reference token overlap"
        elif item.question:
            basis = _tokenize(item.question)
            rationale = "content/question token overlap"
        else:
            basis = content_tokens
            rationale = "self-overlap (no reference/question)"

        if not content_tokens:
            normalized = 0.0
        else:
            normalized = _overlap_ratio(content_tokens, basis)

        raw = _denormalize(normalized, criterion.scale)
        return Verdict(
            criterion=criterion.name,
            raw_score=raw,
            normalized=round(normalized, 6),
            rationale=f"{rationale}: {normalized:.2f}",
        )


def _denormalize(normalized: float, scale: Scale) -> float:
    """Map a 0..1 normalized score back into the criterion's raw scale."""
    if scale.ordinal:
        # 0..1 → 1..max
        span = max(1, scale.max - 1)
        return round(1 + normalized * span)
    return round(scale.min + normalized * (scale.max - scale.min), 6)


# ── LLMJudge (optional) ─────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """You are an evaluator. Score the item on this criterion.

Criterion: {crit_name}
Description: {crit_desc}
Scale: {scale_desc}

Question: {question}
Reference (if any): {reference}
Item: {content}

Respond with ONLY a JSON object: {{"score": <number on the scale>, "rationale": "<short>"}}
"""


class LLMJudge:
    """An LLM-backed judge. Lazy-imports `openai`; needs an API key at runtime."""

    name = "llm"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        client=None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model
        self._client = client
        self._api_key = api_key
        self._base_url = base_url

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            openai = importlib.import_module("openai")  # type: ignore
        except ImportError as e:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "LLMJudge requires the 'openai' package. Install with: pip install 'judgekit[judge]'"
            ) from e
        key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not key:  # pragma: no cover - env-dependent
            raise RuntimeError("OPENAI_API_KEY is required for LLMJudge")
        kwargs = {"api_key": key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = openai.OpenAI(**kwargs)  # type: ignore[attr-defined]
        return self._client

    def judge(self, item: JudgeInput, criterion: Criterion) -> Verdict:  # pragma: no cover - network
        client = self._get_client()
        scale_desc = (
            f"integer 1..{int(criterion.scale.max)} (Likert)"
            if criterion.scale.ordinal
            else f"float {criterion.scale.min}..{criterion.scale.max}"
        )
        prompt = _PROMPT_TEMPLATE.format(
            crit_name=criterion.name,
            crit_desc=criterion.description,
            scale_desc=scale_desc,
            question=item.question or "(none)",
            reference=item.reference or "(none)",
            content=item.content,
        )
        resp = client.chat.completions.create(  # type: ignore[union-attr]
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
        score, rationale = _parse_llm_verdict(text)
        normalized = criterion.scale.normalize(score)
        return Verdict(
            criterion=criterion.name,
            raw_score=score,
            normalized=normalized,
            rationale=rationale,
        )


def _parse_llm_verdict(text: str) -> tuple[float, str]:
    m = re.search(r"\{[^{}]*\}", text)
    if not m:
        return 0.0, f"unparseable: {text[:120]}"
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return 0.0, f"unparseable: {text[:120]}"
    return float(d.get("score", 0.0)), str(d.get("rationale", ""))


def get_judge(kind: str = "mock", **kwargs) -> JudgeBackend:
    """Factory: 'mock' -> MockJudge, 'llm' -> LLMJudge."""
    kind = kind.lower()
    if kind == "mock":
        return MockJudge()
    if kind == "llm":
        return LLMJudge(**kwargs)
    raise ValueError(f"unknown judge kind: {kind!r} (use 'mock' or 'llm')")
