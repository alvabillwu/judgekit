"""Rubric file loading — JSON and a minimal YAML subset.

To stay zero-dependency, this module ships a tiny hand-rolled parser for the
YAML subset that rubrics actually use (nested mappings, lists, scalars,
quoted strings). It also accepts plain JSON (a YAML subset). For full YAML,
install pyyaml and pass a parsed dict to `rubric_from_dict`.

Supported file shape:

    name: qa-quality
    criteria:
      - name: relevance
        description: Does it answer the question?
        weight: 2.0
        scale: {max: 5, ordinal: true}
      - name: fluency
        description: Is it well-written?

`scale` is optional (defaults to continuous 0..1). `weight` defaults to 1.0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import Rubric, Criterion, Scale


def rubric_from_dict(data: dict) -> Rubric:
    """Build a Rubric from a plain dict (already parsed)."""
    name = data.get("name") or "rubric"
    raw_criteria = data.get("criteria")
    if not isinstance(raw_criteria, list) or not raw_criteria:
        raise ValueError("rubric must have a non-empty 'criteria' list")
    criteria: list[Criterion] = []
    for i, c in enumerate(raw_criteria):
        if not isinstance(c, dict):
            raise ValueError(f"criterion #{i} must be a mapping")
        cname = c.get("name")
        if not cname:
            raise ValueError(f"criterion #{i} is missing 'name'")
        scale = _scale_from_dict(c.get("scale"))
        criteria.append(
            Criterion(
                name=str(cname),
                description=str(c.get("description", "")),
                weight=float(c.get("weight", 1.0)),
                scale=scale,
            )
        )
    return Rubric(name=str(name), criteria=tuple(criteria))


def _scale_from_dict(data: Any) -> Scale:
    if data is None:
        return Scale()
    if not isinstance(data, dict):
        raise ValueError("scale must be a mapping")
    return Scale(
        max=float(data.get("max", 1.0)),
        min=float(data.get("min", 0.0)),
        ordinal=bool(data.get("ordinal", False)),
    )


def load_rubric(path: str | Path) -> Rubric:
    """Load a rubric from a .json/.yaml/.yml file.

    JSON files use stdlib json. YAML files use the minimal parser in this
    module (sufficient for rubric files). For arbitrary YAML, parse with
    pyyaml and call `rubric_from_dict`.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        return rubric_from_dict(json.loads(text))
    # YAML path (or unknown extension) — try the minimal parser.
    return rubric_from_dict(parse_yaml(text))


# ── minimal YAML parser (rubric subset) ─────────────────────────────────────
# Supports: nested mappings (2-space indent), block sequences (- item),
# inline flow mappings ({k: v, k: v}), inline flow sequences ([a, b]),
# quoted strings, ints, floats, bools, null. NOT a general YAML parser.


def parse_yaml(text: str) -> dict:
    """Parse the rubric YAML subset into a dict. Raises on unsupported input."""
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError("empty rubric file")
    value, _ = _parse_block(lines, 0, 0)
    if not isinstance(value, dict):
        raise ValueError("rubric root must be a mapping")
    return value


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _parse_block(lines: list[str], i: int, min_indent: int) -> tuple[Any, int]:
    """Parse a block starting at lines[i] with indentation >= min_indent."""
    if i >= len(lines):
        return None, i
    indent = _indent(lines[i])
    if indent < min_indent:
        return None, i
    stripped = lines[i].lstrip()
    if stripped.startswith("- "):
        return _parse_seq(lines, i, indent)
    return _parse_map(lines, i, indent)


def _parse_map(lines: list[str], i: int, indent: int) -> tuple[dict, int]:
    result: dict = {}
    while i < len(lines):
        if _indent(lines[i]) < indent:
            break
        if _indent(lines[i]) > indent:
            # Shouldn't happen at map level; skip defensively.
            i += 1
            continue
        stripped = lines[i].lstrip()
        if stripped.startswith("- "):
            break
        key, sep, rest = stripped.partition(":")
        if not sep:
            raise ValueError(f"invalid mapping line: {lines[i]!r}")
        key = key.strip()
        rest = rest.strip()
        i += 1
        if rest == "":
            # Nested block on following lines (deeper indent).
            if i < len(lines) and _indent(lines[i]) > indent:
                value, i = _parse_block(lines, i, _indent(lines[i]))
            else:
                value = None
        else:
            value = _parse_scalar(rest)
        result[key] = value
    return result, i


def _parse_seq(lines: list[str], i: int, indent: int) -> tuple[list, int]:
    result: list = []
    while i < len(lines):
        if _indent(lines[i]) < indent:
            break
        if _indent(lines[i]) > indent:
            i += 1
            continue
        stripped = lines[i].lstrip()
        if not stripped.startswith("- "):
            break
        item_text = stripped[2:].strip()
        i += 1
        if item_text == "":
            # Nested block under the dash.
            if i < len(lines) and _indent(lines[i]) > indent:
                value, i = _parse_block(lines, i, _indent(lines[i]))
            else:
                value = None
        elif ":" in item_text and not item_text.startswith(("[", "{")):
            # Inline first key of a mapping item, e.g. "- name: relevance"
            # Rebuild a pseudo map starting with this line at indent+2.
            pseudo_indent = indent + 2
            pseudo_lines = [" " * pseudo_indent + item_text]
            while i < len(lines) and _indent(lines[i]) > indent and not lines[i].lstrip().startswith("- "):
                pseudo_lines.append(lines[i])
                i += 1
            value, _ = _parse_map(pseudo_lines, 0, pseudo_indent)
        else:
            value = _parse_scalar(item_text)
        result.append(value)
    return result, i


def _parse_scalar(text: str) -> Any:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return _parse_flow_map(text[1:-1])
    if text.startswith("[") and text.endswith("]"):
        return _parse_flow_seq(text[1:-1])
    # Quoted strings
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    low = text.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none", "~", ""):
        return None
    # Numbers
    try:
        if "." in text or "e" in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _parse_flow_map(inner: str) -> dict:
    out: dict = {}
    for part in _split_flow(inner):
        k, sep, v = part.partition(":")
        if not sep:
            continue
        out[k.strip()] = _parse_scalar(v.strip())
    return out


def _parse_flow_seq(inner: str) -> list:
    return [_parse_scalar(p.strip()) for p in _split_flow(inner)]


def _split_flow(s: str) -> list[str]:
    """Split a flow collection body on top-level commas (respecting braces)."""
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in s:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur))
    return parts
