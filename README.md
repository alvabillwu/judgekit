# ⚖️ judgekit

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange)](#)

**A reusable LLM-as-judge harness — rubrics, pluggable judges, cheap proxy metrics, and calibration against ground truth.**

Define a rubric (weighted criteria on a scale), score items with a pluggable judge, and calibrate the judge against labeled data. Designed as a **library**: drop it into any eval pipeline (RAG, summarization, classification, safety) and swap judges — deterministic mock, cheap proxy, or a real LLM — without touching the rubric.

> Built incrementally as a medium-complexity project. Round 1 ships the core (rubrics + judges + proxies + calibration); a CLI and multi-judge ensembling are on the roadmap.

## Features

- 📋 **Rubrics** — weighted criteria, continuous or ordinal (Likert 1–5) scales, auto-normalized
- 🔌 **Pluggable judges** — `MockJudge` (deterministic heuristic), `LLMJudge` (optional `openai`), and proxy judges — all one `JudgeBackend` protocol
- ⚡ **Cheap proxies** — `LengthJudge`, `OverlapJudge`, `KeywordJudge` for fast no-network baselines
- 🎯 **Calibration** — accuracy/precision/recall/F1 + MAE + confusion matrix vs ground-truth labels
- 🚫 **Zero hard dependencies** — pure stdlib; optional `openai` for the LLM judge

## Quick Start

```bash
pip install judgekit
```

## Usage

### Define a rubric and score items

```python
from judgekit import Rubric, Criterion, Scale, JudgeInput, score_items, MockJudge

rubric = Rubric(
    name="qa-quality",
    criteria=(
        Criterion("relevance", "Does it answer the question?", weight=2.0),
        Criterion("fluency", "Is it well-written?", weight=1.0),
        Criterion("groundedness", "Is it supported by the reference?", weight=2.0,
                  scale=Scale(max=5, ordinal=True)),
    ),
)

items = [
    JudgeInput(id="q1", content="Paris is the capital of France.",
               question="What is the capital of France?",
               reference="The capital of France is Paris."),
]

result = score_items(rubric, items, MockJudge())
print(result.mean_score, result.pass_rate)
for s in result.scored:
    print(s.item_id, round(s.score, 3), [(v.criterion, round(v.normalized, 2)) for v in s.verdicts])
```

### Calibrate a judge against ground truth

```python
from judgekit import calibrate

truth = {"q1": 1.0, "q2": 0.2, "q3": 0.9}  # item_id -> 0..1 truth score
report = calibrate(result, truth, threshold=0.5)
print(report.as_dict())
# {'n': 3, 'accuracy': ..., 'precision': ..., 'recall': ..., 'f1': ..., 'mae': ..., 'confusion': {...}}
```

### Swap judges — same rubric

```python
from judgekit import OverlapJudge, LLMJudge

# Cheap proxy baseline:
score_items(rubric, items, OverlapJudge())

# Real LLM judge (needs OPENAI_API_KEY; pip install 'judgekit[judge]'):
score_items(rubric, items, LLMJudge(model="gpt-4o-mini"))
```

## How it works

1. **Rubric** = weighted `Criterion`s, each with a `Scale`.
2. `score_item` asks the `JudgeBackend` to score each criterion, normalizes each score to 0..1, and returns a weighted aggregate `ScoredItem`.
3. `calibrate` compares the judge's pass/fail (threshold) and continuous scores against ground-truth labels — accuracy/precision/recall/F1, MAE, confusion.

The rubric never knows which judge scored it, so you can A/B a proxy against an LLM judge on the identical rubric.

## CLI

```bash
# Score items against a rubric (see examples/qa.yaml + examples/items.jsonl)
judgekit score examples/qa.yaml examples/items.jsonl --judge mock
judgekit score examples/qa.yaml examples/items.jsonl --judge overlap --json

# Calibrate a judge against ground-truth labels
judgekit calibrate examples/qa.yaml examples/items.jsonl --truth truth.json --judge mock

# List available judges
judgekit judges
```

`items.jsonl` is one JSON object per line: `{"id", "content", "question"?, "reference"?}`.
`truth.json` is `{"item_id": score_in_0..1}`.

## Rubric files (YAML or JSON)

```yaml
name: qa-quality
criteria:
  - name: relevance
    description: Does it answer the question?
    weight: 2.0
  - name: groundedness
    description: Is it supported by the reference?
    weight: 2.0
    scale: {max: 5, ordinal: true}   # Likert 1..5, auto-normalized
  - name: fluency
    description: Is it well-written?
    weight: 1.0
```

The YAML parser is a built-in minimal subset (no PyYAML dependency); plain JSON is also accepted.

## Roadmap

- [x] Rubrics + weighted aggregation (continuous + ordinal scales)
- [x] Mock + LLM judges + 3 proxy judges
- [x] Calibration vs ground truth
- [x] Rubric YAML/JSON file format
- [x] CLI: `judgekit score` / `calibrate` / `judges`
- [ ] Multi-judge ensembling (majority / mean / agreement-filtered)

## Development

```bash
git clone https://github.com/alvabillwu/judgekit.git
cd judgekit
pip install -e ".[dev]"
pytest -v
```

## License

MIT © [alvabillwu](https://github.com/alvabillwu)
