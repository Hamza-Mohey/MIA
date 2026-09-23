"""Text-similarity and structured-extraction evaluation metrics."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any


def normalise(value: Any) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", str(value).lower()).split())


def similarity(left: Any, right: Any) -> float:
    a, b = normalise(left), normalise(right)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    a_tokens, b_tokens = set(a.split()), set(b.split())
    jaccard = len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))
    return max(jaccard, SequenceMatcher(None, a, b).ratio())


def _text(item: Any) -> str:
    if isinstance(item, dict):
        return str(
            item.get("text")
            or item.get("task")
            or item.get("name")
            or item.get("title")
            or item.get("description")
            or ""
        )
    return str(item)


def extraction_metrics(
    gold: list[Any], predicted: list[Any], fuzzy: bool = False, threshold: float = 0.82
) -> dict:
    gold_text = [_text(item) for item in gold if _text(item)]
    predicted_text = [_text(item) for item in predicted if _text(item)]
    if not gold_text and not predicted_text:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "matches": 0, "gold": 0, "predicted": 0}
    matched_gold = set()
    matches = 0
    for prediction in predicted_text:
        candidates = [
            (similarity(prediction, expected), index)
            for index, expected in enumerate(gold_text)
            if index not in matched_gold
        ]
        score, index = max(candidates, default=(0.0, -1))
        required = threshold if fuzzy else 1.0
        if score >= required:
            matched_gold.add(index)
            matches += 1
    precision = matches / len(predicted_text) if predicted_text else 0.0
    recall = matches / len(gold_text) if gold_text else (1.0 if not predicted_text else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
        "gold": len(gold_text),
        "predicted": len(predicted_text),
    }
