"""Score saved transcript-extraction predictions against AMI gold outputs.

Prediction JSONL records must contain ``meeting_id`` and either a ``prediction``
object or the prediction fields at the top level. This separation lets the same
scorer compare a base model, a prompt revision, and an adapter without loading
model weights during metric tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.metrics import extraction_metrics, similarity

CATEGORY_MAP = {
    "decisions": "decisions",
    "actions": "action_items",
    "problems": "risks",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score_predictions(
    meeting_records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    gold_by_id = {
        record["meeting_id"]: record
        for record in meeting_records
        if split == "all" or record["split"] == split
    }
    prediction_by_id = {record["meeting_id"]: record for record in predictions}
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for meeting_id, meeting in sorted(gold_by_id.items()):
        saved = prediction_by_id.get(meeting_id)
        if saved is None:
            missing.append(meeting_id)
            continue
        prediction = saved.get("prediction", saved)
        summary_gold = " ".join(meeting["gold"]["meeting_summary_sentences"])
        categories: dict[str, Any] = {}
        for gold_key, prediction_key in CATEGORY_MAP.items():
            categories[gold_key] = extraction_metrics(
                meeting["gold"][gold_key], prediction.get(prediction_key, []), fuzzy=True
            )
        rows.append(
            {
                "meeting_id": meeting_id,
                "summary_similarity": similarity(
                    summary_gold, prediction.get("meeting_summary", "")
                ),
                "categories": categories,
            }
        )

    def average(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "split": split,
        "expected_meetings": len(gold_by_id),
        "scored_meetings": len(rows),
        "missing_predictions": missing,
        "summary_similarity": average([row["summary_similarity"] for row in rows]),
        "category_macro": {
            category: {
                metric: average([row["categories"][category][metric] for row in rows])
                for metric in ("precision", "recall", "f1")
            }
            for category in CATEGORY_MAP
        },
        "meetings": rows,
        "metric_note": (
            "Summary similarity is max(token Jaccard, character sequence ratio). "
            "Category matches use the project's fuzzy 0.82 threshold; human review remains required."
        ),
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument(
        "--meetings", type=Path, default=root / "generated" / "meeting_records.jsonl"
    )
    parser.add_argument(
        "--split", choices=("train", "validation", "test", "all"), default="validation"
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = score_predictions(
        _read_jsonl(args.meetings), _read_jsonl(args.predictions), args.split
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
