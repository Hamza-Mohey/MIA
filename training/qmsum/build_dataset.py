"""Prepare the official QMSum files without changing their split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "datasets" / "qmsum" / "QMSum-main" / "data"


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _transcript(turns: list[dict[str, Any]], ranges: list[list[str]] | None = None) -> str:
    selected: list[dict[str, Any]] = []
    if ranges:
        seen: set[int] = set()
        for first, last in ranges:
            for index in range(max(0, int(first)), min(len(turns), int(last) + 1)):
                if index not in seen:
                    selected.append(turns[index])
                    seen.add(index)
    else:
        selected = turns
    return "\n".join(
        f"{str(turn.get('speaker', 'Unknown')).strip()}: "
        f"{' '.join(str(turn.get('content', '')).split())}"
        for turn in selected
        if str(turn.get("content", "")).strip()
    )


def _meeting_domain(meeting_id: str) -> str:
    """Infer the official QMSum domain from its stable meeting ID scheme."""
    if meeting_id.startswith(("covid_", "education_")):
        return "committee"
    if meeting_id.startswith(("ES", "IS", "TS")):
        return "product"
    return "academic"


def build(source: Path, output: Path) -> dict[str, Any]:
    meetings: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    topics: list[dict[str, Any]] = []
    ids_by_split: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    for source_split, split in (("train", "train"), ("val", "validation"), ("test", "test")):
        for path in sorted((source / "ALL" / source_split).glob("*.json")):
            meeting_id = path.stem
            raw = json.loads(path.read_text(encoding="utf-8"))
            turns = raw.get("meeting_transcripts", [])
            full_transcript = _transcript(turns)
            general = raw.get("general_query_list") or {}
            if isinstance(general, list):
                general = general[0] if general else {}
            domain = _meeting_domain(meeting_id)
            ids_by_split[split].add(meeting_id)
            meetings.append(
                {
                    "meeting_id": meeting_id,
                    "split": split,
                    "domain": domain,
                    "turn_count": len(turns),
                    "transcript": full_transcript,
                    "query": general.get("query", "Summarize the meeting."),
                    "answer": general.get("answer", ""),
                    "source": "QMSum official general query",
                }
            )
            for index, item in enumerate(raw.get("specific_query_list", [])):
                ranges = item.get("relevant_text_span", [])
                queries.append(
                    {
                        "example_id": f"{meeting_id}.query{index}",
                        "meeting_id": meeting_id,
                        "split": split,
                        "domain": domain,
                        "query": item.get("query", ""),
                        "answer": item.get("answer", ""),
                        "relevant_text_span": ranges,
                        "relevant_transcript": _transcript(turns, ranges),
                        "source": "QMSum official specific query",
                    }
                )
            for index, item in enumerate(raw.get("topic_list", [])):
                ranges = item.get("relevant_text_span", [])
                topics.append(
                    {
                        "example_id": f"{meeting_id}.topic{index}",
                        "meeting_id": meeting_id,
                        "split": split,
                        "domain": domain,
                        "topic": item.get("topic", ""),
                        "relevant_text_span": ranges,
                        "relevant_transcript": _transcript(turns, ranges),
                    }
                )
    overlap = {
        f"{left}_{right}": sorted(ids_by_split[left] & ids_by_split[right])
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    counts = {
        "general_meetings.jsonl": _write_jsonl(output / "general_meetings.jsonl", meetings),
        "specific_queries.jsonl": _write_jsonl(output / "specific_queries.jsonl", queries),
        "topic_spans.jsonl": _write_jsonl(output / "topic_spans.jsonl", topics),
    }
    audit = {
        "meeting_count": len(meetings),
        "specific_query_count": len(queries),
        "topic_count": len(topics),
        "split_counts": dict(Counter(item["split"] for item in meetings)),
        "domain_counts": dict(Counter(item["domain"] for item in meetings)),
        "split_overlap": overlap,
        "generated_records": counts,
        "licence": "MIT (repository LICENSE); retain upstream attribution and citations.",
        "safety_note": (
            "Use QMSum's official split for text experiments. Do not merge its AMI meeting IDs "
            "with the separate local AMI split, because the partitions differ."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "generated")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(build(args.source.resolve(), args.output.resolve()), indent=2))
