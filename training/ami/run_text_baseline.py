"""Run the current transcript model over an AMI split and save predictions.

The output is written after every meeting so an interrupted GPU run can be
resumed. Use ``evaluate_text_outputs`` to score the resulting JSONL file.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--meetings", type=Path, default=root / "generated" / "meeting_records.jsonl"
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=root / "runs" / "qwen_baseline_validation.jsonl"
    )
    parser.add_argument(
        "--restart", action="store_true", help="Discard saved predictions and run again."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meetings = [record for record in read_jsonl(args.meetings) if record["split"] == args.split]
    if args.limit:
        meetings = meetings[: args.limit]
    saved = [] if args.restart else read_jsonl(args.output)
    completed = {record["meeting_id"] for record in saved}
    from modules.config import MAX_TRANSCRIPT_LENGTH, TEXT_MODEL_NAME
    from modules.text_analysis import analyse_transcript

    for index, meeting in enumerate(meetings, start=1):
        meeting_id = meeting["meeting_id"]
        if meeting_id in completed:
            print(f"[{index}/{len(meetings)}] {meeting_id}: already saved", flush=True)
            continue
        transcript = meeting["transcript_text"]
        print(
            f"[{index}/{len(meetings)}] {meeting_id}: analysing {len(transcript):,} characters",
            flush=True,
        )
        started = time.perf_counter()
        prediction = analyse_transcript(transcript)
        saved.append(
            {
                "meeting_id": meeting_id,
                "split": args.split,
                "model": TEXT_MODEL_NAME,
                "input_characters": len(transcript),
                "input_was_truncated": len(transcript) > MAX_TRANSCRIPT_LENGTH,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "prediction": prediction,
            }
        )
        write_jsonl(args.output, saved)
        print(f"[{index}/{len(meetings)}] {meeting_id}: saved", flush=True)
    print(f"Wrote {len(saved)} prediction(s) to {args.output}")


if __name__ == "__main__":
    main()
