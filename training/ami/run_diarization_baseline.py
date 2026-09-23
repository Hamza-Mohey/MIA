"""Benchmark pyannote Community-1 against AMI's manual speaker segments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pyannote.core import Annotation, Segment
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate

from modules.speaker_diarization import diarize_audio

ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def as_annotation(meeting_id: str, turns: list[dict[str, Any]]) -> Annotation:
    annotation = Annotation(uri=meeting_id)
    for index, turn in enumerate(turns):
        annotation[Segment(float(turn["start"]), float(turn["end"])), index] = turn["speaker"]
    return annotation


def parse_args() -> argparse.Namespace:
    directory = Path(__file__).resolve().parent / "generated"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=directory / "diarization_manifest.jsonl")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / "diarization_baseline_validation.json",
    )
    parser.add_argument("--known-speaker-count", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [record for record in read_jsonl(args.manifest) if record["split"] == args.split]
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit(f"No {args.split} records found in {args.manifest}")
    reference_path = ROOT / records[0]["reference_rttm_path"]
    references = load_rttm(reference_path)
    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        meeting_id = record["meeting_id"]
        print(f"[{index}/{len(records)}] Diarizing {meeting_id}", flush=True)
        prediction = diarize_audio(
            ROOT / record["audio_path"],
            num_speakers=record["expected_speaker_count"] if args.known_speaker_count else None,
        )
        hypothesis = as_annotation(meeting_id, prediction["turns"])
        per_meeting_metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
        detail = per_meeting_metric(references[meeting_id], hypothesis, detailed=True)
        metric(references[meeting_id], hypothesis)
        rows.append(
            {
                "meeting_id": meeting_id,
                "reference_speakers": record["expected_speaker_count"],
                "predicted_speakers": prediction["speaker_count"],
                "diarization_error_rate": abs(per_meeting_metric),
                "detail": {key: float(value) for key, value in detail.items()},
            }
        )
    report = {
        "model": "pyannote/speaker-diarization-community-1",
        "split": args.split,
        "known_speaker_count": args.known_speaker_count,
        "meeting_count": len(rows),
        "aggregate_diarization_error_rate": abs(metric),
        "settings": {"collar_seconds": 0.0, "skip_overlap": False},
        "meetings": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
