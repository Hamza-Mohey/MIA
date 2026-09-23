"""One-shot model worker used to guarantee RAM/VRAM reclamation on exit."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any


def _execute(request: dict[str, Any]) -> Any:
    stage = request.get("stage")
    payload = request.get("payload", {})
    if stage == "transcribe":
        from modules.audio_transcription import transcribe_audio

        return transcribe_audio(payload["audio_path"])
    if stage == "diarize":
        from modules.speaker_diarization import diarize_audio

        return diarize_audio(
            payload["audio_path"],
            num_speakers=payload.get("num_speakers"),
            min_speakers=payload.get("min_speakers"),
            max_speakers=payload.get("max_speakers"),
        )
    if stage == "analyse":
        from modules.text_analysis import analyse_transcript

        return analyse_transcript(
            payload["transcript"],
            transcript_quality=float(payload["transcript_quality"]),
            quality_basis=payload["quality_basis"],
        )
    if stage == "analyse_chunks":
        from modules.text_analysis import analyse_transcript_chunks

        return analyse_transcript_chunks(
            payload["chunks"],
            transcript_quality=float(payload["transcript_quality"]),
            quality_basis=payload["quality_basis"],
        )
    if stage == "answer":
        from modules.text_analysis import answer_meeting_question

        return answer_meeting_question(payload["evidence_text"], payload["question"])
    raise ValueError(f"Unknown model worker stage: {stage!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()
    os.environ["MODEL_WORKER_ACTIVE"] = "1"
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        response = {"ok": True, "result": _execute(request)}
        code = 0
    except Exception as exc:
        response = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        code = 1
    args.response.write_text(
        json.dumps(response, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
