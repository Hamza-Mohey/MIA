"""Subprocess boundaries for memory-safe local model inference."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from modules.config import MODEL_WORKER_TIMEOUT_SECONDS, TEXT_WORKER_TIMEOUT_SECONDS
from modules.speaker_diarization import (
    assign_speakers_to_segments,
    format_speaker_transcript,
)


def run_model_worker(stage: str, payload: dict[str, Any]) -> Any:
    """Run exactly one heavyweight local model stage and return its JSON result."""
    request_path: Path | None = None
    response_path: Path | None = None
    worker_started = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", encoding="utf-8", delete=False
        ) as request:
            json.dump({"stage": stage, "payload": payload}, request, ensure_ascii=False)
            request_path = Path(request.name)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as response:
            response_path = Path(response.name)

        command = [
            sys.executable,
            "-m",
            "modules.model_worker",
            "--request",
            str(request_path),
            "--response",
            str(response_path),
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "PYTHONUNBUFFERED": "1",
            }
        )
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        timeout_seconds = (
            TEXT_WORKER_TIMEOUT_SECONDS
            if stage in {"analyse", "analyse_chunks", "answer"}
            else MODEL_WORKER_TIMEOUT_SECONDS
        )
        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parent.parent),
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            creationflags=flags,
            check=False,
        )
        if not response_path.exists() or not response_path.stat().st_size:
            detail = completed.stderr.strip() or completed.stdout.strip() or "No worker response"
            raise RuntimeError(f"{stage} worker exited without a result: {detail}")
        response = json.loads(response_path.read_text(encoding="utf-8"))
        if not response.get("ok"):
            raise RuntimeError(
                f"{stage} worker failed ({response.get('error_type', 'Error')}): "
                f"{response.get('error', 'Unknown worker error')}"
            )
        if completed.returncode != 0:
            raise RuntimeError(f"{stage} worker exited with code {completed.returncode}")
        result = response.get("result")
        if isinstance(result, dict):
            result.setdefault("_runtime", {})["subprocess_seconds"] = round(
                time.perf_counter() - worker_started, 3
            )
        return result
    except subprocess.TimeoutExpired as exc:
        timeout_seconds = (
            TEXT_WORKER_TIMEOUT_SECONDS
            if stage in {"analyse", "analyse_chunks", "answer"}
            else MODEL_WORKER_TIMEOUT_SECONDS
        )
        raise RuntimeError(
            f"{stage} worker exceeded the {timeout_seconds}-second timeout."
        ) from exc
    finally:
        for path in (request_path, response_path):
            if path is not None:
                path.unlink(missing_ok=True)


def transcribe_audio_isolated(audio_path: str | Path) -> dict[str, Any]:
    return run_model_worker("transcribe", {"audio_path": str(Path(audio_path).resolve())})


def diarize_audio_isolated(
    audio_path: str | Path,
    *,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> dict[str, Any]:
    return run_model_worker(
        "diarize",
        {
            "audio_path": str(Path(audio_path).resolve()),
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        },
    )


def combine_transcription_and_diarization(
    transcription: dict[str, Any], diarization: dict[str, Any]
) -> dict[str, Any]:
    """Align cached stage outputs without rerunning either model."""
    combined = deepcopy(transcription)
    attributed = assign_speakers_to_segments(
        combined.get("segments", []), diarization.get("exclusive_turns", [])
    )
    combined["plain_transcript"] = combined.get("transcript", "")
    combined["transcript"] = format_speaker_transcript(attributed) or combined["plain_transcript"]
    combined["speaker_segments"] = attributed
    combined["diarization"] = deepcopy(diarization)
    combined["diarization_error"] = None
    combined["models"] = [combined.get("model"), diarization.get("model")]
    return combined


def preserve_transcription_after_diarization_failure(
    transcription: dict[str, Any], error: Exception
) -> dict[str, Any]:
    preserved = deepcopy(transcription)
    preserved["plain_transcript"] = preserved.get("transcript", "")
    preserved["speaker_segments"] = []
    preserved["diarization"] = None
    preserved["diarization_error"] = str(error)
    preserved["models"] = [preserved.get("model")]
    preserved.setdefault("notes", []).append(
        "Speaker diarization was skipped because its isolated worker failed."
    )
    return preserved


def transcribe_audio_with_speakers_isolated(
    audio_path: str | Path,
    *,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> dict[str, Any]:
    resolved = str(Path(audio_path).resolve())
    transcription = transcribe_audio_isolated(resolved)
    try:
        diarization = diarize_audio_isolated(
            resolved,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
    except RuntimeError as exc:
        return preserve_transcription_after_diarization_failure(transcription, exc)
    return combine_transcription_and_diarization(transcription, diarization)


def analyse_transcript_isolated(
    transcript: str,
    *,
    transcript_quality: float,
    quality_basis: str,
) -> dict[str, Any]:
    return run_model_worker(
        "analyse",
        {
            "transcript": transcript,
            "transcript_quality": transcript_quality,
            "quality_basis": quality_basis,
        },
    )


def analyse_transcript_chunks_isolated(
    chunks: list[str],
    *,
    transcript_quality: float,
    quality_basis: str,
) -> dict[str, Any]:
    return run_model_worker(
        "analyse_chunks",
        {
            "chunks": chunks,
            "transcript_quality": transcript_quality,
            "quality_basis": quality_basis,
        },
    )


def answer_meeting_question_isolated(evidence_text: str, question: str) -> dict[str, Any]:
    return run_model_worker("answer", {"evidence_text": evidence_text, "question": question})
