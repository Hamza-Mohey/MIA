"""Local speaker diarization and timestamp reconciliation.

The heavy, optional pyannote dependency is imported only when diarization is
requested. Pure alignment helpers remain dependency-free and unit-testable.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from modules.config import DIARIZATION_MODEL_NAME, HF_CACHE_DIR, cached_model_snapshot

_pipeline = None


def _timestamp(value: Any) -> tuple[float | None, float | None]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        start = float(value[0]) if value[0] is not None else None
        end = float(value[1]) if value[1] is not None else None
        return start, end
    return None, None


def load_diarization_model():
    """Load Community-1 once and move it to CUDA when available."""
    global _pipeline
    if _pipeline is None:
        try:
            import torch

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"(?s).*torchcodec is not installed correctly.*",
                    category=UserWarning,
                )
                from pyannote.audio import Pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Speaker diarization is not installed. Install requirements.txt "
                "in the active environment."
            ) from exc

        # Prefer the completed project-local snapshot. This prevents a Hub
        # metadata/download check every time the Streamlit process restarts.
        token = os.getenv("HF_TOKEN") or None
        checkpoint = cached_model_snapshot(DIARIZATION_MODEL_NAME) or DIARIZATION_MODEL_NAME
        try:
            _pipeline = Pipeline.from_pretrained(
                checkpoint,
                token=token,
                cache_dir=HF_CACHE_DIR,
            )
        except Exception as exc:
            raise RuntimeError(
                "The diarization model could not be loaded. Sign in with `hf auth login` and accept "
                f"the access conditions at https://huggingface.co/{DIARIZATION_MODEL_NAME}. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        if torch.cuda.is_available():
            _pipeline.to(torch.device("cuda"))
    return _pipeline


def _annotation_turns(annotation: Any) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        turns.append(
            {
                "start": round(float(segment.start), 3),
                "end": round(float(segment.end), 3),
                "speaker": str(speaker),
            }
        )
    return turns


def load_audio_in_memory(audio_path: str | Path) -> dict[str, Any]:
    """Load audio without pyannote's optional torchcodec file decoder.

    Pyannote accepts an in-memory waveform directly. This is also more robust
    on Windows, where torchcodec requires a matching shared-library FFmpeg
    build. Formats unsupported by libsndfile are normalized through FFmpeg.
    """
    audio_path = Path(audio_path)
    import soundfile as sf
    import torch

    converted_path: Path | None = None
    try:
        try:
            waveform, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        except (RuntimeError, sf.LibsndfileError):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
                converted_path = Path(temporary.name)
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        str(audio_path),
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        str(converted_path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                waveform, sample_rate = sf.read(
                    str(converted_path), dtype="float32", always_2d=True
                )
            except (FileNotFoundError, subprocess.CalledProcessError) as exc:
                detail = getattr(exc, "stderr", None) or str(exc)
                raise RuntimeError(
                    "This audio format needs FFmpeg conversion before diarization, "
                    f"but conversion failed: {detail}"
                ) from exc

        return {
            "waveform": torch.from_numpy(waveform.T.copy()),
            "sample_rate": int(sample_rate),
        }
    finally:
        if converted_path is not None:
            converted_path.unlink(missing_ok=True)


def diarize_audio(
    audio_path: str | Path,
    *,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> dict[str, Any]:
    """Return ordinary and exclusive speaker turns for a local audio file."""
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    kwargs = {
        key: value
        for key, value in {
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        }.items()
        if value is not None
    }
    total_started = time.perf_counter()
    load_started = time.perf_counter()
    pipeline = load_diarization_model()
    model_load_seconds = time.perf_counter() - load_started
    preprocess_started = time.perf_counter()
    audio = load_audio_in_memory(path)
    preprocess_seconds = time.perf_counter() - preprocess_started
    inference_started = time.perf_counter()
    output = pipeline(audio, **kwargs)
    inference_seconds = time.perf_counter() - inference_started
    postprocess_started = time.perf_counter()
    ordinary = _annotation_turns(output.speaker_diarization)
    exclusive_annotation = getattr(output, "exclusive_speaker_diarization", None)
    exclusive = (
        _annotation_turns(exclusive_annotation) if exclusive_annotation is not None else ordinary
    )
    speakers = sorted({turn["speaker"] for turn in ordinary})
    postprocess_seconds = time.perf_counter() - postprocess_started
    return {
        "model": DIARIZATION_MODEL_NAME,
        "speaker_count": len(speakers),
        "speakers": speakers,
        "turns": ordinary,
        "exclusive_turns": exclusive,
        "notes": [
            "Speaker labels are anonymous within each recording until a user names them.",
            "Overlapping speech may reduce both diarization and transcription accuracy.",
        ],
        "_runtime": {
            "model_load_seconds": round(model_load_seconds, 3),
            "preprocess_seconds": round(preprocess_seconds, 3),
            "inference_seconds": round(inference_seconds, 3),
            "postprocess_seconds": round(postprocess_seconds, 3),
            "total_seconds": round(time.perf_counter() - total_started, 3),
        },
    }


def _overlap(start: float, end: float, turn: dict[str, Any]) -> float:
    return max(0.0, min(end, float(turn["end"])) - max(start, float(turn["start"])))


def assign_speakers_to_segments(
    transcription_segments: Iterable[dict[str, Any]],
    speaker_turns: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign each ASR segment to the speaker with greatest temporal overlap."""
    turns = list(speaker_turns)
    attributed: list[dict[str, Any]] = []
    for index, segment in enumerate(transcription_segments):
        start, end = _timestamp(segment.get("timestamp"))
        if start is None:
            start = float(segment.get("start", 0.0))
        if end is None:
            end = float(segment.get("end", start))
        if end < start:
            start, end = end, start
        scores: dict[str, float] = {}
        for turn in turns:
            speaker = str(turn["speaker"])
            scores[speaker] = scores.get(speaker, 0.0) + _overlap(start, end, turn)
        speaker = (
            max(scores, key=scores.get)
            if scores and max(scores.values()) > 0
            else "SPEAKER_UNKNOWN"
        )
        attributed.append(
            {
                "index": index,
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": speaker,
                "text": " ".join(str(segment.get("text", "")).split()),
                "speaker_overlap_seconds": round(scores.get(speaker, 0.0), 3),
            }
        )
    return attributed


def format_speaker_transcript(segments: Iterable[dict[str, Any]]) -> str:
    """Format attributed ASR segments for Qwen and the evidence-oriented UI."""
    lines: list[str] = []
    for segment in segments:
        start = float(segment.get("start", 0.0))
        minutes, seconds = divmod(start, 60)
        hours, minutes = divmod(int(minutes), 60)
        lines.append(
            f"[{hours:02d}:{minutes:02d}:{seconds:05.2f}] "
            f"{segment.get('speaker', 'SPEAKER_UNKNOWN')}: {segment.get('text', '')}"
        )
    return "\n".join(lines)
