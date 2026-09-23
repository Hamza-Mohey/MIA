"""Whisper ASR adapter and safe temporary-upload handling."""

from __future__ import annotations

import os
import re
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from difflib import SequenceMatcher
from pathlib import Path

from modules.config import (
    ASR_TRANSCRIPT_QUALITY,
    FASTER_WHISPER_MODEL_DIR,
    WHISPER_AUTO_RETRY,
    WHISPER_BATCH_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_CRITICAL_WORDS_PER_MINUTE,
    WHISPER_MAX_COMPRESSION_RATIO,
    WHISPER_MAX_NO_SPEECH_PROBABILITY,
    WHISPER_MIN_AVERAGE_LOG_PROBABILITY,
    WHISPER_MIN_WORDS_PER_MINUTE,
    WHISPER_MODEL_NAME,
    WHISPER_RETRY_IMPROVEMENT_RATIO,
    WHISPER_RETRY_MIN_DURATION_SECONDS,
    WHISPER_VAD_MODE,
)

_asr_pipeline = None
_asr_compute_type = WHISPER_COMPUTE_TYPE
_torch_dll_handle = None
_WORD = re.compile(r"[\w']+")


def load_whisper_model():
    """Load the CTranslate2 Whisper Medium runtime once per worker process."""
    global _asr_pipeline, _asr_compute_type, _torch_dll_handle
    if _asr_pipeline is None:
        if not (FASTER_WHISPER_MODEL_DIR / "model.bin").exists():
            raise RuntimeError(
                "Faster-Whisper Medium is not cached. Run `python -m "
                "modules.prefetch_faster_whisper` once while online."
            )
        from faster_whisper import WhisperModel

        try:
            # The CUDA PyTorch wheel already contains the matching CUDA/cuDNN
            # DLLs. Importing it registers that DLL directory for CTranslate2
            # on Windows without requiring a second system-wide CUDA install.
            import torch

            torch_lib = Path(torch.__file__).resolve().parent / "lib"
            if os.name == "nt" and torch_lib.exists():
                _torch_dll_handle = os.add_dll_directory(str(torch_lib))
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            _asr_pipeline = WhisperModel(
                str(FASTER_WHISPER_MODEL_DIR),
                device="cuda",
                compute_type=WHISPER_COMPUTE_TYPE,
                local_files_only=True,
            )
            _asr_compute_type = WHISPER_COMPUTE_TYPE
        except RuntimeError:
            # Keep transcript-only development functional on machines without
            # a compatible CUDA runtime. CPU uses a memory-conscious INT8 path.
            _asr_pipeline = WhisperModel(
                str(FASTER_WHISPER_MODEL_DIR),
                device="cpu",
                compute_type="int8",
                local_files_only=True,
            )
            _asr_compute_type = "int8"
    return _asr_pipeline


@contextmanager
def temporary_audio_file(uploaded_file) -> Iterator[str]:
    """Write a Streamlit-like upload to disk and always remove it afterward."""
    suffix = Path(getattr(uploaded_file, "name", "meeting.wav")).suffix or ".wav"
    path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(uploaded_file.getvalue())
            path = handle.name
        yield path
    finally:
        if path:
            with suppress(FileNotFoundError):
                os.unlink(path)


def _materialise_transcription(
    engine, path: Path, transcribe_kwargs: dict[str, object]
) -> tuple[list[dict], object, float]:
    started = time.perf_counter()
    segments_iterator, information = engine.transcribe(str(path), **transcribe_kwargs)
    segments = [
        {
            "text": segment.text.strip(),
            "timestamp": (round(float(segment.start), 3), round(float(segment.end), 3)),
            "avg_logprob": round(float(getattr(segment, "avg_logprob", 0.0) or 0.0), 4),
            "no_speech_prob": round(
                float(getattr(segment, "no_speech_prob", 0.0) or 0.0), 4
            ),
            "compression_ratio": round(
                float(getattr(segment, "compression_ratio", 0.0) or 0.0), 4
            ),
        }
        for segment in segments_iterator
        if segment.text.strip()
    ]
    return segments, information, time.perf_counter() - started


def _normalised_segment_text(value: str) -> str:
    return " ".join(_WORD.findall(value.casefold()))


def _filter_unreliable_segments(
    segments: list[dict],
) -> tuple[list[dict], list[dict[str, object]]]:
    """Remove low-confidence and repetitive no-VAD hallucinations."""
    kept: list[dict] = []
    rejected: list[dict[str, object]] = []
    occurrences: dict[str, int] = {}
    for segment in segments:
        text_key = _normalised_segment_text(segment["text"])
        occurrences[text_key] = occurrences.get(text_key, 0) + 1
        reasons: list[str] = []
        average_logprob = float(segment.get("avg_logprob", 0.0))
        no_speech = float(segment.get("no_speech_prob", 0.0))
        compression = float(segment.get("compression_ratio", 0.0))
        if average_logprob < WHISPER_MIN_AVERAGE_LOG_PROBABILITY:
            reasons.append("low_log_probability")
        if (
            no_speech > WHISPER_MAX_NO_SPEECH_PROBABILITY
            and average_logprob < -0.55
        ):
            reasons.append("probable_silence")
        if compression > WHISPER_MAX_COMPRESSION_RATIO:
            reasons.append("repetitive_compression")
        if text_key and occurrences[text_key] > 2 and len(text_key.split()) >= 4:
            reasons.append("repeated_phrase")
        if reasons:
            rejected.append(
                {
                    "text": segment["text"],
                    "timestamp": segment["timestamp"],
                    "reasons": reasons,
                }
            )
        else:
            kept.append(segment)
    return kept, rejected


def _segment_overlap_ratio(left: dict, right: dict) -> float:
    left_start, left_end = (float(value) for value in left["timestamp"])
    right_start, right_end = (float(value) for value in right["timestamp"])
    overlap = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    return overlap / max(0.001, right_end - right_start)


def _segment_text_similarity(left: dict, right: dict) -> float:
    left_text = _normalised_segment_text(left["text"])
    right_text = _normalised_segment_text(right["text"])
    if not left_text or not right_text:
        return 0.0
    left_tokens = set(left_text.split())
    right_tokens = set(right_text.split())
    containment = len(left_tokens & right_tokens) / max(
        1, min(len(left_tokens), len(right_tokens))
    )
    return max(containment, SequenceMatcher(None, left_text, right_text).ratio())


def _segments_are_near_duplicates(left: dict, right: dict) -> bool:
    left_start, left_end = (float(value) for value in left["timestamp"])
    right_start, right_end = (float(value) for value in right["timestamp"])
    temporal_distance = max(0.0, max(left_start, right_start) - min(left_end, right_end))
    return temporal_distance <= 12.0 and _segment_text_similarity(left, right) >= 0.72


def _merge_transcription_passes(
    base_segments: list[dict], recovery_segments: list[dict]
) -> tuple[list[dict], int]:
    """Add only reliable no-VAD segments that fill gaps in the VAD transcript."""
    if not base_segments:
        return recovery_segments, len(recovery_segments)
    recovered: list[dict] = []
    base_text = {_normalised_segment_text(segment["text"]) for segment in base_segments}
    base_start = min(float(segment["timestamp"][0]) for segment in base_segments)
    base_end = max(float(segment["timestamp"][1]) for segment in base_segments)
    for segment in recovery_segments:
        text_key = _normalised_segment_text(segment["text"])
        if not text_key or text_key in base_text:
            continue
        segment_start = float(segment["timestamp"][0])
        segment_end = float(segment["timestamp"][1])
        # Hybrid mode fills internal VAD gaps. It does not extend into an
        # unbounded leading/trailing region where no-VAD hallucinations cluster.
        if len(base_segments) >= 2 and (
            segment_end < base_start - 5.0 or segment_start > base_end + 5.0
        ):
            continue
        if any(_segment_overlap_ratio(existing, segment) >= 0.25 for existing in base_segments):
            continue
        if any(_segments_are_near_duplicates(existing, segment) for existing in base_segments):
            continue
        if any(_segments_are_near_duplicates(existing, segment) for existing in recovered):
            continue
        recovered.append({**segment, "_merge_source": "recovery"})
    candidates = sorted(
        [*({**segment, "_merge_source": "vad"} for segment in base_segments), *recovered],
        key=lambda segment: (float(segment["timestamp"][0]), float(segment["timestamp"][1])),
    )
    combined: list[dict] = []
    for candidate in candidates:
        duplicate_index = next(
            (
                index
                for index in range(max(0, len(combined) - 4), len(combined))
                if _segments_are_near_duplicates(combined[index], candidate)
            ),
            None,
        )
        if duplicate_index is None:
            combined.append(candidate)
            continue
        existing = combined[duplicate_index]
        if len(_WORD.findall(candidate["text"])) > len(_WORD.findall(existing["text"])):
            combined[duplicate_index] = candidate
    retained_recovery = sum(
        segment.get("_merge_source") == "recovery" for segment in combined
    )
    for segment in combined:
        segment.pop("_merge_source", None)
    return combined, retained_recovery


def _coverage_diagnostics(segments: list[dict], information: object) -> dict[str, object]:
    duration = float(getattr(information, "duration", 0.0) or 0.0)
    if duration <= 0 and segments:
        duration = max(float(segment["timestamp"][1]) for segment in segments)
    word_count = sum(len(_WORD.findall(segment["text"])) for segment in segments)
    words_per_minute = word_count / max(duration / 60.0, 1 / 60.0)
    covered_seconds = sum(
        max(0.0, float(segment["timestamp"][1]) - float(segment["timestamp"][0]))
        for segment in segments
    )
    long_recording = duration >= WHISPER_RETRY_MIN_DURATION_SECONDS
    low_coverage = long_recording and words_per_minute < WHISPER_MIN_WORDS_PER_MINUTE
    critical_coverage = (
        long_recording and words_per_minute < WHISPER_CRITICAL_WORDS_PER_MINUTE
    )
    return {
        "duration_seconds": round(duration, 3),
        "word_count": word_count,
        "words_per_minute": round(words_per_minute, 1),
        "transcribed_audio_ratio": round(min(1.0, covered_seconds / max(duration, 0.001)), 3),
        "low_coverage": low_coverage,
        "critical_coverage": critical_coverage,
        "coverage_status": "critical" if critical_coverage else "review" if low_coverage else "good",
        "minimum_words_per_minute": WHISPER_MIN_WORDS_PER_MINUTE,
        "critical_words_per_minute": WHISPER_CRITICAL_WORDS_PER_MINUTE,
    }


def transcribe_audio(audio_path: str | Path) -> dict:
    """Transcribe meeting audio, retry suspicious VAD output, and report coverage."""
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    total_started = time.perf_counter()
    try:
        load_started = time.perf_counter()
        model = load_whisper_model()
        model_load_seconds = time.perf_counter() - load_started
        engine = model
        transcribe_kwargs: dict[str, object] = {
            "beam_size": 5,
            "vad_filter": WHISPER_VAD_MODE != "off",
            "condition_on_previous_text": True,
        }
        if WHISPER_BATCH_SIZE > 1:
            from faster_whisper import BatchedInferencePipeline

            engine = BatchedInferencePipeline(model=model)
            transcribe_kwargs["batch_size"] = WHISPER_BATCH_SIZE

        raw_segments, information, inference_seconds = _materialise_transcription(
            engine, path, transcribe_kwargs
        )
        segments, rejected_segments = _filter_unreliable_segments(raw_segments)
        diagnostics = _coverage_diagnostics(segments, information)
        selected_strategy = "vad" if transcribe_kwargs["vad_filter"] else "no_vad"
        attempts = [
            {
                "vad_filter": bool(transcribe_kwargs["vad_filter"]),
                "word_count": diagnostics["word_count"],
                "words_per_minute": diagnostics["words_per_minute"],
                "rejected_segments": len(rejected_segments),
                "inference_seconds": round(inference_seconds, 3),
                "selected": True,
            }
        ]
        should_retry = (
            WHISPER_VAD_MODE == "auto"
            and WHISPER_AUTO_RETRY
            and bool(diagnostics["low_coverage"])
        )
        if should_retry:
            retry_kwargs = {**transcribe_kwargs, "vad_filter": False}
            raw_retry_segments, retry_information, retry_seconds = _materialise_transcription(
                engine, path, retry_kwargs
            )
            retry_segments, retry_rejected = _filter_unreliable_segments(raw_retry_segments)
            hybrid_segments, recovered_count = _merge_transcription_passes(
                segments, retry_segments
            )
            hybrid_diagnostics = _coverage_diagnostics(hybrid_segments, retry_information)
            retry_is_better = int(hybrid_diagnostics["word_count"]) >= max(
                int(diagnostics["word_count"]) + 1,
                int(float(diagnostics["word_count"]) * WHISPER_RETRY_IMPROVEMENT_RATIO),
            )
            attempts[0]["selected"] = not retry_is_better
            attempts.append(
                {
                    "vad_filter": False,
                    "strategy": "hybrid_recovery",
                    "word_count": hybrid_diagnostics["word_count"],
                    "words_per_minute": hybrid_diagnostics["words_per_minute"],
                    "recovered_segments": recovered_count,
                    "rejected_segments": len(retry_rejected),
                    "inference_seconds": round(retry_seconds, 3),
                    "selected": retry_is_better,
                }
            )
            inference_seconds += retry_seconds
            if retry_is_better:
                segments = hybrid_segments
                information = retry_information
                diagnostics = hybrid_diagnostics
                rejected_segments = [*rejected_segments, *retry_rejected]
                selected_strategy = "hybrid"
        postprocess_started = time.perf_counter()
        transcript = " ".join(segment["text"] for segment in segments).strip()
        postprocess_seconds = time.perf_counter() - postprocess_started
    except Exception as exc:
        raise RuntimeError(
            "Audio transcription failed. Check that FFmpeg supports this format and that the file is valid. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    notes = [
        "Whisper transcription is evidence, not ground truth.",
        "Noise, accents, and overlapping speakers can reduce downstream extraction quality.",
    ]
    if diagnostics["low_coverage"]:
        notes.append(
            "Low transcript coverage was detected. Review the transcript before analysis; "
            "quiet or overlapping speech may be missing."
        )
    quality_score = ASR_TRANSCRIPT_QUALITY
    if diagnostics["critical_coverage"]:
        quality_score = min(quality_score, 0.50)
    elif diagnostics["low_coverage"]:
        quality_score = min(quality_score, 0.65)
    return {
        "model": WHISPER_MODEL_NAME,
        "engine": "faster-whisper/CTranslate2",
        "compute_type": _asr_compute_type,
        "batch_size": max(1, WHISPER_BATCH_SIZE),
        "vad_filter": bool(transcribe_kwargs["vad_filter"]),
        "vad_mode": WHISPER_VAD_MODE,
        "transcription_strategy": selected_strategy,
        "transcript": transcript,
        "segments": segments,
        "language": getattr(information, "language", None),
        "language_probability": getattr(information, "language_probability", None),
        "transcript_quality_score": quality_score,
        "transcript_quality_basis": (
            "Configurable ASR-source heuristic; not a calibrated Whisper confidence probability."
        ),
        "coverage": {
            **diagnostics,
            "attempts": attempts,
            "filtered_segment_count": len(rejected_segments),
        },
        "notes": notes,
        "_runtime": {
            "model_load_seconds": round(model_load_seconds, 3),
            "inference_seconds": round(inference_seconds, 3),
            "postprocess_seconds": round(postprocess_seconds, 3),
            "total_seconds": round(time.perf_counter() - total_started, 3),
        },
    }
