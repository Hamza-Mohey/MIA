"""Central configuration for the local meeting-intelligence models.

Every value can be overridden with an environment variable. The defaults are
local deployment choices rather than calibrated model parameters.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
HF_CACHE_DIR = BASE_DIR / "hf_cache"
PROMPTS_DIR = BASE_DIR / "prompts"
FASTER_WHISPER_MODEL_DIR = HF_CACHE_DIR / "faster-whisper-medium"
PIPELINE_VERSION = "4.4"


def cached_model_snapshot(model_name: str) -> Path | None:
    """Return a completed local Hub snapshot when one is already available."""
    snapshots = HF_CACHE_DIR / f"models--{model_name.replace('/', '--')}" / "snapshots"
    if not snapshots.exists():
        return None
    for candidate in snapshots.iterdir():
        if candidate.is_dir() and (
            (candidate / "config.json").exists() or (candidate / "config.yaml").exists()
        ):
            return candidate
    return None


def hub_load_kwargs(model_name: str) -> dict[str, object]:
    """Use one project cache and avoid network checks after a model is cached."""
    return {
        "cache_dir": str(HF_CACHE_DIR),
        "local_files_only": cached_model_snapshot(model_name) is not None,
    }


def _optional_path(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "openai/whisper-medium")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
WHISPER_BATCH_SIZE = _int("WHISPER_BATCH_SIZE", 0)
WHISPER_VAD_MODE = os.getenv("WHISPER_VAD_MODE", "auto").strip().casefold()
if WHISPER_VAD_MODE not in {"auto", "on", "off"}:
    WHISPER_VAD_MODE = "auto"
WHISPER_MIN_WORDS_PER_MINUTE = _float("WHISPER_MIN_WORDS_PER_MINUTE", 100.0)
WHISPER_CRITICAL_WORDS_PER_MINUTE = _float("WHISPER_CRITICAL_WORDS_PER_MINUTE", 80.0)
WHISPER_RETRY_MIN_DURATION_SECONDS = _float(
    "WHISPER_RETRY_MIN_DURATION_SECONDS", 120.0
)
WHISPER_RETRY_IMPROVEMENT_RATIO = _float("WHISPER_RETRY_IMPROVEMENT_RATIO", 1.05)
WHISPER_AUTO_RETRY = _bool("WHISPER_AUTO_RETRY", True)
WHISPER_MAX_NO_SPEECH_PROBABILITY = _float("WHISPER_MAX_NO_SPEECH_PROBABILITY", 0.65)
WHISPER_MIN_AVERAGE_LOG_PROBABILITY = _float(
    "WHISPER_MIN_AVERAGE_LOG_PROBABILITY", -1.0
)
WHISPER_MAX_COMPRESSION_RATIO = _float("WHISPER_MAX_COMPRESSION_RATIO", 2.4)
MODEL_WORKER_TIMEOUT_SECONDS = _int("MODEL_WORKER_TIMEOUT_SECONDS", 7_200)
TEXT_WORKER_TIMEOUT_SECONDS = _int("TEXT_WORKER_TIMEOUT_SECONDS", 360)
TEXT_ADAPTER_PATH = _optional_path("TEXT_ADAPTER_PATH")
_LOCAL_QA_ADAPTER = BASE_DIR / "artifacts" / "qwen_qmsum_qa_lora"
QA_ADAPTER_PATH = _optional_path("QA_ADAPTER_PATH") or (
    _LOCAL_QA_ADAPTER if _LOCAL_QA_ADAPTER.exists() else None
)
DIARIZATION_MODEL_NAME = os.getenv(
    "DIARIZATION_MODEL_NAME", "pyannote/speaker-diarization-community-1"
)
RETRIEVAL_MODEL_NAME = os.getenv("RETRIEVAL_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
MAX_TRANSCRIPT_LENGTH = _int("MAX_TRANSCRIPT_LENGTH", 24_000)
TEXT_CHUNK_LENGTH = _int("TEXT_CHUNK_LENGTH", 16_000)
ANALYSIS_EVIDENCE_MAX_CHARACTERS = _int("ANALYSIS_EVIDENCE_MAX_CHARACTERS", 9_000)
MAX_NEW_TOKENS = _int("MAX_NEW_TOKENS", 512)
TEXT_GENERATION_TIMEOUT_SECONDS = _float("TEXT_GENERATION_TIMEOUT_SECONDS", 65.0)
RETRIEVAL_MAX_CHARACTERS = _int("RETRIEVAL_MAX_CHARACTERS", 6_000)
RETRIEVAL_DENSE_WEIGHT = _float("RETRIEVAL_DENSE_WEIGHT", 0.65)
RETRIEVAL_MIN_SCORE = _float("RETRIEVAL_MIN_SCORE", 0.18)
EVIDENCE_SUPPORT_THRESHOLD = _float("EVIDENCE_SUPPORT_THRESHOLD", 0.38)
EVIDENCE_TOPIC_SUPPORT_THRESHOLD = _float(
    "EVIDENCE_TOPIC_SUPPORT_THRESHOLD", 0.30
)
EVIDENCE_KEY_POINT_SUPPORT_THRESHOLD = _float(
    "EVIDENCE_KEY_POINT_SUPPORT_THRESHOLD", 0.34
)
EVIDENCE_MAX_INDEX_SPAN = _int("EVIDENCE_MAX_INDEX_SPAN", 3)
EVIDENCE_MAX_TIME_SPAN_SECONDS = _float("EVIDENCE_MAX_TIME_SPAN_SECONDS", 120.0)

# Named heuristic defaults are deliberately exposed and described in
# output metadata instead of being presented as calibrated model probabilities.
UPLOADED_TRANSCRIPT_QUALITY = _float("UPLOADED_TRANSCRIPT_QUALITY", 0.95)
ASR_TRANSCRIPT_QUALITY = _float("ASR_TRANSCRIPT_QUALITY", 0.75)
DEFAULT_LLM_CONFIDENCE = _float("DEFAULT_LLM_CONFIDENCE", 0.70)
