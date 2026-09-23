"""Streamlit workspace for local, speaker-aware general meeting intelligence."""

from __future__ import annotations

import json
import os
import re
import time
from copy import deepcopy
from html import escape
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
HF_CACHE_DIR = BASE_DIR / "hf_cache"
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR))
# Runtime inference uses only prefetched models so a meeting never triggers an
# unexpected network download.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st

from modules.analysis_workflow import analyse_segments_with_cache
from modules.audio_transcription import temporary_audio_file
from modules.cache_keys import content_digest, file_fingerprint, stage_cache_key
from modules.config import (
    ANALYSIS_EVIDENCE_MAX_CHARACTERS,
    DIARIZATION_MODEL_NAME,
    MAX_NEW_TOKENS,
    PIPELINE_VERSION,
    QA_ADAPTER_PATH,
    RETRIEVAL_DENSE_WEIGHT,
    RETRIEVAL_MAX_CHARACTERS,
    RETRIEVAL_MIN_SCORE,
    RETRIEVAL_MODEL_NAME,
    TEXT_ADAPTER_PATH,
    TEXT_CHUNK_LENGTH,
    TEXT_GENERATION_TIMEOUT_SECONDS,
    TEXT_MODEL_NAME,
    WHISPER_BATCH_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_CRITICAL_WORDS_PER_MINUTE,
    WHISPER_MIN_WORDS_PER_MINUTE,
    WHISPER_MODEL_NAME,
    WHISPER_RETRY_IMPROVEMENT_RATIO,
    WHISPER_RETRY_MIN_DURATION_SECONDS,
    WHISPER_VAD_MODE,
)
from modules.evidence import build_source_segments, format_timestamp
from modules.general_meeting import generate_general_report, run_general_pipeline
from modules.isolated_inference import (
    analyse_transcript_chunks_isolated,
    answer_meeting_question_isolated,
    combine_transcription_and_diarization,
    diarize_audio_isolated,
    preserve_transcription_after_diarization_failure,
    transcribe_audio_isolated,
)
from modules.retrieval import build_retrieval_index, retrieve_evidence
from modules.text_analysis import PROMPT_PATH

SAMPLES = {
    "Weekly team planning": (
        "SPEAKER_00: We approved moving the customer review to Friday at 10 AM.\n"
        "SPEAKER_01: I will send the revised agenda by Wednesday.\n"
        "SPEAKER_00: The meeting room is still unconfirmed, so that remains an open question."
    ),
    "Project status": (
        "SPEAKER_00: The data migration is two days behind because access has not been approved.\n"
        "SPEAKER_01: I will contact security today and report back tomorrow.\n"
        "SPEAKER_02: We agreed to keep the current launch date unless access is delayed past Thursday."
    ),
    "Committee discussion": (
        "SPEAKER_00: The committee reviewed the training budget and the three venue proposals.\n"
        "SPEAKER_01: We selected the central library for the workshop.\n"
        "SPEAKER_02: I will confirm accessibility and catering costs before the next session."
    ),
}


class SessionUpload:
    """Minimal upload interface for retrying cached audio stages after a rerun."""

    def __init__(self, data: bytes, name: str, mime_type: str):
        self._data = data
        self.name = name
        self.type = mime_type

    def getvalue(self) -> bytes:
        return self._data


def initialise_state() -> None:
    defaults = {
        "working_transcript": "",
        "audio_result": None,
        "meeting_result": None,
        "input_origin": "",
        "qa_history": [],
        "audio_bytes": None,
        "audio_name": "",
        "audio_mime": "audio/wav",
        "speaker_count_hint": 0,
        "stage_cache": {},
        "cache_events": [],
        "selected_evidence_ids": [],
        "playback_start": None,
        "editing_item": None,
        "evidence_review_started": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_workspace() -> None:
    for key in (
        "working_transcript",
        "audio_result",
        "meeting_result",
        "input_origin",
        "qa_history",
        "audio_bytes",
        "audio_name",
        "speaker_count_hint",
        "selected_evidence_ids",
        "playback_start",
        "editing_item",
        "evidence_review_started",
    ):
        if key in {"working_transcript", "input_origin"}:
            st.session_state[key] = ""
        elif key in {"qa_history", "selected_evidence_ids"}:
            st.session_state[key] = []
        elif key == "evidence_review_started":
            st.session_state[key] = {}
        elif key == "audio_name":
            st.session_state[key] = ""
        elif key == "speaker_count_hint":
            st.session_state[key] = 0
        else:
            st.session_state[key] = None


def rename_speakers(transcript: str, mapping: dict[str, str]) -> str:
    result = transcript
    for speaker, name in mapping.items():
        clean_name = " ".join(name.split())
        if clean_name:
            result = re.sub(rf"\b{re.escape(speaker)}(?=:)", clean_name, result)
    return result


def count_items(context: dict[str, Any], key: str) -> int:
    value = context.get(key, [])
    return len(value) if isinstance(value, list) else int(bool(value))


def display_item_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("task") or item.get("text") or item.get("speaker") or "")
    return str(item)


def render_grounded_list(items: list[Any]) -> None:
    for item in items:
        st.markdown(f"- {display_item_text(item)}")
        if isinstance(item, dict) and item.get("evidence_segment_ids"):
            st.caption("Evidence: " + ", ".join(item["evidence_segment_ids"]))


def record_cache_event(stage: str, hit: bool, detail: str = "") -> None:
    events = st.session_state["cache_events"]
    events.append({"stage": stage, "result": "hit" if hit else "miss", "detail": detail})
    del events[:-50]


def cached_result(value: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(value)
    previous_runtime = result.get("_runtime", {})
    result["_runtime"] = {
        "cache_hit": True,
        "model_load_seconds": 0.0,
        "inference_seconds": 0.0,
        "postprocess_seconds": 0.0,
        "subprocess_seconds": 0.0,
        "original_runtime": previous_runtime,
    }
    return result


def prepare_audio(upload: Any, use_diarization: bool, speaker_count: int) -> dict[str, Any]:
    """Reuse transcription independently from optional diarization."""
    audio_bytes = upload.getvalue()
    st.session_state["audio_bytes"] = audio_bytes
    st.session_state["audio_name"] = getattr(upload, "name", "meeting.wav")
    st.session_state["audio_mime"] = getattr(upload, "type", None) or "audio/wav"
    st.session_state["speaker_count_hint"] = speaker_count
    cache = st.session_state["stage_cache"]
    audio_digest = content_digest(audio_bytes)
    transcription_key = stage_cache_key(
        "transcription",
        audio_digest,
        versions={"pipeline": PIPELINE_VERSION, "model": WHISPER_MODEL_NAME},
        configuration={
            "compute_type": WHISPER_COMPUTE_TYPE,
            "batch_size": WHISPER_BATCH_SIZE,
            "beam_size": 5,
            "vad_mode": WHISPER_VAD_MODE,
            "minimum_words_per_minute": WHISPER_MIN_WORDS_PER_MINUTE,
            "critical_words_per_minute": WHISPER_CRITICAL_WORDS_PER_MINUTE,
            "retry_minimum_duration_seconds": WHISPER_RETRY_MIN_DURATION_SECONDS,
            "retry_improvement_ratio": WHISPER_RETRY_IMPROVEMENT_RATIO,
            "condition_on_previous_text": True,
        },
    )
    diarization_key = stage_cache_key(
        "diarization",
        audio_digest,
        versions={"pipeline": PIPELINE_VERSION, "model": DIARIZATION_MODEL_NAME},
        configuration={"num_speakers": speaker_count or None},
    )
    transcription_hit = transcription_key in cache
    diarization_hit = use_diarization and diarization_key in cache
    transcription = cached_result(cache[transcription_key]) if transcription_hit else None
    diarization = cached_result(cache[diarization_key]) if diarization_hit else None
    record_cache_event("transcription", transcription_hit, audio_digest[:12])
    if use_diarization:
        record_cache_event("diarization", diarization_hit, audio_digest[:12])

    if transcription is None or (use_diarization and diarization is None):
        with temporary_audio_file(upload) as audio_path:
            if transcription is None:
                transcription = transcribe_audio_isolated(audio_path)
                transcription.setdefault("_runtime", {})["cache_hit"] = False
                cache[transcription_key] = deepcopy(transcription)
            if use_diarization and diarization is None:
                try:
                    diarization = diarize_audio_isolated(
                        audio_path, num_speakers=speaker_count or None
                    )
                    diarization.setdefault("_runtime", {})["cache_hit"] = False
                    cache[diarization_key] = deepcopy(diarization)
                except RuntimeError as exc:
                    return preserve_transcription_after_diarization_failure(transcription, exc)

    if use_diarization and diarization is not None:
        result = combine_transcription_and_diarization(transcription, diarization)
        result["_cache"] = {
            "transcription": "hit" if transcription_hit else "miss",
            "diarization": "hit" if diarization_hit else "miss",
        }
        return result
    transcription["_cache"] = {"transcription": "hit" if transcription_hit else "miss"}
    return transcription


def analyse_meeting(
    transcript: str,
    audio_result: dict[str, Any],
    progress_callback: Any = None,
) -> dict[str, Any]:
    """Run only changed transcript chunks, then rebuild evidence and the report."""
    total_started = time.perf_counter()
    source_segments = build_source_segments(transcript, audio_result)
    if progress_callback:
        progress_callback("Checking the versioned analysis cache", 0.08)
    analysis, cache_stats = analyse_segments_with_cache(
        source_segments,
        st.session_state["stage_cache"],
        transcript_quality=audio_result.get("transcript_quality_score", 0.95),
        quality_basis=audio_result.get("transcript_quality_basis", "uploaded transcript heuristic"),
        versions={
            "pipeline": PIPELINE_VERSION,
            "model": TEXT_MODEL_NAME,
            "prompt": file_fingerprint(PROMPT_PATH),
            "text_adapter": file_fingerprint(TEXT_ADAPTER_PATH),
        },
        configuration={
            "analysis_evidence_characters": ANALYSIS_EVIDENCE_MAX_CHARACTERS,
            "analysis_windows": 3,
            "max_new_tokens": MAX_NEW_TOKENS,
            "generation_timeout_seconds": TEXT_GENERATION_TIMEOUT_SECONDS,
        },
        batch_analyser=analyse_transcript_chunks_isolated,
        progress_callback=progress_callback,
    )
    record_cache_event(
        "transcript analysis",
        cache_stats["misses"] == 0,
        f"{cache_stats['hits']} hit / {cache_stats['misses']} miss",
    )
    if progress_callback:
        progress_callback("Building the evidence-linked meeting brief", 0.88)
    result = run_general_pipeline(
        transcript,
        diarization=audio_result.get("diarization"),
        transcript_quality=audio_result.get("transcript_quality_score", 0.95),
        quality_basis=audio_result.get("transcript_quality_basis", "uploaded transcript heuristic"),
        source_segments=source_segments,
        text_analyser=lambda *_args, **_kwargs: deepcopy(analysis),
        progress_callback=None,
    )
    result["transcript"] = transcript
    result["cache"] = {"analysis": cache_stats, "scope": "current Streamlit session"}
    result["timings"]["transcription"] = audio_result.get("_runtime", {})
    result["timings"]["diarization"] = (audio_result.get("diarization") or {}).get("_runtime", {})
    analysis_request_seconds = round(time.perf_counter() - total_started, 3)
    result["timings"]["analysis_request_seconds"] = analysis_request_seconds
    transcription_seconds = float(
        result["timings"]["transcription"].get("total_seconds", 0.0) or 0.0
    )
    diarization_seconds = float(
        result["timings"]["diarization"].get("total_seconds", 0.0) or 0.0
    )
    result["timings"]["pipeline_processing_seconds"] = round(
        transcription_seconds + diarization_seconds + analysis_request_seconds, 3
    )
    result["meeting_context"]["quality"]["transcription_coverage"] = audio_result.get(
        "coverage", {}
    )
    result["models"]["retrieval"] = RETRIEVAL_MODEL_NAME
    if progress_callback:
        progress_callback("Complete", 1.0)
    return result


def store_meeting_result(result: dict[str, Any]) -> None:
    st.session_state["meeting_result"] = result


def retrieval_index_for(result: dict[str, Any]) -> dict[str, Any]:
    segments = result.get("source_segments", [])
    key = stage_cache_key(
        "retrieval_index",
        segments,
        versions={"pipeline": PIPELINE_VERSION, "model": RETRIEVAL_MODEL_NAME},
        configuration={"unit": "source_segment"},
    )
    cache = st.session_state["stage_cache"]
    if key in cache:
        record_cache_event("retrieval index", True, f"{len(segments)} segments")
        index = cache[key]
        index["_current_cache_hit"] = True
        return index
    index = build_retrieval_index(segments)
    index["_current_cache_hit"] = False
    cache[key] = index
    record_cache_event("retrieval index", False, f"{len(segments)} segments")
    return index


def answer_with_cache(question: str, retrieval: dict[str, Any]) -> dict[str, Any]:
    if retrieval["status"] != "supported":
        return {
            "answer": (
                "I could not find strong supporting evidence in the indexed transcript. "
                "This does not prove that the topic was never discussed."
            ),
            "_runtime": {"skipped_generation": True},
        }
    key = stage_cache_key(
        "meeting_answer",
        {"question": question, "evidence": retrieval["evidence_text"]},
        versions={
            "pipeline": PIPELINE_VERSION,
            "model": TEXT_MODEL_NAME,
            "qa_adapter": file_fingerprint(QA_ADAPTER_PATH),
        },
        configuration={"max_new_tokens": min(256, MAX_NEW_TOKENS)},
    )
    cache = st.session_state["stage_cache"]
    if key in cache:
        record_cache_event("meeting answer", True, question[:60])
        return cached_result(cache[key])
    answer = answer_meeting_question_isolated(retrieval["evidence_text"], question)
    answer.setdefault("_runtime", {})["cache_hit"] = False
    cache[key] = deepcopy(answer)
    record_cache_event("meeting answer", False, question[:60])
    return answer


def select_evidence(item: dict[str, Any], item_key: str, *, play: bool = False) -> None:
    st.session_state["selected_evidence_ids"] = item.get("evidence_segment_ids", [])
    st.session_state["playback_start"] = item.get("evidence_start_seconds") if play else None
    st.session_state["evidence_review_started"].setdefault(item_key, time.time())


def render_selected_evidence(result: dict[str, Any]) -> None:
    selected_ids = set(st.session_state.get("selected_evidence_ids", []))
    if not selected_ids:
        return
    selected = [
        segment for segment in result.get("source_segments", []) if segment["id"] in selected_ids
    ]
    if not selected:
        st.warning("The selected evidence no longer exists in the current transcript.")
        return
    st.markdown("#### Evidence inspector")
    for segment in selected:
        timestamp = format_timestamp(segment.get("start")) or "text only"
        speaker = escape(segment.get("speaker") or "Unknown speaker")
        text = escape(segment.get("text", ""))
        st.markdown(
            f'<div class="source-segment selected"><div class="segment-meta">'
            f"{escape(segment['id'])} · {timestamp} · {speaker}</div>{text}</div>",
            unsafe_allow_html=True,
        )
    playback_start = st.session_state.get("playback_start")
    if playback_start is not None and st.session_state.get("audio_bytes"):
        st.audio(
            st.session_state["audio_bytes"],
            format=st.session_state.get("audio_mime") or "audio/wav",
            start_time=int(max(0, playback_start)),
        )


def update_review_item(
    result: dict[str, Any], list_key: str, index: int, updated: dict[str, Any]
) -> None:
    result["meeting_context"][list_key][index] = updated
    analysis_items = result.get("text_analysis", {}).get(list_key, [])
    if index < len(analysis_items):
        analysis_items[index] = deepcopy(updated)
    result["report_markdown"] = generate_general_report(result["meeting_context"])
    st.session_state["meeting_result"] = result


def render_review_item(
    result: dict[str, Any], list_key: str, index: int, item: dict[str, Any]
) -> None:
    item_key = f"{list_key}-{index}"
    title = item.get("task") or item.get("text") or "Untitled item"
    with st.container(border=True):
        st.subheader(title)
        details = []
        if item.get("owner"):
            details.append(f"Owner: {item['owner']}")
        if item.get("deadline"):
            details.append(f"Deadline: {item['deadline']}")
        if item.get("commitment_status"):
            details.append(f"Status: {item['commitment_status']}")
        details.append(f"Review: {item.get('review_status', 'draft')}")
        if item.get("verification_seconds") is not None:
            details.append(f"Verified in {item['verification_seconds']:.1f}s")
        st.caption(" · ".join(details))
        if item.get("evidence_status") == "linked":
            st.markdown("**Evidence**")
            st.write(f"“{item.get('evidence', '')}”")
            st.caption("Source: " + ", ".join(item.get("evidence_segment_ids", [])))
        else:
            st.warning("No valid source segment could be linked. Treat this item as unverified.")
        view, play, edit, confirm = st.columns(4)
        if view.button(
            "View transcript",
            key=f"view-{item_key}",
            disabled=not item.get("evidence_segment_ids"),
            use_container_width=True,
        ):
            select_evidence(item, item_key)
            st.rerun()
        can_play = bool(
            st.session_state.get("audio_bytes") and item.get("evidence_start_seconds") is not None
        )
        if play.button(
            "Play audio",
            key=f"play-{item_key}",
            disabled=not can_play,
            use_container_width=True,
        ):
            select_evidence(item, item_key, play=True)
            st.rerun()
        if edit.button("Edit", key=f"edit-{item_key}", use_container_width=True):
            st.session_state["editing_item"] = item_key
            st.rerun()
        if confirm.button(
            "Confirm",
            key=f"confirm-{item_key}",
            disabled=item.get("review_status") == "confirmed",
            use_container_width=True,
        ):
            updated = deepcopy(item)
            updated["review_status"] = "confirmed"
            started = st.session_state["evidence_review_started"].get(item_key)
            if started is not None:
                updated["verification_seconds"] = round(time.time() - started, 2)
            update_review_item(result, list_key, index, updated)
            st.rerun()

        if st.session_state.get("editing_item") == item_key:
            with st.form(f"edit-form-{item_key}"):
                text_field = "task" if list_key == "action_items" else "text"
                edited_text = st.text_input("Text", value=item.get(text_field, ""))
                owner = deadline = None
                if list_key == "action_items":
                    owner = st.text_input("Owner", value=item.get("owner") or "")
                    deadline = st.text_input("Deadline", value=item.get("deadline") or "")
                save, cancel = st.columns(2)
                save_clicked = save.form_submit_button(
                    "Save changes", type="primary", use_container_width=True
                )
                cancel_clicked = cancel.form_submit_button("Cancel", use_container_width=True)
            if save_clicked and edited_text.strip():
                updated = deepcopy(item)
                updated[text_field] = edited_text.strip()
                if list_key == "action_items":
                    updated["owner"] = owner.strip() or None
                    updated["deadline"] = deadline.strip() or None
                updated["review_status"] = "edited"
                st.session_state["editing_item"] = None
                update_review_item(result, list_key, index, updated)
                st.rerun()
            if cancel_clicked:
                st.session_state["editing_item"] = None
                st.rerun()


initialise_state()
st.set_page_config(
    page_title="Meeting Intelligence",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
<style>
    :root {color-scheme: dark; --ink:#f8f7f2; --muted:#b5b8c2; --line:#3a3f4c; --panel:#161921; --accent:#8b7cf6; --accent-strong:#7868ed; --mint:#65d6b4;}
    html, body, [class*="css"] {font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;}
    .stApp {background:#0d0f13; color:var(--ink);}
    [data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stHeader"], #MainMenu, footer {display:none !important;}
    .block-container {width:100%; max-width:none; padding:1.35rem clamp(1.25rem, 4vw, 4.5rem) 5rem;}

    .topbar {display:flex; align-items:center; gap:.75rem; min-height:2.6rem;}
    .brand-mark {display:grid; place-items:center; width:2rem; height:2rem; border-radius:10px; background:var(--ink); color:#111318; font-size:.95rem; font-weight:900;}
    .brand-copy {font-size:.94rem; font-weight:700; letter-spacing:-.01em; color:var(--ink);}
    .brand-copy span {display:block; color:#777b86; font-size:.66rem; font-weight:500; letter-spacing:.04em; margin-top:.05rem; text-transform:uppercase;}
    .runtime-badge {display:inline-flex; align-items:center; gap:.45rem; padding:.42rem .7rem; border:1px solid var(--line); border-radius:999px; color:#c8cad1; background:#121419; font-size:.72rem; white-space:nowrap; margin-top:.25rem;}
    .runtime-dot {width:.42rem; height:.42rem; border-radius:50%; background:var(--mint); box-shadow:0 0 0 3px rgba(101,214,180,.1);}

    .hero {position:relative; padding:6.3rem 0 4.2rem; border-bottom:1px solid var(--line); overflow:hidden;}
    .hero:after {content:""; position:absolute; width:25rem; height:25rem; right:-10rem; top:-8rem; border-radius:50%; background:rgba(139,124,246,.10); filter:blur(2px); pointer-events:none;}
    .eyebrow {color:#a99df9; font-size:.7rem; font-weight:750; letter-spacing:.16em; text-transform:uppercase;}
    .hero h1 {position:relative; max-width:1100px; font-size:clamp(2.7rem,6vw,5.4rem); line-height:.98; letter-spacing:-.065em; margin:.75rem 0 1.15rem; color:var(--ink); z-index:1;}
    .hero h1 span {color:#8d9099;}
    .hero p {max-width:850px; color:var(--muted); font-size:1.08rem; line-height:1.7; margin:0;}
    .trust-row {display:flex; flex-wrap:wrap; gap:1.2rem; margin-top:1.65rem; color:#747883; font-size:.75rem;}
    .trust-row span:before {content:"✓"; color:var(--mint); margin-right:.42rem;}
    .section-intro {margin:3.4rem 0 1.35rem;}
    .section-kicker {color:#777b85; font-size:.7rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; margin-bottom:.45rem;}
    .section-intro h2 {font-size:1.8rem; letter-spacing:-.035em; margin:0; color:var(--ink);}
    .result-heading {padding:3rem 0 1rem;}

    h1, h2, h3, h4, h5, h6, p, label, [data-testid="stMarkdownContainer"] {color:var(--ink);}
    h2, h3 {letter-spacing:-.025em;}
    [data-testid="stCaptionContainer"], .stCaption {color:var(--muted) !important;}
    hr {border-color:var(--line) !important; margin:2rem 0 !important;}

    [data-baseweb="tab-list"] {gap:.35rem; background:#121419; border:1px solid var(--line); border-radius:12px; padding:.28rem;}
    [data-baseweb="tab-border"] {display:none;}
    [data-baseweb="tab"] {height:2.45rem; border-radius:8px; padding:0 1rem; color:#c2c5ce; font-size:.82rem;}
    [aria-selected="true"][data-baseweb="tab"] {background:var(--accent-strong); color:#fff !important;}

    [data-testid="stFileUploader"] {background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:.25rem;}
    [data-testid="stFileUploaderDropzone"] {background:#11141a; color:var(--ink); border:1px dashed #626978; border-radius:12px; min-height:9rem;}
    [data-testid="stFileUploaderDropzone"] button {background:var(--accent-strong) !important; color:#fff !important; border:1px solid #a79cf8 !important;}
    [data-testid="stStatusWidget"], [data-testid="stExpander"] {background:var(--panel); border:1px solid var(--line); border-radius:12px;}
    [data-testid="stMetric"] {background:var(--panel); border:1px solid var(--line); padding:1.05rem 1.1rem; border-radius:14px; box-shadow:none;}
    [data-testid="stMetricLabel"] {color:#b8bbc5;}
    [data-testid="stMetricValue"] {color:var(--ink); letter-spacing:-.04em;}
    [data-testid="stDataFrame"] {border:1px solid var(--line); border-radius:12px; overflow:hidden;}
    .source-segment {background:#11141a; border:1px solid var(--line); border-radius:10px; padding:.85rem 1rem; margin:.45rem 0; color:var(--ink); line-height:1.55;}
    .source-segment.selected {border-color:var(--accent); background:#1b1930; box-shadow:0 0 0 1px rgba(139,124,246,.22);}
    .segment-meta {color:#a99df9; font-size:.68rem; font-weight:750; letter-spacing:.06em; text-transform:uppercase; margin-bottom:.35rem;}

    .stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input {background:#121419; color:var(--ink); border-color:#555c6b; border-radius:10px;}
    .stTextInput input:focus, .stTextArea textarea:focus {border-color:var(--accent); box-shadow:0 0 0 1px var(--accent);}
    .stButton button, .stDownloadButton button, [data-testid="stFormSubmitButton"] button {
        min-height:2.65rem; background:#252a36; color:#ffffff; border:1px solid #697184; border-radius:10px; font-weight:650; box-shadow:0 1px 0 rgba(255,255,255,.04);
    }
    .stButton button *, .stDownloadButton button *, [data-testid="stFormSubmitButton"] button * {color:#ffffff !important;}
    .stButton button:hover, .stDownloadButton button:hover, [data-testid="stFormSubmitButton"] button:hover {background:#343a49; border-color:#aaa2f6; color:#fff;}
    .stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button[kind="primary"] {background:var(--ink); border-color:var(--ink); color:#111318;}
    .stButton button[kind="primary"] *, [data-testid="stFormSubmitButton"] button[kind="primary"] * {color:#111318 !important;}
    .stButton button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button[kind="primary"]:hover {background:#dcdad4; border-color:#dcdad4; color:#111318;}
    .stButton button:disabled, [data-testid="stFormSubmitButton"] button:disabled {background:#1b1e26; color:#858a98; border-color:#3f4552; opacity:1;}
    .stButton button:disabled *, [data-testid="stFormSubmitButton"] button:disabled * {color:#858a98 !important;}
    [data-testid="stNumberInput"] button {background:#252a36; color:#fff; border-color:#555c6b;}
    [data-testid="stAlert"] {background:#151820; border:1px solid #303540; border-radius:12px; color:var(--ink);}
    [data-testid="stProgress"] {background:#151820; border:1px solid #303540; border-radius:12px; padding:.8rem 1rem .9rem; margin:.75rem 0;}
    [data-testid="stProgress"] > div:first-child {padding-bottom:.55rem; line-height:1;}
    [data-testid="stProgress"] [data-testid="stMarkdownContainer"] p {color:#d7d9e0 !important; font-size:.78rem; font-weight:600;}
    [data-testid="stProgressBarTrack"] {height:.48rem; background:#292e39 !important; border-radius:999px !important; overflow:hidden;}
    [data-testid="stProgressBarTrack"] > div {background:linear-gradient(90deg, var(--accent-strong), #a79cf8) !important; border-radius:999px;}

    @media (max-width: 700px) {
        .block-container {padding:1rem 1rem 3rem;}
        .hero {padding:4.2rem 0 3rem;}
        .hero h1 {font-size:2.75rem;}
        .runtime-badge {display:none;}
        [data-baseweb="tab-list"] {overflow-x:auto;}
    }
</style>
""",
    unsafe_allow_html=True,
)

qwen_runtime = "isolated local worker"
nav_brand, nav_status, nav_action = st.columns([6, 2.2, 1.35], vertical_alignment="center")
with nav_brand:
    st.markdown(
        '<div class="topbar"><div class="brand-mark">M</div><div class="brand-copy">Meeting Intelligence<span>Private analysis workspace</span></div></div>',
        unsafe_allow_html=True,
    )
with nav_status:
    st.markdown(
        f'<div class="runtime-badge"><span class="runtime-dot"></span>{qwen_runtime}</div>',
        unsafe_allow_html=True,
    )
with nav_action:
    if st.button("New meeting", use_container_width=True):
        reset_workspace()
        st.rerun()

if st.session_state["meeting_result"] is None:
    st.markdown(
        """
<div class="hero">
  <div class="eyebrow">From conversation to clarity</div>
  <h1>Meetings in.<br><span>Clear next steps out.</span></h1>
  <p>Upload a recording or transcript. Get speaker-aware notes, decisions,
  action items, risks, and grounded answers in one focused workspace.</p>
  <div class="trust-row">
    <span>Three-model pipeline</span>
    <span>Evidence-aware output</span>
    <span>Human review built in</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="result-heading"><div class="eyebrow">Analysis complete</div></div>',
        unsafe_allow_html=True,
    )

if st.session_state["meeting_result"] is None:
    st.markdown(
        '<div class="section-intro"><div class="section-kicker">01 · Add material</div><h2>Start with a recording or transcript</h2></div>',
        unsafe_allow_html=True,
    )
    audio_tab, transcript_tab, sample_tab = st.tabs(
        ["Audio recording", "Existing transcript", "Try a sample"]
    )
    audio_file = None
    uploaded_text = ""
    with audio_tab:
        audio_file = st.file_uploader(
            "Upload a recording",
            type=["wav", "mp3", "m4a", "flac"],
            help="For best diarization, use clear audio with limited background noise.",
        )
        if audio_file is not None:
            st.audio(audio_file)
        option_left, option_right = st.columns(2)
        with option_left:
            use_diarization = st.toggle("Identify speaker turns", value=True)
        with option_right:
            known_speakers = st.number_input(
                "Known speaker count (0 = automatic)", min_value=0, max_value=12, value=0, step=1
            )
        if st.button(
            "Prepare speaker transcript",
            disabled=audio_file is None,
            use_container_width=True,
        ):
            try:
                label = (
                    "Transcribing and separating speakers"
                    if use_diarization
                    else "Transcribing audio"
                )
                with st.status(label, expanded=True) as status:
                    st.write("Running Whisper transcription…")
                    if use_diarization:
                        st.write("Running pyannote speaker diarization after transcription…")
                    result = prepare_audio(audio_file, use_diarization, int(known_speakers))
                    status.update(label="Speaker transcript ready", state="complete")
                st.session_state["audio_result"] = result
                st.session_state["working_transcript"] = result.get("transcript", "")
                st.session_state["input_origin"] = "audio"
                st.rerun()
            except Exception as exc:
                st.error(f"Audio preparation failed: {exc}")
    with transcript_tab:
        transcript_file = st.file_uploader(
            "Upload a UTF-8 transcript", type=["txt"], key="transcript_upload"
        )
        if transcript_file is not None:
            uploaded_text = transcript_file.getvalue().decode("utf-8", errors="replace")
        pasted_text = st.text_area(
            "Or paste a transcript",
            value="" if uploaded_text else st.session_state.get("pasted_transcript", ""),
            height=180,
            placeholder="SPEAKER_00: Let's begin with the first agenda item…",
        )
        candidate = uploaded_text or pasted_text
        if st.button(
            "Use this transcript", disabled=not candidate.strip(), use_container_width=True
        ):
            st.session_state["working_transcript"] = candidate.strip()
            st.session_state["audio_result"] = None
            st.session_state["input_origin"] = "transcript"
            st.rerun()
    with sample_tab:
        sample_name = st.selectbox("Sample meeting", list(SAMPLES))
        st.code(SAMPLES[sample_name], language=None)
        if st.button("Load sample", use_container_width=True):
            st.session_state["working_transcript"] = SAMPLES[sample_name]
            st.session_state["audio_result"] = None
            st.session_state["input_origin"] = "sample"
            st.rerun()

    transcript = st.session_state.get("working_transcript", "")
    if transcript:
        st.divider()
        st.markdown(
            '<div class="section-intro"><div class="section-kicker">02 · Review</div><h2>Check the transcript before analysis</h2></div>',
            unsafe_allow_html=True,
        )
        audio_result = st.session_state.get("audio_result") or {}
        coverage = audio_result.get("coverage") or {}
        low_coverage = bool(coverage.get("low_coverage"))
        critical_coverage = bool(coverage.get("critical_coverage"))
        allow_low_coverage_analysis = True
        if coverage:
            st.caption(
                f"Transcript coverage: {coverage.get('word_count', 0):,} words · "
                f"{coverage.get('words_per_minute', 0)} words/min · "
                f"{coverage.get('duration_seconds', 0) / 60:.1f} min audio"
            )
        if low_coverage:
            st.warning(
                "This transcript has lower-than-typical speech coverage. Quiet, distant, "
                "or overlapping speakers may be missing, so review it before generating a brief."
            )
            if critical_coverage:
                allow_low_coverage_analysis = st.checkbox(
                    "I reviewed the potentially incomplete transcript and want to analyse it anyway",
                    key="allow_low_coverage_analysis",
                )
        if audio_result.get("diarization_error"):
            st.warning(
                "Whisper transcription succeeded, but speaker separation is unavailable. "
                "Accept the Community-1 conditions on Hugging Face, then retry if you need speaker labels."
            )
            with st.expander("Speaker diarization details"):
                st.code(audio_result["diarization_error"], language=None)
            if st.session_state.get("audio_bytes") and st.button(
                "Retry speaker separation", use_container_width=True
            ):
                try:
                    retry_upload = SessionUpload(
                        st.session_state["audio_bytes"],
                        st.session_state.get("audio_name") or "meeting.wav",
                        st.session_state.get("audio_mime") or "audio/wav",
                    )
                    with st.spinner("Retrying pyannote; reusing the Whisper transcript…"):
                        retried = prepare_audio(
                            retry_upload,
                            True,
                            int(st.session_state.get("speaker_count_hint") or 0),
                        )
                    st.session_state["audio_result"] = retried
                    st.session_state["working_transcript"] = retried.get("transcript", "")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Speaker separation retry failed: {exc}")
        speakers = (audio_result.get("diarization") or {}).get("speakers", [])
        mapping: dict[str, str] = {}
        if speakers:
            st.caption("Optional: replace anonymous speaker labels with names before analysis.")
            columns = st.columns(min(4, len(speakers)))
            for index, speaker in enumerate(speakers):
                with columns[index % len(columns)]:
                    mapping[speaker] = st.text_input(
                        speaker, key=f"speaker_name_{speaker}", placeholder="Name"
                    )
        edited = st.text_area("Transcript", transcript, height=300, key="transcript_editor")
        final_transcript = rename_speakers(edited, mapping)
        st.caption(
            f"{len(final_transcript):,} characters • later sections are processed in bounded chunks"
        )
        if st.button(
            "Generate meeting brief",
            type="primary",
            use_container_width=True,
            disabled=not allow_low_coverage_analysis,
        ):
            try:
                progress = st.progress(0, text="Starting analysis")

                def update_progress(stage: str, fraction: float) -> None:
                    progress.progress(int(fraction * 100), text=stage)

                with st.spinner("Qwen is extracting supported meeting facts…"):
                    meeting_result = analyse_meeting(
                        final_transcript, audio_result, update_progress
                    )
                store_meeting_result(meeting_result)
                st.session_state["selected_evidence_ids"] = []
                st.session_state["qa_history"] = []
                st.rerun()
            except Exception as exc:
                st.error(f"Meeting analysis failed: {type(exc).__name__}: {exc}")

result = st.session_state.get("meeting_result")
if result:
    context = result["meeting_context"]
    st.markdown(
        '<div class="section-intro" style="margin-top:.5rem"><div class="section-kicker">Meeting brief</div><h2>The conversation, distilled</h2></div>',
        unsafe_allow_html=True,
    )
    metric_columns = st.columns(6)
    speaker_count = context.get("quality", {}).get("speaker_count")
    metric_columns[0].metric(
        "Speakers", speaker_count if speaker_count is not None else count_items(context, "participants")
    )
    metric_columns[1].metric("Topics", count_items(context, "topics"))
    metric_columns[2].metric("Requirements", count_items(context, "requirements"))
    metric_columns[3].metric("Decisions", count_items(context, "decisions"))
    metric_columns[4].metric("Actions", count_items(context, "action_items"))
    metric_columns[5].metric("Open questions", count_items(context, "open_questions"))

    overview, actions, decisions, ask_tab, transcript_tab, system_tab = st.tabs(
        [
            "Overview",
            "Actions",
            "Decisions & risks",
            "Ask the meeting",
            "Transcript",
            "System details",
        ]
    )
    with overview:
        st.markdown("### Summary")
        st.write(context.get("meeting_summary", ""))
        left, right = st.columns(2)
        with left:
            st.markdown("### Topics")
            render_grounded_list(context.get("topics", []))
        with right:
            st.markdown("### Key points")
            render_grounded_list(context.get("key_points", []))
        st.markdown("### Requirements and constraints")
        render_grounded_list(context.get("requirements", []))
        with st.expander("Full rendered brief"):
            st.markdown(result["report_markdown"])
    with actions:
        render_selected_evidence(result)
        action_items = context.get("action_items", [])
        if action_items:
            for index, item in enumerate(action_items):
                if isinstance(item, dict):
                    render_review_item(result, "action_items", index, item)
        else:
            st.info("No explicit action commitments were found.")
        st.markdown("### Open questions")
        render_grounded_list(context.get("open_questions", []))
    with decisions:
        render_selected_evidence(result)
        left, right = st.columns(2)
        with left:
            st.markdown("### Decisions")
            decision_items = context.get("decisions", [])
            if decision_items:
                for index, item in enumerate(decision_items):
                    if isinstance(item, dict):
                        render_review_item(result, "decisions", index, item)
            else:
                st.info("No explicit decisions were found.")
        with right:
            st.markdown("### Risks and blockers")
            render_grounded_list(context.get("risks", []))
    with ask_tab:
        st.markdown("### Ask a grounded follow-up")
        st.caption(
            "Answers use the most relevant transcript sections and should still be reviewed."
        )
        with st.form("meeting_question_form", clear_on_submit=True):
            question = st.text_input(
                "Question",
                placeholder="Who owns the follow-up, and when is it due?",
            )
            ask_submitted = st.form_submit_button(
                "Ask Qwen", type="primary", use_container_width=True
            )
        if ask_submitted and question.strip():
            try:
                with st.spinner("Finding transcript evidence and drafting an answer…"):
                    retrieval_index = retrieval_index_for(result)
                    retrieval = retrieve_evidence(
                        question,
                        retrieval_index,
                        max_characters=RETRIEVAL_MAX_CHARACTERS,
                        dense_weight=RETRIEVAL_DENSE_WEIGHT,
                        minimum_score=RETRIEVAL_MIN_SCORE,
                    )
                    answer_result = answer_with_cache(question.strip(), retrieval)
                st.session_state["qa_history"].append(
                    {
                        "question": question.strip(),
                        "answer": answer_result["answer"],
                        "answer_runtime": answer_result.get("_runtime", {}),
                        "retrieval": retrieval,
                    }
                )
            except Exception as exc:
                st.error(f"Question answering failed: {type(exc).__name__}: {exc}")
        for exchange in reversed(st.session_state.get("qa_history", [])):
            st.markdown(f"**Q: {exchange['question']}**")
            st.write(exchange["answer"])
            retrieval = exchange.get("retrieval", {})
            mode = retrieval.get("mode", "unknown")
            score = retrieval.get("best_score", 0.0)
            st.caption(
                f"Retrieval: {mode} · best score {score:.3f} · "
                f"{retrieval.get('estimated_tokens', 0)} estimated evidence tokens"
            )
            if retrieval.get("status") == "low_confidence":
                st.warning(
                    "Retrieval did not find strong support. This is different from proving "
                    "that the topic was absent from the complete meeting."
                )
            if retrieval.get("embedding_error"):
                st.warning(
                    "Semantic retrieval was unavailable for this question, so lexical search "
                    "was used instead. " + retrieval["embedding_error"]
                )
            with st.expander("Passages supplied to Qwen"):
                for segment in retrieval.get("selected_segments", []):
                    timestamp = format_timestamp(segment.get("start")) or "text only"
                    st.markdown(
                        f"**{segment['id']} · {timestamp} · "
                        f"{segment.get('speaker') or 'Unknown speaker'}**"
                    )
                    st.write(segment.get("text", ""))
            st.divider()
    with transcript_tab:
        render_selected_evidence(result)
        selected_ids = set(st.session_state.get("selected_evidence_ids", []))
        with st.expander("Segmented transcript", expanded=bool(selected_ids)):
            for segment in result.get("source_segments", []):
                selected_class = " selected" if segment["id"] in selected_ids else ""
                timestamp = format_timestamp(segment.get("start")) or "text only"
                st.markdown(
                    f'<div class="source-segment{selected_class}"><div class="segment-meta">'
                    f"{escape(segment['id'])} · {timestamp} · "
                    f"{escape(segment.get('speaker') or 'Unknown speaker')}</div>"
                    f"{escape(segment.get('text', ''))}</div>",
                    unsafe_allow_html=True,
                )
        corrected_transcript = st.text_area(
            "Correct transcript",
            result.get("transcript", ""),
            height=420,
            key=f"result-transcript-{content_digest(result.get('transcript', ''))[:12]}",
        )
        st.caption(
            "Reanalysis reuses every unchanged versioned evidence window. Only changed windows "
            "are sent back to Qwen, then every claim is revalidated against the transcript."
        )
        if st.button(
            "Reanalyse transcript",
            type="primary",
            disabled=not corrected_transcript.strip(),
            use_container_width=True,
        ):
            try:
                progress = st.progress(0, text="Comparing meeting evidence windows")

                def update_reanalysis_progress(stage: str, fraction: float) -> None:
                    progress.progress(int(fraction * 100), text=stage)

                updated_result = analyse_meeting(
                    corrected_transcript.strip(),
                    st.session_state.get("audio_result") or {},
                    update_reanalysis_progress,
                )
                store_meeting_result(updated_result)
                st.session_state["working_transcript"] = corrected_transcript.strip()
                st.session_state["selected_evidence_ids"] = []
                st.session_state["qa_history"] = []
                st.rerun()
            except Exception as exc:
                st.error(f"Transcript reanalysis failed: {type(exc).__name__}: {exc}")
    with system_tab:
        st.json(
            {
                "models": result.get("models", {}),
                "quality": context.get("quality", {}),
                "timings": result.get("timings", {}),
                "runtime": {
                    "qwen": qwen_runtime,
                    "whisper": "isolated local worker",
                    "pyannote": "isolated local worker",
                },
                "uncertainties": context.get("uncertainties", []),
                "adapters": {
                    "qwen": str(TEXT_ADAPTER_PATH) if TEXT_ADAPTER_PATH else None,
                    "qwen_qa": str(QA_ADAPTER_PATH) if QA_ADAPTER_PATH else None,
                },
                "cache": {
                    "result": result.get("cache", {}),
                    "entries": len(st.session_state.get("stage_cache", {})),
                    "recent_events": st.session_state.get("cache_events", [])[-12:],
                    "retention": "memory only; cleared when the Streamlit session ends",
                },
                "configuration": {
                    "whisper_compute_type": WHISPER_COMPUTE_TYPE,
                    "whisper_batch_size": WHISPER_BATCH_SIZE,
                    "whisper_vad_mode": WHISPER_VAD_MODE,
                    "whisper_minimum_words_per_minute": WHISPER_MIN_WORDS_PER_MINUTE,
                    "whisper_critical_words_per_minute": WHISPER_CRITICAL_WORDS_PER_MINUTE,
                    "retrieval_dense_weight": RETRIEVAL_DENSE_WEIGHT,
                    "retrieval_minimum_score": RETRIEVAL_MIN_SCORE,
                    "retrieval_evidence_character_budget": RETRIEVAL_MAX_CHARACTERS,
                    "qwen_source_chunk_ceiling": TEXT_CHUNK_LENGTH,
                    "qwen_evidence_digest_characters": ANALYSIS_EVIDENCE_MAX_CHARACTERS,
                    "qwen_evidence_windows": 3,
                    "qwen_max_new_tokens": MAX_NEW_TOKENS,
                    "qwen_generation_timeout_seconds": TEXT_GENERATION_TIMEOUT_SECONDS,
                },
                "review_measurement": {
                    "definition": (
                        "Seconds from first evidence view/play to confirmation; omitted when "
                        "an item is confirmed without opening evidence."
                    ),
                    "confirmed_item_seconds": [
                        item["verification_seconds"]
                        for key in ("action_items", "decisions")
                        for item in context.get(key, [])
                        if isinstance(item, dict) and item.get("verification_seconds") is not None
                    ],
                },
            }
        )
        st.caption("Model confidence and source-quality values are not calibrated probabilities.")
        if st.button("Clear session cache", use_container_width=True):
            st.session_state["stage_cache"] = {}
            st.session_state["cache_events"] = []
            st.success("Cached transcripts, analyses, retrieval indexes, and answers were deleted.")

    json_data = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    download_left, download_middle, download_right = st.columns(3)
    download_left.download_button(
        "Download brief (.md)",
        result["report_markdown"],
        "meeting_brief.md",
        "text/markdown",
        use_container_width=True,
    )
    download_middle.download_button(
        "Download data (.json)",
        json_data,
        "meeting_intelligence.json",
        "application/json",
        use_container_width=True,
    )
    download_right.download_button(
        "Download transcript (.txt)",
        result.get("transcript", ""),
        "speaker_transcript.txt",
        "text/plain",
        use_container_width=True,
    )
    st.warning("Review speaker names, decisions, owners, and deadlines before sharing this brief.")
