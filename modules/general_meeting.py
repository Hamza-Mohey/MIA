"""Deterministic context and report rendering for general meetings."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from modules.config import TEXT_MODEL_NAME, WHISPER_MODEL_NAME
from modules.evidence import link_analysis_evidence
from modules.schemas import TranscriptAnalysisResult, as_list, clean_text


def _unique(items: list[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for item in items:
        key = clean_text(
            item
            if not isinstance(item, dict)
            else item.get("task")
            or item.get("text")
            or item.get("speaker")
            or item.get("name")
            or item
        )
        normalized = key.casefold()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(item)
    return result


def _item_text(item: Any) -> str:
    if isinstance(item, dict):
        return clean_text(item.get("task") or item.get("text") or item.get("speaker"))
    return clean_text(item)


def _known_participant(item: Any) -> bool:
    speaker = _item_text(item).casefold()
    return bool(speaker) and speaker not in {
        "unknown",
        "unknown speaker",
        "speaker_unknown",
        "speaker unknown",
    }


def _summary_duplicate(left: str, right: str) -> bool:
    left_tokens = set(left.casefold().split())
    right_tokens = set(right.casefold().split())
    if not left_tokens or not right_tokens:
        return False
    return len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens)) >= 0.7


def build_grounded_summary(analysis: dict[str, Any], max_words: int = 100) -> str:
    """Summarise grounded purpose, constraints, outcomes, and follow-up work."""
    topic_items = [
        item
        for item in analysis.get("topics", [])
        if isinstance(item, dict) and _item_text(item)
    ]
    topic_items.sort(
        key=lambda item: (
            -float(item.get("importance_score", 0.0)),
            item.get("evidence_start_seconds") is None,
            item.get("evidence_start_seconds") or 0.0,
        )
    )
    purpose_item = next(
        (
            item
            for item in topic_items
            if item.get("source")
            in {"deterministic_purpose_recovery", "deterministic_presentation_recovery"}
        ),
        topic_items[0] if topic_items else None,
    )
    purpose = _item_text(purpose_item) if purpose_item else ""
    key_points = [
        item
        for item in analysis.get("key_points", [])
        if isinstance(item, dict) and _item_text(item)
    ]
    key_points.sort(
        key=lambda item: (
            -float(item.get("importance_score", 0.0)),
            item.get("evidence_start_seconds") is None,
            item.get("evidence_start_seconds") or 0.0,
        )
    )
    sentences: list[str] = []
    if purpose:
        sentences.append(f"The meeting focused on {purpose}.")

    requirements = [
        _item_text(item).rstrip(".")
        for item in analysis.get("requirements", [])
        if isinstance(item, dict)
        and item.get("evidence_status") == "linked"
        and _item_text(item)
    ][:5]
    if requirements:
        sentences.append(f"Key requirements: {'; '.join(requirements)}.")

    decisions = [
        _item_text(item).rstrip(".")
        for item in analysis.get("decisions", [])
        if isinstance(item, dict)
        and item.get("evidence_status") == "linked"
        and _item_text(item)
    ][:2]
    if decisions:
        sentences.append(f"Decisions: {'; '.join(decisions)}.")

    assignments = [
        item
        for item in analysis.get("action_items", [])
        if isinstance(item, dict)
        and item.get("evidence_status") == "linked"
        and item.get("commitment_status") == "confirmed"
        and _item_text(item)
    ]
    assignments.sort(
        key=lambda item: (
            item.get("source") != "deterministic_role_assignment",
            item.get("evidence_start_seconds") is None,
            item.get("evidence_start_seconds") or 0.0,
        )
    )
    assignment_text = [
        f"{clean_text(item.get('owner'))}: {_item_text(item).rstrip('.')}"
        if clean_text(item.get("owner"))
        else _item_text(item).rstrip(".")
        for item in assignments[:3]
    ]
    if assignment_text:
        sentences.append(f"Follow-up work: {'; '.join(assignment_text)}.")

    selected_points: list[str] = []
    for item in key_points:
        text = _item_text(item)
        if any(
            _summary_duplicate(text, existing)
            for existing in [purpose, *requirements, *decisions, *assignment_text, *selected_points]
            if existing
        ):
            continue
        selected_points.append(text)
        if len(selected_points) == 2:
            break
    sentences.extend(
        text if text.endswith((".", "?", "!")) else f"{text}."
        for text in selected_points
    )
    if not sentences:
        return "No sufficiently grounded summary could be constructed from the transcript."
    return " ".join(" ".join(sentences).split()[:max_words])


def build_general_context(
    text_analysis: dict[str, Any],
    *,
    diarization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text = TranscriptAnalysisResult.from_mapping(text_analysis)
    questions = _unique(text.open_questions)
    participants = [item for item in _unique(text.participants) if _known_participant(item)]
    if not participants and diarization:
        participants = [
            {"speaker": speaker, "name": None, "role": None}
            for speaker in diarization.get("speakers", [])
            if _known_participant(speaker)
        ]
    speaker_count = (diarization or {}).get("speaker_count")
    if speaker_count is None:
        speaker_count = len(
            {
                clean_text(item.get("speaker") if isinstance(item, dict) else item)
                for item in participants
                if _known_participant(item)
            }
        )
    return {
        "schema_version": "3.0",
        "meeting_mode": "general",
        "meeting_summary": text.meeting_summary or "No reliable meeting summary was extracted.",
        "participants": participants,
        "topics": _unique(text.topics),
        "key_points": _unique(text.key_points),
        "requirements": _unique(text.requirements),
        "decisions": _unique(text.decisions),
        "action_items": _unique(text.action_items),
        "risks": _unique(text.risks),
        "open_questions": questions,
        "assumptions": _unique(text.assumptions),
        "missing_information": _unique(text.missing_information),
        "uncertainties": _unique(text.uncertainties),
        "diarization": diarization or {},
        "quality": {
            "transcript_score": text.transcript_quality_score,
            "transcript_basis": text.transcript_quality_basis,
            "speaker_count": speaker_count,
            "diarization_model": (diarization or {}).get("model"),
        },
    }


def _section(title: str, values: Any, empty: str) -> str:
    lines = [f"## {title}", ""]
    items = as_list(values)
    if not items:
        lines.append(empty)
    else:
        for item in items:
            if isinstance(item, dict):
                if title == "Participants":
                    speaker = clean_text(item.get("speaker"))
                    name = clean_text(item.get("name"))
                    role = clean_text(item.get("role"))
                    text = name or speaker
                    participant_details = [value for value in (role, speaker if name else "") if value]
                    if participant_details:
                        text += f" ({'; '.join(participant_details)})"
                else:
                    text = clean_text(
                        item.get("task") or item.get("text") or item.get("speaker") or item
                    )
                details = [
                    f"owner: {item['owner']}" if item.get("owner") else "",
                    f"deadline: {item['deadline']}" if item.get("deadline") else "",
                    (
                        f"status: {item['commitment_status']}"
                        if item.get("commitment_status")
                        else ""
                    ),
                ]
                details = [detail for detail in details if detail]
                lines.append(f"- {text}" + (f" ({'; '.join(details)})" if details else ""))
                evidence_ids = item.get("evidence_segment_ids") or []
                if evidence_ids:
                    lines.append(f"  - Evidence: {', '.join(evidence_ids)}")
                if item.get("review_status") == "confirmed":
                    lines.append("  - Review status: confirmed")
            else:
                lines.append(f"- {clean_text(item)}")
    return "\n".join(lines)


def generate_general_report(context: dict[str, Any]) -> str:
    lines = [
        "# General Meeting Intelligence Report",
        "",
        "## Summary",
        "",
        clean_text(context.get("meeting_summary")) or "No reliable summary was extracted.",
        "",
    ]
    for title, key, empty in (
        ("Participants", "participants", "No participants were confidently identified."),
        ("Topics", "topics", "No clear topics were extracted."),
        ("Key Points", "key_points", "No key points were extracted."),
        ("Requirements and Constraints", "requirements", "No explicit requirements were extracted."),
        ("Decisions", "decisions", "No explicit decisions were extracted."),
        ("Action Items", "action_items", "No explicit action items were extracted."),
        ("Risks and Blockers", "risks", "No explicit risks or blockers were extracted."),
        ("Open Questions", "open_questions", "No open questions were extracted."),
        ("Uncertainties", "uncertainties", "No additional uncertainties were recorded."),
    ):
        lines.extend([_section(title, context.get(key), empty), ""])
    lines.extend(
        [
            "---",
            "",
            "AI-generated draft. Verify decisions, owners, deadlines, and speaker names before distribution.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def run_general_pipeline(
    transcript_text: str,
    *,
    diarization: dict[str, Any] | None = None,
    transcript_quality: float = 0.95,
    quality_basis: str = "uploaded transcript heuristic",
    source_segments: list[dict[str, Any]] | None = None,
    text_analyser: Callable[..., dict[str, Any]] | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    """Run the general meeting path without OCR or software classification."""
    if not isinstance(transcript_text, str) or not transcript_text.strip():
        raise ValueError("A transcript is required for general meeting analysis.")
    if text_analyser is None:
        from modules.text_analysis import analyse_transcript

        text_analyser = analyse_transcript
    if progress_callback:
        progress_callback("Extracting meeting intelligence", 0.15)
    started = time.perf_counter()
    analysis = text_analyser(
        transcript_text,
        transcript_quality=transcript_quality,
        quality_basis=quality_basis,
    )
    analysis_runtime = analysis.pop("_runtime", {})
    measured_analysis_seconds = time.perf_counter() - started
    worker_analysis_seconds = float(
        analysis_runtime.get("subprocess_seconds")
        or analysis_runtime.get("total_seconds")
        or 0.0
    )
    analysis_seconds = round(max(measured_analysis_seconds, worker_analysis_seconds), 3)
    if progress_callback:
        progress_callback("Building evidence-linked report", 0.85)
    postprocess_started = time.perf_counter()
    if source_segments:
        analysis["model_meeting_summary"] = analysis.get("meeting_summary", "")
        analysis = link_analysis_evidence(analysis, source_segments)
        analysis["meeting_summary"] = build_grounded_summary(analysis)
        if not analysis.get("participants"):
            analysis["participants"] = [
                {"speaker": speaker, "name": None, "role": None}
                for speaker in dict.fromkeys(
                    clean_text(segment.get("speaker")) for segment in source_segments
                )
                if _known_participant(speaker)
            ]
    context = build_general_context(analysis, diarization=diarization)
    report = generate_general_report(context)
    postprocess_seconds = round(time.perf_counter() - postprocess_started, 3)
    if progress_callback:
        progress_callback("Complete", 1.0)
    return {
        "schema_version": "3.0",
        "text_analysis": analysis,
        "meeting_context": context,
        "source_segments": source_segments or [],
        "report_markdown": report,
        "timings": {
            "text_analysis_seconds": analysis_seconds,
            "postprocess_seconds": postprocess_seconds,
            "model_worker": analysis_runtime,
        },
        "models": {
            "transcription": WHISPER_MODEL_NAME,
            "diarization": (diarization or {}).get("model"),
            "meeting_intelligence": TEXT_MODEL_NAME,
        },
    }
