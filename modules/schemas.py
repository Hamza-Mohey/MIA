"""Dependency-free schemas and validation helpers for model output."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\n", " ").split()).strip()


def clip(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


@dataclass
class ActionItem:
    task: str
    owner: str | None = None
    deadline: str | None = None
    evidence: str = ""
    confidence: float = 0.0
    source: str = "transcript"
    evidence_segment_ids: list[str] = field(default_factory=list)
    evidence_start_seconds: float | None = None
    evidence_end_seconds: float | None = None
    evidence_status: str = "unlinked"
    review_status: str = "draft"
    commitment_status: str = "confirmed"

    @classmethod
    def from_value(cls, value: Any) -> ActionItem | None:
        if isinstance(value, Mapping):
            task = clean_text(value.get("task") or value.get("action") or value.get("description"))
            if not task:
                return None
            owner = clean_text(value.get("owner")) or None
            deadline = clean_text(value.get("deadline")) or None
            return cls(
                task=task,
                owner=None if owner in {"Unclear", "Unknown", "N/A"} else owner,
                deadline=None if deadline in {"Unclear", "Unknown", "N/A"} else deadline,
                evidence=clean_text(value.get("evidence")),
                confidence=clip(value.get("confidence"), 0.70),
                source=clean_text(value.get("source")) or "transcript",
                evidence_segment_ids=[
                    clean_text(segment_id)
                    for segment_id in as_list(
                        value.get("evidence_segment_ids") or value.get("source_segment_ids")
                    )
                    if clean_text(segment_id)
                ],
                evidence_start_seconds=value.get("evidence_start_seconds"),
                evidence_end_seconds=value.get("evidence_end_seconds"),
                evidence_status=clean_text(value.get("evidence_status")) or "unlinked",
                review_status=clean_text(value.get("review_status")) or "draft",
                commitment_status=(
                    clean_text(value.get("commitment_status") or value.get("status"))
                    or "confirmed"
                ),
            )
        task = clean_text(value)
        return cls(task=task, confidence=0.60) if task else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionItem:
    text: str
    evidence: str = ""
    confidence: float = 0.0
    source: str = "transcript"
    evidence_segment_ids: list[str] = field(default_factory=list)
    evidence_start_seconds: float | None = None
    evidence_end_seconds: float | None = None
    evidence_status: str = "unlinked"
    review_status: str = "draft"

    @classmethod
    def from_value(cls, value: Any) -> DecisionItem | None:
        if isinstance(value, Mapping):
            text = clean_text(value.get("text") or value.get("decision") or value.get("name"))
            if not text:
                return None
            return cls(
                text=text,
                evidence=clean_text(value.get("evidence")),
                confidence=clip(value.get("confidence"), 0.70),
                source=clean_text(value.get("source")) or "transcript",
                evidence_segment_ids=[
                    clean_text(segment_id)
                    for segment_id in as_list(
                        value.get("evidence_segment_ids") or value.get("source_segment_ids")
                    )
                    if clean_text(segment_id)
                ],
                evidence_start_seconds=value.get("evidence_start_seconds"),
                evidence_end_seconds=value.get("evidence_end_seconds"),
                evidence_status=clean_text(value.get("evidence_status")) or "unlinked",
                review_status=clean_text(value.get("review_status")) or "draft",
            )
        text = clean_text(value)
        return cls(text=text, confidence=0.60) if text else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimItem:
    text: str
    evidence_segment_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence: str = ""
    evidence_start_seconds: float | None = None
    evidence_end_seconds: float | None = None
    evidence_status: str = "unlinked"

    @classmethod
    def from_value(cls, value: Any) -> ClaimItem | None:
        if isinstance(value, Mapping):
            text = clean_text(
                value.get("text")
                or value.get("topic")
                or value.get("point")
                or value.get("question")
                or value.get("risk")
                or value.get("name")
            )
            if not text:
                return None
            return cls(
                text=text,
                evidence_segment_ids=[
                    clean_text(segment_id)
                    for segment_id in as_list(
                        value.get("evidence_segment_ids") or value.get("source_segment_ids")
                    )
                    if clean_text(segment_id)
                ],
                confidence=clip(value.get("confidence"), 0.70),
                evidence=clean_text(value.get("evidence")),
                evidence_start_seconds=value.get("evidence_start_seconds"),
                evidence_end_seconds=value.get("evidence_end_seconds"),
                evidence_status=clean_text(value.get("evidence_status")) or "unlinked",
            )
        text = clean_text(value)
        return cls(text=text, confidence=0.60) if text else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _claims(value: Any) -> list[dict[str, Any]]:
    return [
        parsed.to_dict()
        for item in as_list(value)
        if (parsed := ClaimItem.from_value(item)) is not None
    ]


@dataclass
class TranscriptAnalysisResult:
    meeting_summary: str = ""
    participants: list[Any] = field(default_factory=list)
    topics: list[Any] = field(default_factory=list)
    key_points: list[Any] = field(default_factory=list)
    requirements: list[Any] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    action_items: list[dict[str, Any]] = field(default_factory=list)
    risks: list[Any] = field(default_factory=list)
    open_questions: list[Any] = field(default_factory=list)
    assumptions: list[Any] = field(default_factory=list)
    missing_information: list[Any] = field(default_factory=list)
    uncertainties: list[Any] = field(default_factory=list)
    transcript_quality_score: float = 0.95
    transcript_quality_basis: str = "uploaded transcript heuristic"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> TranscriptAnalysisResult:
        raw = raw or {}
        actions = []
        for item in as_list(raw.get("action_items")):
            parsed = ActionItem.from_value(item)
            if parsed:
                actions.append(parsed.to_dict())
        decisions = []
        for item in as_list(raw.get("decisions")):
            parsed = DecisionItem.from_value(item)
            if parsed:
                decisions.append(parsed.to_dict())
        return cls(
            meeting_summary=clean_text(raw.get("meeting_summary")),
            participants=as_list(raw.get("participants")),
            topics=_claims(raw.get("topics")),
            key_points=_claims(raw.get("key_points")),
            requirements=_claims(raw.get("requirements")),
            decisions=decisions,
            action_items=actions,
            risks=_claims(raw.get("risks")),
            open_questions=_claims(raw.get("open_questions")),
            assumptions=as_list(raw.get("assumptions")),
            missing_information=as_list(
                raw.get("missing_information") or raw.get("unclear_points")
            ),
            uncertainties=as_list(raw.get("uncertainties")),
            transcript_quality_score=clip(raw.get("transcript_quality_score"), 0.95),
            transcript_quality_basis=clean_text(raw.get("transcript_quality_basis"))
            or "uploaded transcript heuristic",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
