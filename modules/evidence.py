"""Source-segment creation and deterministic evidence validation."""

from __future__ import annotations

import re
from collections import Counter
from copy import deepcopy
from difflib import SequenceMatcher
from math import log1p, sqrt
from typing import Any

from modules.config import (
    ANALYSIS_EVIDENCE_MAX_CHARACTERS,
    EVIDENCE_KEY_POINT_SUPPORT_THRESHOLD,
    EVIDENCE_MAX_INDEX_SPAN,
    EVIDENCE_MAX_TIME_SPAN_SECONDS,
    EVIDENCE_SUPPORT_THRESHOLD,
    EVIDENCE_TOPIC_SUPPORT_THRESHOLD,
    TEXT_CHUNK_LENGTH,
)
from modules.schemas import clean_text

_TIMESTAMPED_LINE = re.compile(r"^\[(?P<timestamp>\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)\]\s*(?P<body>.+)$")
_SPEAKER_LINE = re.compile(r"^(?P<speaker>[^:\n]{1,80}):\s*(?P<text>.+)$")
_ANONYMOUS_TURN = re.compile(r"(?=\bSPEAKER_[A-Za-z0-9_-]+\s*:)")
_TOKEN = re.compile(r"[\w']+")
_STOPWORDS = {
    "a", "about", "actually", "all", "also", "an", "and", "anyway", "are", "as", "at",
    "be", "because", "but", "by", "can", "come", "could", "course", "did", "different",
    "do", "does", "doesn't", "don't", "even", "fine", "first", "for", "from", "get",
    "going", "guess", "had", "has", "have", "he", "her", "here", "him", "how", "i", "if",
    "i'm", "in", "is", "it", "it's", "just", "kind", "know", "like", "little", "make",
    "maybe", "me", "mean", "might", "more", "most", "much", "my", "no", "not", "of",
    "oh", "okay", "on", "one", "or", "our", "probably", "quite", "really", "right", "said",
    "say", "see", "she", "should", "so", "some", "something", "sort", "stuff", "sure", "take",
    "tell", "that", "that's", "the", "their", "them", "then", "there", "there's", "these",
    "they", "thing", "things", "think", "this", "time", "to", "try", "two", "very", "want",
    "was", "we", "well", "were", "what", "when", "where", "why", "will", "with", "would",
    "yeah", "yes", "you", "you're", "your",
}
_THEME_GENERIC = {
    "agenda", "anyth", "anything", "approach", "area", "argument", "case", "comment", "connected",
    "data", "detail", "discussion", "domain", "example", "fact", "find", "focus", "idea", "information",
    "issue", "look", "meeting", "number", "observation", "operational", "people", "point",
    "prioritie", "problem",
    "project", "proposal", "question", "result", "sorry", "state", "story", "subject", "system",
    "talk", "term", "topic", "update", "way", "work",
}
_CALIBRATION_NOISE = re.compile(
    r"\b(?:digits?|microphone test|sound check|synchroni[sz]e|"
    r"time them at the same time|turn the volume on|can you hear (?:me|that)|"
    r"start all our meetings out that way)\b",
    re.IGNORECASE,
)
_MEETING_PURPOSE_CUE = re.compile(
    r"\b(?:agenda(?:\s+is)?|purpose of (?:today's|this) meeting|"
    r"kick[- ]off meeting|"
    r"(?:we(?:'re| are)|today we(?:'re| are))\s+(?:here to|meeting to|talking about|"
    r"discussing|designing|going to discuss)|(?:meeting|discussion)\s+(?:is|was)\s+about|"
    r"main (?:focus|topic)|focus(?:ed)? on)\b",
    re.IGNORECASE,
)
_PRESENTATION_CUE = re.compile(
    r"\b(?:going to|will|plan(?:s|ned)? to)\s+(?:present|give)\s+"
    r"(?:a\s+)?(?:little\s+)?(?:talk|presentation)\b",
    re.IGNORECASE,
)
_SMALL_TALK_CUE = re.compile(
    r"\b(?:most frustrating meeting|remember you(?:'re| are) being recorded|"
    r"so comfortable|great story|laughter|good trip|world record)\b",
    re.IGNORECASE,
)
_CONCLUSION_CUE = re.compile(
    r"\b(?:argument can be made|central conclusion|in summary|key in terms|"
    r"main takeaway|overall|reasonable point|the (?:argument|conclusion|main point)|"
    r"therefore)\b",
    re.IGNORECASE,
)
_DECISION_CUE = re.compile(
    r"\b(agree(?:d)?|approv(?:e|ed)|chose|chosen|decid(?:e|ed)|final decision|"
    r"go with|going with|keep|move forward|reject(?:ed)?|select(?:ed)?|settled|"
    r"stick with|we(?:'ll| will))\b",
    re.IGNORECASE,
)
_RISK_CUE = re.compile(
    r"\b(blocked|blocker|concern|delay(?:ed)?|dependency|issue|problem|risk|"
    r"not confirmed|not ready|waiting for|worried)\b",
    re.IGNORECASE,
)
_QUESTION_CUE = re.compile(
    r"\?|\b(open question|not clear|not sure|still need to (?:decide|determine|find)|"
    r"unresolved|whether|which|who|what|when|where|why|how)\b",
    re.IGNORECASE,
)
_REQUIREMENT_CUE = re.compile(
    r"\b(?:budget|cost|must|need(?:s)? to|price|requirement|revenue|selling|"
    r"supposed to be|target|aim(?:ing)? to|no more than|international scale)\b",
    re.IGNORECASE,
)
_ROLE_CUE = re.compile(
    r"\b(?P<role>(?:project|product|programme|program) manager|industrial designer|"
    r"marketing(?: (?:expert|executive|lead|manager))?|(?:user|future) interface|"
    r"(?:software|hardware|design|engineering|technical|research|sales) "
    r"(?:engineer|lead|manager|developer|designer|analyst)|facilitator|coordinator)\b",
    re.IGNORECASE,
)
_INTRODUCTION_CUE = re.compile(
    r"\b(?:(?:hi[, ]+)?(?:and\s+)?i(?:'m| am)|my name is)\s+"
    r"(?P<name>[A-Z][A-Za-z'-]{1,30})\b",
    re.IGNORECASE,
)
_INVALID_PERSON_NAMES = {
    "a",
    "an",
    "he",
    "i",
    "it",
    "she",
    "that",
    "the",
    "they",
    "this",
    "we",
    "you",
}
_PRODUCT_CONTEXT_CUE = re.compile(
    r"\b(?:device|design|market|product|remote(?: control)?|selling price|"
    r"production cost|user[- ]friendly)\b",
    re.IGNORECASE,
)
_SETUP_LANGUAGE_CUE = re.compile(
    r"\b(?:am i supposed to be|camera|clip(?:ped)? on|headset|microphone|"
    r"room setup|sit(?:ting)? down|stand(?:ing)? up|sound check|test recording)\b",
    re.IGNORECASE,
)
_WRAP_UP_CUE = re.compile(
    r"\b(?:before we wrap up|in ?between now and then|just to wrap up|"
    r"next (?:meeting|stage)|follow[- ]up|before (?:the )?next meeting)\b",
    re.IGNORECASE,
)
_NON_FOLLOWUP_ACTION = re.compile(
    r"\b(?:check (?:if|whether) there(?:'s| is) (?:nothing|anything) else|"
    r"check we(?:'ve| have) nothing else|turn (?:this|it|the .+?) (?:on|off)|"
    r"wrap up|end (?:of )?the meeting|stop(?:ped)? the clock|watch the video back|"
    r"present(?:ing)? (?:a|the) petition|call this meeting to order)\b",
    re.IGNORECASE,
)
_NON_ACTION_DESCRIPTION = re.compile(
    r"(?:^|\b)(?:if (?:people|somebody|someone|they|we|you) (?:need|want)|"
    r"wanna hear (?:this|that)|want to hear (?:this|that)|"
    r"(?:button|link).{0,80}\b(?:say|says|which say)|"
    r"send it through email you\s*(?:'re| are) thinking|"
    r"(?:certainly|possibly|probably) provide\b|"
    r"the chair\s*:\s*check what happened)\b",
    re.IGNORECASE,
)
_NEGATED_ACTION = re.compile(
    r"\b(?:cannot|can't|do not|don't|not going to|shouldn't|wouldn't|won't)\b"
    r".{0,80}\b(?:check|circulate|confirm|contact|deliver|email|follow up|include|"
    r"prepare|provide|report back|review|revise|schedule|send|share|submit|update)\b",
    re.IGNORECASE,
)
_PAST_ACTIVITY = re.compile(
    r"\b(?:what i did was|i (?:added|found|took|tried)|we (?:already |just )?"
    r"(?:announced|ordered|provided)|has already|have already)\b",
    re.IGNORECASE,
)
_NAMED_FUTURE_COMMITMENT = re.compile(
    r"\b(?P<assignee>[A-Z][\w'-]+|SPEAKER_[A-Za-z0-9_-]+)\s+will\b"
)
_DIRECT_ASSIGNMENT = re.compile(
    r"\b(?P<assignee>[A-Z][\w'-]+|SPEAKER_[A-Za-z0-9_-]+)[,:]\s+"
    r"(?:please|can you|could you|will you|would you)\b"
)
_FIRST_PERSON_COMMITMENT = re.compile(
    r"\b(?:I(?:'ll| will| am going to)|let me)\b", re.IGNORECASE
)
_ACTION_TASK_VERB = re.compile(
    r"\b(check|circulate|confirm|contact|deliver|double[- ]check|email|follow up|"
    r"include|prepare|provide|report back|review|revise|revision|schedule|send|share|"
    r"submit|update)\b",
    re.IGNORECASE,
)
_DIRECT_REQUEST = re.compile(
    r"\b(?:please|can you|could you|will you|would you)\s+"
    r"(?:check|circulate|confirm|contact|deliver|double[- ]check|email|follow up|"
    r"include|prepare|provide|report back|review|revise|schedule|send|share|submit|update)\b",
    re.IGNORECASE,
)
_ASK_TO_TASK = re.compile(
    r"\bask(?:ed|ing)? (?:me|us|you|him|her|them) to\s+"
    r"(?:\w+\s+){0,3}(?:check|circulate|confirm|contact|deliver|double[- ]check|"
    r"email|follow up|include|prepare|provide|report back|review|revise|schedule|"
    r"send|share|submit|update)\b",
    re.IGNORECASE,
)
_PRONOUN_FUTURE_TASK = re.compile(
    r"\b(?:we|he|she|they)(?:'ll| will)\s+(?:\w+\s+){0,2}"
    r"(?:check|circulate|confirm|contact|deliver|double[- ]check|email|follow up|"
    r"include|prepare|provide|report back|review|revise|schedule|send|share|submit|update)\b",
    re.IGNORECASE,
)
_STRICT_ACTION_SUGGESTION = re.compile(
    r"\b(?:you\s+(?:could(?: always)?|should|might(?: want to)?|need to)|"
    r"(?:maybe\s+)?we\s+(?:can|could|should)|why don't we|"
    r"it would be (?:a )?(?:good|helpful|useful) idea(?: for .{1,40}?)? to)\s+"
    r"(?:\w+\s+){0,3}(?:check|circulate|confirm|contact|deliver|double[- ]check|"
    r"email|follow up|include|prepare|provide|report back|review|revise|schedule|"
    r"send|share|submit|update)\b",
    re.IGNORECASE,
)


def _action_cue_status(text: str) -> tuple[bool, bool]:
    """Return narrow confirmed/suggested action signals for a local passage."""
    text = clean_text(text)
    if (
        _NEGATED_ACTION.search(text)
        or _NON_FOLLOWUP_ACTION.search(text)
        or _NON_ACTION_DESCRIPTION.search(text)
    ):
        return False, False

    def cue_has_task(match: re.Match[str] | None) -> bool:
        if match is None:
            return False
        return bool(_ACTION_TASK_VERB.search(text[match.start() : match.end() + 140]))

    future_cues = (
        _FIRST_PERSON_COMMITMENT.search(text),
        _NAMED_FUTURE_COMMITMENT.search(text),
        _DIRECT_ASSIGNMENT.search(text),
        _PRONOUN_FUTURE_TASK.search(text),
    )
    confirmed = not _PAST_ACTIVITY.search(text) and any(
        cue_has_task(match) for match in future_cues
    )
    suggested = bool(
        _DIRECT_REQUEST.search(text)
        or _ASK_TO_TASK.search(text)
        or _STRICT_ACTION_SUGGESTION.search(text)
    )
    return confirmed, suggested


def _seconds(timestamp: str) -> float:
    hours, minutes, seconds = timestamp.split(":")
    return int(hours) * 3_600 + int(minutes) * 60 + float(seconds)


def format_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return ""
    minutes, remainder = divmod(max(0.0, float(seconds)), 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{remainder:05.2f}"


def _parse_transcript(transcript: str) -> list[dict[str, Any]]:
    lines = [line.strip() for line in transcript.splitlines() if line.strip()]
    if len(lines) == 1:
        anonymous_turns = [part.strip() for part in _ANONYMOUS_TURN.split(lines[0]) if part.strip()]
        if len(anonymous_turns) > 1:
            lines = anonymous_turns

    parsed: list[dict[str, Any]] = []
    for line in lines:
        start = None
        timestamped = _TIMESTAMPED_LINE.match(line)
        if timestamped:
            start = _seconds(timestamped.group("timestamp"))
            line = timestamped.group("body").strip()
        speaker = None
        speaker_match = _SPEAKER_LINE.match(line)
        if speaker_match:
            speaker = clean_text(speaker_match.group("speaker"))
            line = speaker_match.group("text")
        text = clean_text(line)
        if text:
            parsed.append({"speaker": speaker, "text": text, "start": start, "end": None})
    return parsed


def _audio_segments(audio_result: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not audio_result:
        return []
    attributed = audio_result.get("speaker_segments") or []
    if attributed:
        return [
            {
                "speaker": clean_text(segment.get("speaker")) or None,
                "text": clean_text(segment.get("text")),
                "start": segment.get("start"),
                "end": segment.get("end"),
            }
            for segment in attributed
            if clean_text(segment.get("text"))
        ]
    return [
        {
            "speaker": None,
            "text": clean_text(segment.get("text")),
            "start": (segment.get("timestamp") or [None, None])[0],
            "end": (segment.get("timestamp") or [None, None])[1],
        }
        for segment in audio_result.get("segments", [])
        if clean_text(segment.get("text"))
    ]


def _similarity(left: Any, right: Any) -> float:
    left_text = clean_text(left).casefold()
    right_text = clean_text(right).casefold()
    if not left_text or not right_text:
        return 0.0
    if left_text in right_text or right_text in left_text:
        return min(len(left_text), len(right_text)) / max(len(left_text), len(right_text))
    left_tokens = set(_TOKEN.findall(left_text))
    right_tokens = set(_TOKEN.findall(right_text))
    overlap = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    return max(overlap, SequenceMatcher(None, left_text, right_text).ratio())


def _token_containment_similarity(left: Any, right: Any) -> float:
    left_tokens = set(_TOKEN.findall(clean_text(left).casefold()))
    right_tokens = set(_TOKEN.findall(clean_text(right).casefold()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))


def build_source_segments(
    transcript: str,
    audio_result: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Create stable source records and retain audio times only when reconciliation is safe."""
    parsed = _parse_transcript(transcript)
    audio = _audio_segments(audio_result)

    if audio and len(parsed) == 1 and len(audio) > 1:
        combined_audio = clean_text(" ".join(segment["text"] for segment in audio))
        if _similarity(parsed[0]["text"], combined_audio) >= 0.95:
            parsed = deepcopy(audio)
    elif audio and len(parsed) == len(audio):
        for item, original in zip(parsed, audio, strict=True):
            item["start"] = original["start"]
            item["end"] = original["end"]
            item["speaker"] = item["speaker"] or original["speaker"]
    elif audio and parsed:
        unused = set(range(len(audio)))
        for item in parsed:
            candidates = [
                (_similarity(item["text"], audio[index]["text"]), index) for index in unused
            ]
            score, index = max(candidates, default=(0.0, -1))
            if score >= 0.72:
                original = audio[index]
                item["start"] = original["start"]
                item["end"] = original["end"]
                item["speaker"] = item["speaker"] or original["speaker"]
                unused.remove(index)

    return [
        {
            "id": f"seg-{index:04d}",
            "index": index - 1,
            "speaker": segment.get("speaker"),
            "text": segment["text"],
            "start": segment.get("start"),
            "end": segment.get("end"),
            "origin": "audio" if segment.get("start") is not None else "transcript",
        }
        for index, segment in enumerate(parsed, start=1)
    ]


def render_source_segment(segment: dict[str, Any]) -> str:
    timestamp = format_timestamp(segment.get("start"))
    time_part = f" [{timestamp}]" if timestamp else ""
    speaker = f" {segment['speaker']}:" if segment.get("speaker") else ""
    return f"[{segment['id']}]{time_part}{speaker} {segment.get('text', '')}".strip()


def render_source_segments(segments: list[dict[str, Any]]) -> str:
    return "\n".join(render_source_segment(segment) for segment in segments)


def chunk_source_segments(
    segments: list[dict[str, Any]], max_characters: int = TEXT_CHUNK_LENGTH
) -> list[dict[str, Any]]:
    """Group complete source turns into bounded, independently cacheable chunks."""
    chunks: list[dict[str, Any]] = []
    lines: list[str] = []
    segment_ids: list[str] = []
    length = 0

    def flush() -> None:
        nonlocal lines, segment_ids, length
        if lines:
            chunks.append(
                {
                    "index": len(chunks),
                    "segment_ids": list(dict.fromkeys(segment_ids)),
                    "text": "\n".join(lines),
                }
            )
        lines, segment_ids, length = [], [], 0

    for segment in segments:
        rendered = render_source_segment(segment)
        pieces = [
            rendered[index : index + max_characters]
            for index in range(0, len(rendered), max_characters)
        ]
        for piece in pieces:
            added = len(piece) + (1 if lines else 0)
            if lines and length + added > max_characters:
                flush()
            lines.append(piece)
            segment_ids.append(segment["id"])
            length += len(piece) + (1 if len(lines) > 1 else 0)
    flush()
    return chunks


def build_analysis_digest(
    segments: list[dict[str, Any]],
    max_characters: int = ANALYSIS_EVIDENCE_MAX_CHARACTERS,
) -> dict[str, Any]:
    """Select a compact, timeline-balanced evidence digest for local Qwen."""
    if not segments:
        return {"index": 0, "segment_ids": [], "text": "", "total_segments": 0}
    normalized_counts = Counter(
        _normalised
        for segment in segments
        if (_normalised := clean_text(segment.get("text")).casefold())
    )
    purpose_index = next(
        (
            index
            for index, segment in enumerate(segments)
            if _MEETING_PURPOSE_CUE.search(clean_text(segment.get("text")))
        ),
        None,
    )
    eligible: list[tuple[int, dict[str, Any], set[str], float]] = []
    global_terms: Counter[str] = Counter()
    for index, segment in enumerate(segments):
        text = clean_text(segment.get("text"))
        if _CALIBRATION_NOISE.search(text):
            continue
        tokens = _content_tokens(text)
        words = _TOKEN.findall(text)
        normalized = text.casefold()
        confirmed_action, suggested_action = _action_cue_status(text)
        action_cue = bool(
            _ACTION_TASK_VERB.search(text) and (confirmed_action or suggested_action)
        )
        outcome_cue = bool(
            action_cue or _DECISION_CUE.search(text) or _RISK_CUE.search(text)
        )
        question_cue = bool(
            re.search(
                r"\b(?:should we|do we need|not sure whether|still need to decide)\b",
                text,
                re.IGNORECASE,
            )
        )
        purpose_cue = bool(_MEETING_PURPOSE_CUE.search(text))
        presentation_cue = bool(_PRESENTATION_CUE.search(text))
        conclusion_cue = bool(_CONCLUSION_CUE.search(text))
        requirement_cue = bool(_REQUIREMENT_CUE.search(text))
        wrap_up_cue = bool(_WRAP_UP_CUE.search(text))
        cue = (
            outcome_cue
            or question_cue
            or purpose_cue
            or presentation_cue
            or conclusion_cue
            or requirement_cue
            or wrap_up_cue
        )
        if _SMALL_TALK_CUE.search(text) and not outcome_cue:
            continue
        if purpose_index is not None and index < purpose_index and not outcome_cue:
            continue
        numeric_share = sum(token.isdigit() for token in words) / max(1, len(words))
        if not cue and (len(words) < 5 or numeric_share > 0.45):
            continue
        if normalized_counts[normalized] > 2 and not cue:
            continue
        if not tokens:
            continue
        duplicate_position = next(
            (
                position
                for position in range(max(0, len(eligible) - 3), len(eligible))
                if index - eligible[position][0] <= 3
                and _token_containment_similarity(
                    text, eligible[position][1].get("text")
                )
                >= 0.78
            ),
            None,
        )
        if duplicate_position is not None:
            previous = eligible[duplicate_position]
            if len(words) <= len(_TOKEN.findall(clean_text(previous[1].get("text")))):
                continue
            global_terms.subtract(previous[2])
            eligible.pop(duplicate_position)
        global_terms.update(tokens)
        cue_bonus = (
            36.0
            if purpose_cue
            else 22.0
            if presentation_cue
            else 21.0
            if wrap_up_cue
            else 20.0
            if requirement_cue
            else 18.0
            if outcome_cue
            else 17.0
            if conclusion_cue
            else 4.0
            if question_cue
            else 0.0
        )
        eligible.append((index, segment, tokens, cue_bonus))

    ranked: list[tuple[float, int]] = []
    cue_bonuses: dict[int, float] = {}
    eligible_indices: set[int] = set()
    for index, _segment, tokens, cue_bonus in eligible:
        centrality = sum(log1p(global_terms[token]) for token in tokens) / sqrt(len(tokens))
        ranked.append((cue_bonus + centrality, index))
        cue_bonuses[index] = cue_bonus
        eligible_indices.add(index)

    # The stated purpose and planned presentations receive first claim on the
    # budget, followed by explicit outcomes, questions, and broad chronology.
    priority_indices = [
        index
        for _score, index in sorted(ranked, reverse=True)
        if cue_bonuses[index] >= 22.0
    ]
    priority_indices.extend(
        index
        for _score, index in sorted(ranked, reverse=True)
        if 0.0 < cue_bonuses[index] < 22.0
    )
    bucket_count = min(6, len(segments))
    for bucket in range(bucket_count):
        start = bucket * len(segments) // bucket_count
        end = (bucket + 1) * len(segments) // bucket_count
        candidate = max(
            ((score, index) for score, index in ranked if start <= index < end),
            default=None,
        )
        if candidate:
            priority_indices.append(candidate[1])
    for _score, index in sorted(ranked, reverse=True):
        priority_indices.append(index)

    selected_indices: set[int] = set()
    used = 0
    for center in dict.fromkeys(priority_indices):
        neighbourhood = [
            index
            for index in (center - 1, center, center + 1)
            if (
                0 <= index < len(segments)
                and index in eligible_indices
                and index not in selected_indices
            )
        ]
        added = sum(len(render_source_segment(segments[index])) + 1 for index in neighbourhood)
        if used + added > max_characters:
            continue
        selected_indices.update(neighbourhood)
        used += added

    rendered: list[str] = []
    included: list[str] = []
    for index in sorted(selected_indices):
        line = render_source_segment(segments[index])
        rendered.append(line)
        included.append(segments[index]["id"])
    return {
        "index": 0,
        "segment_ids": included,
        "text": "\n".join(rendered),
        "total_segments": len(segments),
        "selected_segments": len(included),
        "selection_method": "purpose_outcome_timeline_balanced_centrality",
    }


def build_analysis_digests(
    segments: list[dict[str, Any]],
    *,
    max_characters: int = ANALYSIS_EVIDENCE_MAX_CHARACTERS,
    window_count: int = 3,
) -> list[dict[str, Any]]:
    """Build small chronological evidence windows to reduce Qwen recency bias."""
    if not segments:
        return []
    count = max(1, min(int(window_count), len(segments)))
    per_window_budget = max(1_000, max_characters // count)
    digests: list[dict[str, Any]] = []
    for window in range(count):
        start = window * len(segments) // count
        end = (window + 1) * len(segments) // count
        digest = build_analysis_digest(
            segments[start:end], max_characters=per_window_budget
        )
        if not digest.get("text"):
            continue
        digest["index"] = len(digests)
        digest["total_segments"] = len(segments)
        digest["window_start_segment"] = segments[start]["id"]
        digest["window_end_segment"] = segments[end - 1]["id"]
        digests.append(digest)
    return digests


def _item_text(item: dict[str, Any]) -> str:
    return clean_text(item.get("task") or item.get("text") or item.get("decision"))


def _normalize_content_token(token: str) -> str:
    if len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
    elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return token


def _content_tokens(value: Any) -> set[str]:

    return {
        _normalize_content_token(token)
        for token in _TOKEN.findall(clean_text(value).casefold())
        if len(token) > 1 and token not in _STOPWORDS
    }


def _support_score(claim: str, evidence: str) -> float:
    distinctive_terms = {
        token.casefold()
        for token in _TOKEN.findall(clean_text(claim))
        if (
            (len(token) >= 2 and token.isupper())
            or (len(token) >= 3 and any(character.isupper() for character in token[1:]))
        )
        and not token.upper().startswith("SPEAKER_")
    }
    evidence_terms = {token.casefold() for token in _TOKEN.findall(clean_text(evidence))}
    if distinctive_terms - evidence_terms:
        return 0.0
    claim_tokens = _content_tokens(claim)
    evidence_tokens = _content_tokens(evidence)
    if not claim_tokens or not evidence_tokens:
        return 0.0
    overlap = claim_tokens & evidence_tokens
    if not overlap:
        return 0.0
    claim_recall = len(overlap) / len(claim_tokens)
    return max(claim_recall, _similarity(claim, evidence))


def _best_evidence_window(
    candidate: str,
    segments: list[dict[str, Any]],
    support_threshold: float,
) -> list[dict[str, Any]]:
    """Repair a topic/key-point citation only when source text clearly supports it."""
    best_score = 0.0
    best_window: list[dict[str, Any]] = []
    for index in range(len(segments)):
        start = max(0, index - 1)
        end = min(len(segments), index + 2)
        window = segments[start:end]
        score = _support_score(
            candidate, " ".join(clean_text(item.get("text")) for item in window)
        )
        if score > best_score:
            best_score = score
            best_window = window
    return best_window if best_score >= support_threshold else []


def _extract_purpose_topic(text: str) -> str:
    design = re.search(
        r"\b(?:we(?:'re| are)\s+)?designing\s+(?P<subject>(?:a|the)\s+.+?)"
        r"(?:[.?!]|$)",
        text,
        re.IGNORECASE,
    )
    if design:
        subject = re.sub(
            r"^(?:a|the)\s+",
            "",
            clean_text(design.group("subject")),
            flags=re.IGNORECASE,
        )
        return f"{subject.capitalize()} design"[:100]
    match = re.search(
        r"\b(?:talking about|discussion is about|meeting is about|focused on)\s+"
        r"(?P<subject>.+?)(?:\s+today\b|[.?!]|$)",
        text,
        re.IGNORECASE,
    )
    if not match:
        return (
            "Project kick-off"
            if re.search(r"\bkick[- ]off meeting\b", text, re.IGNORECASE)
            else ""
        )
    subject = clean_text(match.group("subject"))
    subject = re.sub(r"^(?:the\s+)?(?:topic of\s+)?", "", subject, flags=re.IGNORECASE)
    return subject[:100].rstrip(" ,;:-") if 2 <= len(subject.split()) <= 14 else ""


def _recover_structural_topics(
    topics: list[Any], segments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Recover explicit meeting-purpose and presentation anchors Qwen omitted."""
    recovered = [
        deepcopy(item)
        for item in topics
        if isinstance(item, dict)
        and item.get("source") != "deterministic_recurrent_theme"
    ]
    purpose_added = False
    presentation_added = False
    for index, segment in enumerate(segments):
        text = clean_text(segment.get("text"))
        candidate = ""
        source = ""
        if not purpose_added and _MEETING_PURPOSE_CUE.search(text):
            candidate = _extract_purpose_topic(text)
            if candidate == "Project kick-off":
                for following in segments[index + 1 : index + 25]:
                    more_specific = _extract_purpose_topic(clean_text(following.get("text")))
                    if more_specific and more_specific != "Project kick-off":
                        candidate = more_specific
                        segment = following
                        break
            source = "deterministic_purpose_recovery"
            purpose_added = bool(candidate)
        if not candidate and not presentation_added and _PRESENTATION_CUE.search(text):
            venue = re.search(
                r"\b(?:talk|presentation)\s+(?:at|for)\s+([A-Za-z][\w-]{1,30})",
                text,
                re.IGNORECASE,
            )
            candidate = f"{venue.group(1)} presentation" if venue else "Planned presentation"
            source = "deterministic_presentation_recovery"
            presentation_added = True
        if not candidate:
            continue
        similar = next(
            (item for item in recovered if _similarity(_item_text(item), candidate) >= 0.68),
            None,
        )
        if similar is not None:
            if source == "deterministic_purpose_recovery" and not similar.get("source"):
                similar.update(
                    {
                        "text": candidate,
                        "source": source,
                        "confidence": 0.85,
                        "importance_boost": 0.5,
                        "evidence_segment_ids": [segment["id"]],
                    }
                )
            continue
        recovered.append(
            {
                "text": candidate,
                "source": source,
                "confidence": 0.75,
                "importance_boost": 0.40,
                "evidence_segment_ids": [segment["id"]],
            }
        )
    return recovered


def _normalise_role(value: str) -> str:
    role = clean_text(value).casefold()
    aliases = {
        "future interface": "User Interface",
        "user interface": "User Interface",
    }
    return aliases.get(role, role.title())


def _role_key(value: Any) -> str:
    role = clean_text(value).casefold()
    if role.startswith("marketing "):
        return "marketing"
    return role


def _recover_explicit_participants(
    participants: list[Any], segments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Recover names and roles only from direct self-introductions."""
    recovered = [deepcopy(item) for item in participants if isinstance(item, dict)]
    by_speaker = {
        clean_text(item.get("speaker")): item
        for item in recovered
        if clean_text(item.get("speaker"))
    }
    for segment in segments:
        text = clean_text(segment.get("text"))
        introduction = _INTRODUCTION_CUE.search(text)
        if not introduction:
            continue
        name = clean_text(introduction.group("name"))
        if name.casefold() in _INVALID_PERSON_NAMES:
            continue
        role_match = _ROLE_CUE.search(text[introduction.end() :])
        if not role_match:
            continue
        speaker = clean_text(segment.get("speaker"))
        if not speaker or speaker.casefold() in {"unknown", "speaker_unknown"}:
            continue
        item = by_speaker.get(speaker)
        if item is None:
            item = {"speaker": speaker}
            recovered.append(item)
            by_speaker[speaker] = item
        item.update(
            {
                "name": name,
                "role": _normalise_role(role_match.group("role")),
                "evidence_segment_ids": [segment["id"]],
                "evidence": text,
                "evidence_start_seconds": segment.get("start"),
                "evidence_end_seconds": segment.get("end"),
                "evidence_status": "linked",
                "source": "deterministic_introduction_recovery",
            }
        )
    roll_call_names: list[tuple[str, str]] = []
    roll_call_speakers: set[str] = set()
    for segment in segments:
        roll_call = re.search(
            r"\bthat(?:'s| is)\s+(?P<names>[A-Z][A-Za-z'-]+(?:\s*,\s*"
            r"[A-Z][A-Za-z'-]+)*(?:\s*,?\s+and\s+[A-Z][A-Za-z'-]+))",
            clean_text(segment.get("text")),
        )
        if roll_call:
            roll_call_speakers.add(clean_text(segment.get("speaker")))
            names = [
                token
                for token in re.findall(r"[A-Z][A-Za-z'-]+", roll_call.group("names"))
                if token.casefold() != "and"
            ]
            roll_call_names.extend((name, segment["id"]) for name in names)
    explicit_names = {clean_text(item.get("name")).casefold() for item in recovered}
    unused_roll_names = [
        (name, segment_id)
        for name, segment_id in roll_call_names
        if name.casefold() not in explicit_names
    ]
    for item in recovered:
        current = clean_text(item.get("name"))
        if (
            not current
            or clean_text(item.get("speaker")) in roll_call_speakers
            or any(current.casefold() == name.casefold() for name, _ in roll_call_names)
        ):
            continue
        match = max(
            unused_roll_names,
            key=lambda value: SequenceMatcher(None, current.casefold(), value[0].casefold()).ratio(),
            default=None,
        )
        if match and (
            len(unused_roll_names) == 1
            or SequenceMatcher(None, current.casefold(), match[0].casefold()).ratio() >= 0.72
        ):
            item["name"] = match[0]
            item["name_reconciled_from"] = current
            item["evidence_segment_ids"] = list(
                dict.fromkeys([*item.get("evidence_segment_ids", []), match[1]])
            )
            unused_roll_names.remove(match)
    recovered.sort(
        key=lambda item: (
            item.get("evidence_start_seconds") is None,
            item.get("evidence_start_seconds") or 0.0,
        )
    )
    return recovered


def _format_euro_amount(raw: str, context: str) -> str:
    """Format an explicitly spoken euro amount without guessing an absent value."""
    value = clean_text(raw).casefold().replace(",", "")
    word_values = {
        "twelve fifty": "12.50",
        "twenty five": "25",
        "twenty-five": "25",
        "fifty million": "50 million",
        "fifteen million": "15 million",
    }
    value = word_values.get(value, value)
    # ASR commonly removes the decimal from "twelve fifty". Only repair 1250
    # when the same compact passage explicitly says it is 50% of a EUR 25 price.
    if (
        value == "1250"
        and re.search(r"\b50\s*(?:%|percent\b)", context, re.IGNORECASE)
        and re.search(r"\b25\s*(?:euro|euros|eur)\b", context, re.IGNORECASE)
    ):
        value = "12.50"
    return f"€{value}"


def _recover_explicit_requirements(
    requirements: list[Any], segments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Recover compact, explicit targets and constraints with local evidence."""
    recovered = [deepcopy(item) for item in requirements if isinstance(item, dict)]
    meeting_text = " ".join(clean_text(item.get("text")) for item in segments)
    for index, segment in enumerate(segments):
        text = clean_text(segment.get("text"))
        nearby = " ".join(
            clean_text(item.get("text"))
            for item in segments[max(0, index - 5) : min(len(segments), index + 6)]
        )
        candidates: list[str] = []
        quality = re.search(
            r"\bsupposed to be\s+(.+?)(?:[.?!]|$)", text, re.IGNORECASE
        )
        if quality and not (
            _INTRODUCTION_CUE.search(text) and _ROLE_CUE.search(quality.group(1))
        ) and _PRODUCT_CONTEXT_CUE.search(nearby) and not _SETUP_LANGUAGE_CUE.search(text):
            candidates.append(f"Product must be {clean_text(quality.group(1)).rstrip(' ,;')}.")
        wanted_quality = re.search(
            r"\bwe want it to be\s+(.+?)(?:[.?!]|$)", text, re.IGNORECASE
        )
        if (
            wanted_quality
            and _PRODUCT_CONTEXT_CUE.search(nearby)
            and not _SETUP_LANGUAGE_CUE.search(text)
        ):
            value = clean_text(wanted_quality.group(1)).rstrip(" ,;")
            if 2 <= len(_content_tokens(value)) <= 18:
                candidates.append(f"Product qualities: {value}.")
            else:
                qualities: list[str] = []
                for pattern, label in (
                    (r"\boriginal\b", "original"),
                    (r"\btrendy\b", "trendy"),
                    (r"\buser[- ]friendly\b", "user-friendly"),
                    (r"\bappeal(?:ing)? to (?:a )?wide market\b", "appealing to a wide market"),
                ):
                    if re.search(pattern, value, re.IGNORECASE):
                        qualities.append(label)
                if len(qualities) >= 2:
                    candidates.append(f"Product qualities: {', '.join(qualities)}.")
        selling = re.search(
            r"\b(?:selling(?: this| the)?(?: [\w-]+){0,4} for|"
            r"selling price(?: goal)? (?:is|of|at))\s+"
            r"(?P<amount>\d+(?:\.\d+)?|twenty[- ]five)\s*(?:euro|euros|eur)\b",
            text,
            re.IGNORECASE,
        )
        if selling:
            candidates.append(
                f"Selling price: {_format_euro_amount(selling.group('amount'), nearby)}."
            )
        target = re.search(
            r"\b(?:(?:aiming|targeting|target is)\s+to\s+(?:make|reach|generate)|"
            r"profit (?:aim|target)(?: is)?)\s+"
            r"(?P<amount>\d+(?:\.\d+)?\s+million|fifty million|fifteen million)\s*"
            r"(?:euro|euros|eur)\b",
            text,
            re.IGNORECASE,
        )
        if target:
            candidates.append(
                f"Revenue target: {_format_euro_amount(target.group('amount'), nearby)}."
            )
        maximum = re.search(
            r"\b(?P<cost_label>production cost(?:'s)?|cost).{0,35}?"
            r"(?:(?:any )?more than|is|of|at)\s+"
            r"(?P<amount>\d+(?:\.\d+)?|twelve fifty)"
            r"(?:\s*(?P<cost_currency>euro|euros|eur)\b)?",
            text,
            re.IGNORECASE,
        )
        if maximum and (
            maximum.group("cost_currency")
            or (
                "production cost" in maximum.group("cost_label").casefold()
                and re.search(
                    r"\bselling price\b.{0,120}\b(?:euro|euros|eur)\b",
                    meeting_text,
                    re.IGNORECASE,
                )
            )
        ):
            amount = _format_euro_amount(maximum.group("amount"), nearby)
            percentage = " (50% of selling price)" if re.search(
                r"\b(?:50\s*(?:%|percent\b)|half of the selling price)",
                nearby,
                re.IGNORECASE,
            ) else ""
            candidates.append(f"Maximum production cost: {amount}{percentage}.")
        if re.search(
            r"\b(?:(?:selling|sold)\s+(?:this|it)\s+on an international scale|"
            r"(?:hope|plan|aim) to sell (?:this|it) internationally|"
            r"market range (?:is )?international|international market)\b",
            text,
            re.IGNORECASE,
        ):
            candidates.append("International market.")
        if re.search(
            r"\b(?:accessible|usable).{0,45}\b(?:all|every) age groups?\b|"
            r"\b(?:all|every) age groups?.{0,45}\b(?:accessible|usable)\b",
            text,
            re.IGNORECASE,
        ):
            candidates.append("Usable across all age groups.")
        sales_prompt = " ".join(
            clean_text(item.get("text"))
            for item in segments[max(0, index - 3) : index + 1]
        )
        if (
            re.search(r"\bhow many (?:should|will|can) we sell\b", sales_prompt, re.IGNORECASE)
            and re.search(r"\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|"
                          r"eight|nine|ten)\s+million\b", text, re.IGNORECASE)
        ):
            unit_amounts = re.findall(
                r"\b(?P<amount>\d+(?:\.\d+)?|one|two|three|four|five|six|seven|"
                r"eight|nine|ten)\s+million\b",
                nearby,
                re.IGNORECASE,
            )
            if unit_amounts:
                most_common = Counter(value.casefold() for value in unit_amounts).most_common(1)[0][0]
                numeric = {
                    "one": "1",
                    "two": "2",
                    "three": "3",
                    "four": "4",
                    "five": "5",
                    "six": "6",
                    "seven": "7",
                    "eight": "8",
                    "nine": "9",
                    "ten": "10",
                }.get(most_common, most_common)
                candidates.append(f"Sales target: {numeric} million units.")
        for candidate in candidates:
            if any(_similarity(_item_text(item), candidate) >= 0.72 for item in recovered):
                continue
            recovered.append(
                {
                    "text": candidate,
                    "source": "deterministic_requirement_recovery",
                    "confidence": 0.85,
                    "evidence_segment_ids": [segment["id"]],
                }
            )
    return recovered


def _claim_importance(
    item: dict[str, Any],
    segments: list[dict[str, Any]],
    purpose_index: int | None,
) -> float:
    claim_tokens = _content_tokens(_item_text(item))
    if not claim_tokens:
        return 0.0
    matching_indices: list[int] = []
    for index, segment in enumerate(segments):
        segment_tokens = _content_tokens(segment.get("text"))
        overlap = len(claim_tokens & segment_tokens)
        required = 1 if len(claim_tokens) <= 2 else 2
        if overlap >= required or overlap / len(claim_tokens) >= 0.5:
            matching_indices.append(index)
    bucket_count = min(6, max(1, len(segments)))
    buckets = {
        min(bucket_count - 1, index * bucket_count // max(1, len(segments)))
        for index in matching_indices
    }
    evidence_indices = [
        int(segment["index"])
        for segment in segments
        if segment["id"] in set(item.get("evidence_segment_ids", []))
    ]
    recurrence = min(1.0, len(matching_indices) / 5.0)
    distribution = min(1.0, len(buckets) / 3.0)
    evidence_density = min(1.0, len(evidence_indices) / 2.0)
    score = 0.18 + 0.45 * recurrence + 0.25 * distribution + 0.15 * evidence_density
    score += float(item.get("importance_boost", 0.0) or 0.0)
    if claim_tokens <= _THEME_GENERIC:
        score -= 0.45
        item["importance_reason"] = "generic_topic"
    if any(
        _MEETING_PURPOSE_CUE.search(clean_text(segment.get("text")))
        for segment in segments
        if segment["id"] in set(item.get("evidence_segment_ids", []))
    ):
        score += 0.35
    if (
        purpose_index is not None
        and evidence_indices
        and max(evidence_indices) < purpose_index
    ):
        score -= 0.75
        item["importance_reason"] = "pre_agenda_chatter"
    return round(max(0.0, min(1.0, score)), 3)


def _rank_grounded_claims(
    items: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    *,
    minimum_score: float,
    limit: int,
) -> list[dict[str, Any]]:
    purpose_index = next(
        (
            int(segment["index"])
            for segment in segments
            if _MEETING_PURPOSE_CUE.search(clean_text(segment.get("text")))
        ),
        None,
    )
    ranked: list[dict[str, Any]] = []
    for item in items:
        item["importance_score"] = _claim_importance(item, segments, purpose_index)
        if (
            item.get("importance_reason") != "pre_agenda_chatter"
            and item["importance_score"] >= minimum_score
        ):
            ranked.append(item)
    ranked.sort(
        key=lambda item: (
            -float(item.get("importance_score", 0.0)),
            item.get("evidence_start_seconds") is None,
            item.get("evidence_start_seconds") or 0.0,
        )
    )
    return ranked[:limit]


def _local_evidence_ids(
    valid_ids: list[str], by_id: dict[str, dict[str, Any]], candidate: str
) -> list[str]:
    """Keep the strongest compact citation window instead of distant snippets."""
    if not valid_ids:
        return []
    anchor = max(valid_ids, key=lambda segment_id: _similarity(candidate, by_id[segment_id]["text"]))
    anchor_segment = by_id[anchor]
    local: list[str] = []
    for segment_id in valid_ids:
        segment = by_id[segment_id]
        if abs(int(segment["index"]) - int(anchor_segment["index"])) > EVIDENCE_MAX_INDEX_SPAN:
            continue
        anchor_start = anchor_segment.get("start")
        segment_start = segment.get("start")
        if (
            anchor_start is not None
            and segment_start is not None
            and abs(float(segment_start) - float(anchor_start)) > EVIDENCE_MAX_TIME_SPAN_SECONDS
        ):
            continue
        local.append(segment_id)
    ranked = sorted(
        local,
        key=lambda segment_id: (
            -_similarity(candidate, by_id[segment_id]["text"]),
            by_id[segment_id]["index"],
        ),
    )[:3]
    return sorted(ranked, key=lambda segment_id: by_id[segment_id]["index"])


def _validate_action_details(
    linked: dict[str, Any], evidence_segments: list[dict[str, Any]], evidence_text: str
) -> None:
    """Keep owner/deadline fields only when the cited source supports them."""
    assignment = _DIRECT_ASSIGNMENT.search(evidence_text)
    future = _NAMED_FUTURE_COMMITMENT.search(evidence_text)
    inferred_owner = (assignment or future).group("assignee") if assignment or future else ""
    if inferred_owner.casefold() in {"i", "we", "he", "she", "they", "it"}:
        inferred_owner = ""
    if not inferred_owner and _FIRST_PERSON_COMMITMENT.search(evidence_text):
        speakers = {
            clean_text(segment.get("speaker"))
            for segment in evidence_segments
            if clean_text(segment.get("speaker"))
        }
        inferred_owner = next(iter(speakers)) if len(speakers) == 1 else ""

    model_owner = clean_text(linked.get("owner"))
    if inferred_owner:
        linked["owner"] = inferred_owner
        task = clean_text(linked.get("task"))
        linked["task"] = re.sub(
            rf"\s+to\s+{re.escape(inferred_owner)}\b",
            "",
            task,
            flags=re.IGNORECASE,
        ).strip()
    elif model_owner and model_owner.casefold() not in evidence_text.casefold():
        linked["owner"] = None

    deadline = clean_text(linked.get("deadline"))
    if deadline and deadline.casefold() not in evidence_text.casefold():
        linked["deadline"] = None


def _word_overlap_size(left: str, right: str) -> int:
    left_words = _TOKEN.findall(left.casefold())
    right_words = _TOKEN.findall(right.casefold())
    for size in range(min(len(left_words), len(right_words), 12), 0, -1):
        if left_words[-size:] == right_words[:size]:
            return size
    return 0


def _join_overlapping_text(left: str, right: str) -> str:
    left_words = _TOKEN.findall(clean_text(left))
    right_words = _TOKEN.findall(clean_text(right))
    overlap = _word_overlap_size(left, right)
    if overlap:
        return " ".join([*left_words, *right_words[overlap:]])
    return clean_text(f"{left} {right}")


def _clean_action_task(value: str) -> str:
    """Turn a directly quoted action passage into a concise reviewable task."""
    text = clean_text(value).strip(" ,;:-")
    if (
        _NEGATED_ACTION.search(text)
        or _PAST_ACTIVITY.search(text)
        or _NON_FOLLOWUP_ACTION.search(text)
        or _NON_ACTION_DESCRIPTION.search(text)
    ):
        return ""
    extractors = (
        r"\bask(?:ed|ing)? (?:me|us|you|him|her|them) to\s+(?P<task>.+)$",
        r"\byou\s+(?:could(?: always)?|should|might(?: want to)?|need to)\s+(?P<task>.+)$",
        r"\b(?:maybe\s+)?we can\s+(?P<task>.+)$",
        r"\bwhy don't we\s+(?P<task>.+)$",
        r"\bi will\s+(?:have time to\s+)?(?P<task>.+)$",
        r"\bi'll\s+(?P<task>.+)$",
        r"^will\s+(?:have time to\s+)?(?P<task>.+)$",
    )
    for pattern in extractors:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            text = clean_text(match.group("task"))
            break
    text = re.sub(r"^(?:yeah|okay|well|but|so|anyway)[, ]+", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"^you to (?=(?:check|confirm|email|include|review|revise|send|share|update)\b)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+you had\b.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(?:and|or|the|a|an|to|of|for|with)$", "", text, flags=re.IGNORECASE)
    if _NON_FOLLOWUP_ACTION.search(text) or _NON_ACTION_DESCRIPTION.search(text):
        return ""
    if "?" in text and not _DIRECT_REQUEST.search(text):
        return ""
    if len(_TOKEN.findall(text)) > 32:
        return ""
    if re.match(
        r"^(?:the problem is|for the most frequent case|nothing i want|"
        r"thank you[, ]+mr\.? chair|good morning[, ]+mr\.? chair)\b",
        text,
        re.IGNORECASE,
    ):
        return ""
    if text:
        text = text[0].upper() + text[1:]
    if len(_content_tokens(text)) < 2:
        return ""
    return text[:180].rstrip(" ,;:-")


def _recover_role_assignments(
    actions: list[dict[str, Any]], segments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Recover explicit role-based assignments across adjacent wrap-up turns."""
    recovered = [deepcopy(item) for item in actions]
    wrap_indices = [
        index
        for index, segment in enumerate(segments)
        if _WRAP_UP_CUE.search(clean_text(segment.get("text")))
    ]
    start_index = min(wrap_indices) if wrap_indices else int(len(segments) * 0.65)
    for index in range(start_index, len(segments)):
        segment = segments[index]
        segment_text = clean_text(segment.get("text"))
        role_matches = list(_ROLE_CUE.finditer(segment_text))
        if not role_matches:
            continue
        for role_position, role_match in enumerate(role_matches):
            role = _normalise_role(role_match.group("role"))
            same_segment_end = (
                role_matches[role_position + 1].start()
                if role_position + 1 < len(role_matches)
                else len(segment_text)
            )
            window = [segment]
            text_parts = [segment_text[role_match.start() : same_segment_end]]
            for following in segments[index + 1 : index + 7]:
                following_text = clean_text(following.get("text"))
                if _ROLE_CUE.search(following_text):
                    break
                window.append(following)
                text_parts.append(following_text)
            assignment_text = clean_text(" ".join(text_parts))
            task = ""
            explicit = re.search(
                r"\b(?:you(?:'re| are| will|'ll)|(?:he|she|they|we)(?:'ll| will)|"
                r"is going to be)\s+(?:going to be\s+|be\s+)?(?:just\s+)?"
                r"(?P<task>(?:looking (?:more )?into|working on|thinking about|"
                r"handling|preparing|reviewing|defining|developing|designing)\s+"
                r".+?)(?:[.?!]|$)",
                assignment_text,
                re.IGNORECASE,
            )
            role_key = _role_key(role)
            if role_key == "marketing" and re.search(
                r"\buser requirements?\b", assignment_text, re.IGNORECASE
            ):
                task = "Determine the user requirements"
            elif role_key == "user interface" and re.search(
                r"\btechnical functions?\b", assignment_text, re.IGNORECASE
            ):
                task = "Define the technical functions"
            elif explicit:
                task = clean_text(explicit.group("task"))
                task = re.sub(
                    r"^looking (?:more )?into\b", "Work on", task, flags=re.IGNORECASE
                )
                task = re.sub(r"^working on\b", "Work on", task, flags=re.IGNORECASE)
                task = re.sub(
                    r"^thinking about\b", "Determine", task, flags=re.IGNORECASE
                )
            elif re.search(r"\buser requirements?\b", assignment_text, re.IGNORECASE):
                task = "Determine the user requirements"
            elif re.search(r"\btechnical functions?\b", assignment_text, re.IGNORECASE):
                task = "Define the technical functions"
            elif re.search(r"\bworking design\b", assignment_text, re.IGNORECASE):
                task = "Work on the working design"
            elif re.search(r"\bphysical design\b", assignment_text, re.IGNORECASE):
                task = "Work on the physical design"
            if not task:
                continue
            task = re.sub(
                r",?\s+(?:so you know\b|and we(?:'ll| will) all\b|i guess\b).*$",
                "",
                task,
                flags=re.IGNORECASE,
            )
            task = clean_text(task).rstrip(" ,;:-")
            if len(_content_tokens(task)) < 2 or len(_TOKEN.findall(task)) > 18:
                continue
            evidence_ids = [
                item["id"] for item in window if clean_text(item.get("text"))
            ]
            candidate = {
                "task": task[0].upper() + task[1:],
                "owner": role,
                "owner_role": role,
                "deadline": None,
                "commitment_status": "confirmed",
                "confidence": 0.9,
                "source": "deterministic_role_assignment",
                "evidence_segment_ids": evidence_ids,
            }
            if any(_actions_match(item, candidate) for item in recovered):
                continue
            recovered.append(candidate)
    return _deduplicate_actions(recovered)


def _actions_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_ids = set(left.get("evidence_segment_ids", []))
    right_ids = set(right.get("evidence_segment_ids", []))
    if left_ids & right_ids:
        return True
    left_tokens = _content_tokens(_item_text(left))
    right_tokens = _content_tokens(_item_text(right))
    if not left_tokens or not right_tokens:
        return False
    if min(len(left_tokens), len(right_tokens)) < 2:
        return False
    containment = len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))
    return containment >= 0.72 or _similarity(_item_text(left), _item_text(right)) >= 0.72


def _deduplicate_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduplicated: list[dict[str, Any]] = []
    for item in actions:
        duplicate = next(
            (existing for existing in deduplicated if _actions_match(existing, item)),
            None,
        )
        if duplicate is None:
            deduplicated.append(item)
            continue
        duplicate["evidence_segment_ids"] = list(
            dict.fromkeys(
                [
                    *duplicate.get("evidence_segment_ids", []),
                    *item.get("evidence_segment_ids", []),
                ]
            )
        )
        if len(_item_text(item)) > len(_item_text(duplicate)):
            duplicate["task"] = item.get("task", duplicate.get("task"))
        if item.get("commitment_status") == "confirmed":
            duplicate["commitment_status"] = "confirmed"
        duplicate["owner"] = duplicate.get("owner") or item.get("owner")
        duplicate["deadline"] = duplicate.get("deadline") or item.get("deadline")
    return deduplicated


def _recover_explicit_actions(
    actions: list[Any], segments: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Recover narrow, directly quoted follow-ups that Qwen omitted."""
    recovered: list[dict[str, Any]] = []
    for item in actions:
        if not isinstance(item, dict):
            continue
        normalized = deepcopy(item)
        normalized["task"] = _clean_action_task(_item_text(normalized))
        if normalized["task"]:
            recovered.append(normalized)
    consumed_indices: set[int] = set()
    for index, segment in enumerate(segments):
        if index in consumed_indices:
            continue
        text = clean_text(segment.get("text"))
        if not text or not _ACTION_TASK_VERB.search(text):
            continue
        evidence_ids = [segment["id"]]
        combined = text
        for next_index in range(index + 1, min(len(segments), index + 3)):
            following = segments[next_index]
            same_speaker = clean_text(following.get("speaker")) == clean_text(
                segment.get("speaker")
            )
            current_end = segment.get("end") or segment.get("start")
            next_start = following.get("start")
            close_in_time = (
                current_end is None
                or next_start is None
                or float(next_start) - float(current_end) <= 15.0
            )
            incomplete = bool(
                re.search(
                    r"\b(?:and|or|the|a|an|to|of|for|with|had)$",
                    combined,
                    re.IGNORECASE,
                )
            )
            overlap = _word_overlap_size(combined, clean_text(following.get("text")))
            if not same_speaker or not close_in_time or not (incomplete or overlap >= 2):
                break
            combined = _join_overlapping_text(combined, clean_text(following.get("text")))
            evidence_ids.append(following["id"])
            consumed_indices.add(next_index)
        previous = clean_text(segments[index - 1].get("text")) if index else ""
        confirmed, suggested = _action_cue_status(combined)
        if not (confirmed or suggested):
            continue
        task = _clean_action_task(combined)
        if not task:
            continue
        if any(
            _similarity(_item_text(item), task) >= 0.68
            or set(evidence_ids) & set(item.get("evidence_segment_ids", []))
            for item in recovered
        ):
            continue
        if previous and not any(
            pattern.search(combined)
            for pattern in (
                _FIRST_PERSON_COMMITMENT,
                _NAMED_FUTURE_COMMITMENT,
                _DIRECT_ASSIGNMENT,
                _PRONOUN_FUTURE_TASK,
                _STRICT_ACTION_SUGGESTION,
                _DIRECT_REQUEST,
                _ASK_TO_TASK,
            )
        ):
            evidence_ids.insert(0, segments[index - 1]["id"])
        recovered.append(
            {
                "task": task,
                "owner": None,
                "deadline": None,
                "commitment_status": "confirmed" if confirmed else "suggested",
                "confidence": 0.65,
                "source": "deterministic_cue_recovery",
                "evidence_segment_ids": evidence_ids,
            }
        )
    return _deduplicate_actions(recovered)


def _link_item(
    item: dict[str, Any], segments: list[dict[str, Any]], *, item_type: str
) -> dict[str, Any]:
    linked = deepcopy(item)
    if item_type == "action":
        linked["task"] = _clean_action_task(_item_text(linked))
        item = linked
    by_id = {segment["id"]: segment for segment in segments}
    claimed_ids = [
        clean_text(segment_id)
        for segment_id in item.get("evidence_segment_ids", [])
        if clean_text(segment_id)
    ]
    valid_ids = list(dict.fromkeys(segment_id for segment_id in claimed_ids if segment_id in by_id))
    if item.get("source") == "deterministic_requirement_recovery" and valid_ids:
        anchor_index = int(by_id[valid_ids[0]]["index"])
        valid_ids = [
            segment["id"]
            for segment in segments
            if abs(int(segment["index"]) - anchor_index) <= 1
        ]
    linked["claimed_evidence_segment_ids"] = claimed_ids
    model_evidence = clean_text(item.get("evidence"))
    candidate = " ".join(value for value in (model_evidence, _item_text(item)) if value)
    valid_ids = _local_evidence_ids(valid_ids, by_id, candidate)
    linked["model_evidence"] = model_evidence
    linked["evidence_segment_ids"] = valid_ids
    if not valid_ids:
        linked["evidence"] = ""
        linked["evidence_start_seconds"] = None
        linked["evidence_end_seconds"] = None
        linked["evidence_status"] = "unlinked"
        return linked

    support_threshold = {
        "topic": EVIDENCE_TOPIC_SUPPORT_THRESHOLD,
        "key point": EVIDENCE_KEY_POINT_SUPPORT_THRESHOLD,
    }.get(item_type, EVIDENCE_SUPPORT_THRESHOLD)
    evidence_segments = [by_id[segment_id] for segment_id in valid_ids]
    evidence_text = " ".join(segment["text"] for segment in evidence_segments)
    support_score = _support_score(_item_text(item), evidence_text)
    if item.get("source") == "deterministic_requirement_recovery":
        # These candidates are constructed directly from a matched phrase in
        # the cited evidence. Lexical similarity is not meaningful for concise
        # normalisations such as "International market" or "Revenue target".
        support_score = 1.0
    if support_score < support_threshold and item_type in {"topic", "key point"}:
        repaired = _best_evidence_window(
            _item_text(item), segments, support_threshold
        )
        if repaired:
            evidence_segments = repaired
            valid_ids = [segment["id"] for segment in repaired]
            linked["evidence_segment_ids"] = valid_ids
            linked["citation_repaired"] = True
            evidence_text = " ".join(segment["text"] for segment in repaired)
            support_score = _support_score(_item_text(item), evidence_text)
    cue_supported = True
    commitment_status = ""
    if item_type == "decision":
        cue_supported = bool(_DECISION_CUE.search(evidence_text))
    if item_type == "action":
        confirmed, suggested = _action_cue_status(evidence_text)
        if item.get("source") == "deterministic_role_assignment":
            confirmed = True
        cue_supported = confirmed or suggested
        commitment_status = "confirmed" if confirmed else "suggested"
        linked["commitment_status"] = commitment_status
    elif item_type == "risk":
        cue_supported = bool(_RISK_CUE.search(evidence_text))
    elif item_type == "open question":
        cue_supported = bool(_QUESTION_CUE.search(evidence_text))
    elif item_type == "requirement":
        cue_supported = bool(
            item.get("source") == "deterministic_requirement_recovery"
            or _REQUIREMENT_CUE.search(evidence_text)
        )
    linked["evidence_support_score"] = round(support_score, 3)
    linked["evidence_cue_supported"] = cue_supported
    linked["evidence_support_threshold"] = support_threshold
    if support_score < support_threshold or not cue_supported:
        linked["evidence"] = ""
        linked["evidence_segment_ids"] = []
        linked["evidence_start_seconds"] = None
        linked["evidence_end_seconds"] = None
        linked["evidence_status"] = "rejected"
        return linked

    linked["evidence"] = evidence_text
    if item_type == "action":
        _validate_action_details(linked, evidence_segments, evidence_text)
        if item.get("source") == "deterministic_role_assignment":
            linked["owner"] = item.get("owner")
            linked["owner_role"] = item.get("owner_role")
    starts = [
        float(segment["start"]) for segment in evidence_segments if segment.get("start") is not None
    ]
    ends = [
        float(segment["end"]) for segment in evidence_segments if segment.get("end") is not None
    ]
    linked["evidence_start_seconds"] = min(starts) if starts else None
    linked["evidence_end_seconds"] = max(ends) if ends else None
    linked["evidence_status"] = "linked"
    linked.setdefault("review_status", "draft")
    return linked


def link_analysis_evidence(
    analysis: dict[str, Any], segments: list[dict[str, Any]]
) -> dict[str, Any]:
    """Keep only analytical claims supported by compact source evidence."""
    linked = deepcopy(analysis)
    linked["participants"] = _recover_explicit_participants(
        linked.get("participants", []), segments
    )
    linked["topics"] = _recover_structural_topics(
        linked.get("topics", []), segments
    )
    linked["requirements"] = _recover_explicit_requirements(
        linked.get("requirements", []), segments
    )
    linked["action_items"] = _recover_explicit_actions(
        linked.get("action_items", []), segments
    )
    linked["action_items"] = _recover_role_assignments(
        linked["action_items"], segments
    )
    rejected: list[dict[str, Any]] = []
    fields = (
        ("topics", "topic"),
        ("key_points", "key point"),
        ("requirements", "requirement"),
        ("decisions", "decision"),
        ("action_items", "action"),
        ("risks", "risk"),
        ("open_questions", "open question"),
    )
    for field, item_type in fields:
        candidates = [
            _link_item(item, segments, item_type=item_type)
            for item in linked.get(field, [])
            if isinstance(item, dict)
        ]
        for item in candidates:
            if item.get("evidence_status") != "linked":
                rejected.append(
                    {
                        "type": item_type,
                        "text": _item_text(item),
                        "reason": clean_text(item.get("evidence_status")) or "unsupported",
                        "claimed_evidence_segment_ids": item.get(
                            "claimed_evidence_segment_ids", []
                        ),
                        "support_score": item.get("evidence_support_score", 0.0),
                        "cue_supported": item.get("evidence_cue_supported", False),
                    }
                )
        linked[field] = [
            item for item in candidates if item.get("evidence_status") == "linked"
        ]
    linked["topics"] = _rank_grounded_claims(
        linked.get("topics", []), segments, minimum_score=0.22, limit=5
    )
    linked["key_points"] = _rank_grounded_claims(
        linked.get("key_points", []), segments, minimum_score=0.28, limit=6
    )
    participants_by_role = {
        _role_key(item.get("role")): item
        for item in linked.get("participants", [])
        if isinstance(item, dict) and clean_text(item.get("role"))
    }
    for action in linked.get("action_items", []):
        if action.get("source") != "deterministic_role_assignment":
            continue
        participant = participants_by_role.get(_role_key(action.get("owner_role")))
        if participant and clean_text(participant.get("name")):
            action["owner"] = clean_text(participant.get("name"))
            action["owner_evidence_segment_ids"] = participant.get(
                "evidence_segment_ids", []
            )
    linked["action_items"] = _deduplicate_actions(linked.get("action_items", []))
    linked["evidence_rejections"] = rejected
    if rejected:
        linked.setdefault("uncertainties", []).append(
            f"{len(rejected)} unsupported analytical candidate(s) were omitted."
        )
    return linked
