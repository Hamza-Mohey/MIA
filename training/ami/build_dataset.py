"""Build leak-resistant manifests from the local AMI corpus.

It combines the local mixed-headset recordings with AMI's official manual
annotations, then writes JSONL manifests suitable for evaluation and carefully
scoped fine-tuning experiments.
"""

from __future__ import annotations

import argparse
import json
import re
import wave
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

NITE = "http://nite.sourceforge.net/"
NITE_ID = f"{{{NITE}}}id"
MEETING_RE = re.compile(r"^ES\d{4}[a-d]$")
PROJECT_RE = re.compile(r"^(ES\d{4})[a-d]$")
HREF_RE = re.compile(r"#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?")
PUNCTUATION = frozenset(",.!?;:%)]}")
NO_SPACE_AFTER = frozenset("([{£$")
MAX_ASR_SEGMENT_SECONDS = 28.0


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _clean_text(value: str | None) -> str:
    return " ".join((value or "").split())


def _relative(path: Path, workspace_root: Path) -> str:
    try:
        return path.resolve().relative_to(workspace_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _write_rttm(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write speaker turns in NIST RTTM format for diarization evaluation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            duration = float(record.get("duration", 0.0))
            if duration <= 0:
                continue
            speaker = record.get("speaker_id") or record.get("agent") or "unknown"
            handle.write(
                f"SPEAKER {record['meeting_id']} 1 {float(record['start']):.3f} "
                f"{duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
            )
            count += 1
    return count


def join_tokens(tokens: list[dict[str, Any]]) -> str:
    """Join AMI word tokens while preserving ordinary punctuation spacing."""
    result = ""
    for token in tokens:
        text = _clean_text(token.get("text"))
        if not text:
            continue
        is_punctuation = bool(token.get("punctuation")) or text in PUNCTUATION
        if not result or is_punctuation or result[-1] in NO_SPACE_AFTER:
            result += text
        else:
            result += " " + text
    return result.strip()


def read_words(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Return ordered token ids and token metadata from an AMI words file."""
    ordered: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    root = ET.parse(path).getroot()
    for element in root:
        kind = _local_name(element.tag)
        token_id = element.attrib.get(NITE_ID, "")
        if not token_id:
            continue
        # Non-lexical events remain addressable for segment ranges, but do not
        # become hallucinated words in the reference transcript.
        text = "" if kind != "w" else _clean_text(element.text)
        token = {
            "id": token_id,
            "kind": kind,
            "text": text,
            "start": float(element.attrib.get("starttime", 0.0)),
            "end": float(element.attrib.get("endtime", 0.0)),
            "punctuation": element.attrib.get("punc", "false").lower() == "true",
        }
        ordered.append(token_id)
        by_id[token_id] = token
    return ordered, by_id


def _tokens_for_href(
    href: str, ordered_ids: list[str], by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    match = HREF_RE.search(href)
    if not match:
        return []
    start_id, end_id = match.group(1), match.group(2) or match.group(1)
    positions = {token_id: index for index, token_id in enumerate(ordered_ids)}
    if start_id not in positions or end_id not in positions:
        return []
    first, last = positions[start_id], positions[end_id]
    if first > last:
        first, last = last, first
    return [by_id[token_id] for token_id in ordered_ids[first : last + 1]]


def read_segments(
    segments_path: Path,
    words_path: Path,
    meeting_id: str,
    agent: str,
    speaker: dict[str, str],
) -> list[dict[str, Any]]:
    """Reconstruct speaker-attributed reference utterances from NXT links."""
    ordered_ids, by_id = read_words(words_path)
    root = ET.parse(segments_path).getroot()
    segments: list[dict[str, Any]] = []
    for element in root:
        if _local_name(element.tag) != "segment":
            continue
        tokens: list[dict[str, Any]] = []
        for child in element:
            if _local_name(child.tag) == "child":
                tokens.extend(_tokens_for_href(child.attrib.get("href", ""), ordered_ids, by_id))
        lexical = [token for token in tokens if token.get("text") and not token.get("punctuation")]
        if not lexical:
            continue
        annotation_start = float(element.attrib.get("transcriber_start", lexical[0]["start"]))
        annotation_end = float(element.attrib.get("transcriber_end", lexical[-1]["end"]))
        chunks: list[list[dict[str, Any]]] = [[]]
        chunk_start = lexical[0]["start"]
        for token in tokens:
            if (
                token.get("text")
                and chunks[-1]
                and token["end"] - chunk_start > MAX_ASR_SEGMENT_SECONDS
            ):
                chunks.append([])
                chunk_start = token["start"]
            chunks[-1].append(token)
        chunks = [
            chunk
            for chunk in chunks
            if any(token.get("text") and not token.get("punctuation") for token in chunk)
        ]
        base_id = element.attrib.get(NITE_ID, "")
        for index, chunk in enumerate(chunks, start=1):
            chunk_lexical = [
                token for token in chunk if token.get("text") and not token.get("punctuation")
            ]
            if len(chunks) == 1 and annotation_end - annotation_start <= MAX_ASR_SEGMENT_SECONDS:
                start, end = annotation_start, annotation_end
                segment_id = base_id
            else:
                start, end = chunk_lexical[0]["start"], chunk_lexical[-1]["end"]
                segment_id = f"{base_id}.part{index}" if len(chunks) > 1 else base_id
            segments.append(
                {
                    "segment_id": segment_id,
                    "meeting_id": meeting_id,
                    "agent": agent,
                    "role": speaker.get("role", "unknown"),
                    "speaker_id": speaker.get("global_name", ""),
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration": round(max(0.0, end - start), 3),
                    "text": join_tokens(chunk),
                }
            )
    return segments


def read_abstractive(path: Path) -> dict[str, list[str]]:
    """Read AMI's gold abstract, actions, decisions, and problems."""
    result = {"abstract": [], "actions": [], "decisions": [], "problems": []}
    if not path.exists():
        return result
    root = ET.parse(path).getroot()
    for section in root:
        name = _local_name(section.tag)
        if name not in result:
            continue
        result[name] = [
            text for child in section if (text := _clean_text("".join(child.itertext())))
        ]
    return result


def read_participant_summaries(directory: Path, meeting_id: str) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    for path in sorted(directory.glob(f"{meeting_id}.*.summ.xml")):
        agent_match = re.match(rf"{re.escape(meeting_id)}\.([A-D])\.summ\.xml$", path.name)
        if not agent_match:
            continue
        sections: dict[str, list[str]] = {}
        for section in ET.parse(path).getroot():
            name = _local_name(section.tag).removeprefix("participant_")
            sections[name] = [
                text for child in section if (text := _clean_text("".join(child.itertext())))
            ]
        result[agent_match.group(1)] = sections
    return result


def read_meeting_metadata(path: Path) -> dict[str, dict[str, Any]]:
    """Read the official role mapping and official train/development split."""
    metadata: dict[str, dict[str, Any]] = {}
    for meeting in ET.parse(path).getroot():
        if _local_name(meeting.tag) != "meeting":
            continue
        meeting_id = meeting.attrib.get("observation", "")
        if not MEETING_RE.fullmatch(meeting_id):
            continue
        speakers: dict[str, dict[str, str]] = {}
        for element in meeting:
            if _local_name(element.tag) != "speaker":
                continue
            agent = element.attrib.get("nxt_agent", "")
            if agent:
                speakers[agent] = {
                    "role": element.attrib.get("role", "unknown"),
                    "global_name": element.attrib.get("global_name", ""),
                    "channel": element.attrib.get("channel", ""),
                }
        seen_type = meeting.attrib.get("seen_type", "").lower()
        split = {"training": "train", "development": "validation"}.get(seen_type, "test")
        metadata[meeting_id] = {
            "duration_seconds_official": float(meeting.attrib.get("duration", 0.0)),
            "official_seen_type": seen_type or "unseen",
            "split": split,
            "speakers": speakers,
        }
    return metadata


def read_audio_metadata(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as audio:
        frame_rate = audio.getframerate()
        frames = audio.getnframes()
        return {
            "sample_rate_hz": frame_rate,
            "channels": audio.getnchannels(),
            "sample_width_bytes": audio.getsampwidth(),
            "frames": frames,
            "duration_seconds": round(frames / frame_rate, 3) if frame_rate else 0.0,
        }


def _format_timestamp(seconds: float) -> str:
    minutes, remainder = divmod(max(0.0, seconds), 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{remainder:05.2f}"


def _gold_target(gold: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "meeting_summary": " ".join(gold["abstract"]),
        "topics": [],
        "decisions": gold["decisions"],
        "action_items": [
            {
                "task": item,
                "owner": None,
                "deadline": None,
                "evidence": "AMI abstractive action annotation",
                "confidence": 1.0,
            }
            for item in gold["actions"]
        ],
        "risks": gold["problems"],
        "assumptions": [],
        "missing_information": [],
        "uncertainties": [],
    }


def build(corpus_root: Path, annotations_root: Path, output_dir: Path) -> dict[str, Any]:
    workspace_root = corpus_root.resolve().parent
    meeting_dirs = sorted(
        path for path in corpus_root.iterdir() if path.is_dir() and MEETING_RE.fullmatch(path.name)
    )
    metadata = read_meeting_metadata(annotations_root / "corpusResources" / "meetings.xml")
    meeting_records: list[dict[str, Any]] = []
    asr_records: list[dict[str, Any]] = []
    sft_records: list[dict[str, Any]] = []
    warnings: list[str] = []

    for meeting_dir in meeting_dirs:
        meeting_id = meeting_dir.name
        if meeting_id not in metadata:
            warnings.append(f"No official meeting metadata for {meeting_id}")
            continue
        meta = metadata[meeting_id]
        audio_paths = sorted((meeting_dir / "audio").glob("*.Mix-Headset.wav"))
        if len(audio_paths) != 1:
            warnings.append(
                f"Expected one mixed-headset WAV for {meeting_id}; found {len(audio_paths)}"
            )
            continue
        audio_path = audio_paths[0]
        audio_meta = read_audio_metadata(audio_path)
        utterances: list[dict[str, Any]] = []
        for agent in "ABCD":
            words_path = annotations_root / "words" / f"{meeting_id}.{agent}.words.xml"
            segments_path = annotations_root / "segments" / f"{meeting_id}.{agent}.segments.xml"
            if not words_path.exists() or not segments_path.exists():
                warnings.append(f"Missing word/segment annotation for {meeting_id}.{agent}")
                continue
            utterances.extend(
                read_segments(
                    segments_path,
                    words_path,
                    meeting_id,
                    agent,
                    meta["speakers"].get(agent, {}),
                )
            )
        utterances.sort(key=lambda item: (item["start"], item["end"], item["agent"]))
        audio_relative = _relative(audio_path, workspace_root)
        for utterance in utterances:
            asr_records.append(
                {
                    **utterance,
                    "audio_path": audio_relative,
                    "audio_channel": "mono_mixed_headset",
                    "split": meta["split"],
                    "project_group": PROJECT_RE.match(meeting_id).group(1),  # type: ignore[union-attr]
                    "reference_quality": "official_manual_words_forced_aligned",
                    "overlap_possible": True,
                }
            )
        transcript = "\n".join(
            f"[{_format_timestamp(item['start'])}] {item['role']} ({item['agent']}): {item['text']}"
            for item in utterances
        )
        gold = read_abstractive(annotations_root / "abstractive" / f"{meeting_id}.abssumm.xml")
        participants = read_participant_summaries(
            annotations_root / "participantSummaries", meeting_id
        )
        project_match = PROJECT_RE.match(meeting_id)
        record = {
            "schema_version": "1.0",
            "meeting_id": meeting_id,
            "project_group": project_match.group(1) if project_match else meeting_id,
            "split": meta["split"],
            "official_seen_type": meta["official_seen_type"],
            "meeting_domain": "software_product_remote_control_design",
            "audio_path": audio_relative,
            "audio": audio_meta,
            "speakers": meta["speakers"],
            "utterance_count": len(utterances),
            "transcript_text": transcript,
            "gold": {
                "meeting_summary_sentences": gold["abstract"],
                "actions": gold["actions"],
                "decisions": gold["decisions"],
                "problems": gold["problems"],
            },
            "participant_summaries": participants,
            "limitations": [
                "Mixed-headset audio can contain overlapping speakers.",
            ],
        }
        meeting_records.append(record)
        sft_records.append(
            {
                "meeting_id": meeting_id,
                "project_group": record["project_group"],
                "split": meta["split"],
                "messages": [
                    {
                        "role": "system",
                        "content": "Extract supported meeting intelligence as one valid JSON object only.",
                    },
                    {"role": "user", "content": f"TRANSCRIPT:\n{transcript}"},
                    {
                        "role": "assistant",
                        "content": json.dumps(_gold_target(gold), ensure_ascii=False),
                    },
                ],
                "warning": "Only 30 local meetings exist; use primarily for evaluation or parameter-efficient experiments, not full-model training.",
            }
        )

    extension_counts: Counter[str] = Counter()
    raw_file_count = 0
    raw_bytes = 0
    for meeting_dir in meeting_dirs:
        for path in meeting_dir.rglob("*"):
            if path.is_file():
                raw_file_count += 1
                raw_bytes += path.stat().st_size
                extension_counts[path.suffix.lower() or "[none]"] += 1
    split_counts = Counter(record["split"] for record in meeting_records)
    audio_seconds = sum(record["audio"]["duration_seconds"] for record in meeting_records)
    audit = {
        "schema_version": "1.0",
        "corpus_root": _relative(corpus_root, workspace_root),
        "annotations_root": _relative(annotations_root, workspace_root),
        "meeting_count": len(meeting_records),
        "project_group_count": len({record["project_group"] for record in meeting_records}),
        "split_counts": dict(sorted(split_counts.items())),
        "audio_hours": round(audio_seconds / 3600, 3),
        "audio_seconds": round(audio_seconds, 3),
        "asr_segment_count": len(asr_records),
        "raw_file_count": raw_file_count,
        "raw_size_bytes": raw_bytes,
        "extension_counts": dict(extension_counts.most_common()),
        "gold_coverage": {
            key: sum(bool(record["gold"][key]) for record in meeting_records)
            for key in ("meeting_summary_sentences", "actions", "decisions", "problems")
        },
        "warnings": warnings,
        "safety_decisions": [
            "Official AMI train/development/unseen metadata determines project-grouped train/validation/test splits.",
            "All local meetings are software-product design meetings; they must not train a general meeting-type classifier.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    rttm_path = output_dir / "diarization_reference.rttm"
    rttm_turn_count = _write_rttm(rttm_path, asr_records)
    diarization_records = [
        {
            "meeting_id": record["meeting_id"],
            "project_group": record["project_group"],
            "split": record["split"],
            "audio_path": record["audio_path"],
            "reference_rttm_path": _relative(rttm_path, workspace_root),
            "expected_speaker_count": len(record["speakers"]),
            "reference_quality": "official_manual_speaker_segments",
        }
        for record in meeting_records
    ]
    counts = {
        "meeting_records.jsonl": _write_jsonl(
            output_dir / "meeting_records.jsonl", meeting_records
        ),
        "asr_segments.jsonl": _write_jsonl(output_dir / "asr_segments.jsonl", asr_records),
        "text_sft.jsonl": _write_jsonl(output_dir / "text_sft.jsonl", sft_records),
        "diarization_manifest.jsonl": _write_jsonl(
            output_dir / "diarization_manifest.jsonl", diarization_records
        ),
        "diarization_reference.rttm": rttm_turn_count,
    }
    audit["generated_records"] = counts
    (output_dir / "corpus_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=workspace_root / "amicorpus")
    parser.add_argument(
        "--annotations-root",
        type=Path,
        default=workspace_root / "amicorpus" / "annotations" / "manual_1.6.2",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "generated")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build(
        args.corpus_root.resolve(), args.annotations_root.resolve(), args.output_dir.resolve()
    )
    summary = {
        key: audit[key]
        for key in (
            "meeting_count",
            "project_group_count",
            "split_counts",
            "audio_hours",
            "asr_segment_count",
            "gold_coverage",
            "warnings",
            "generated_records",
        )
    }
    summary["audit_path"] = str(args.output_dir.resolve() / "corpus_audit.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
