"""Run local Qwen extraction against one QMSum transcript for diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from modules.evidence import build_source_segments, chunk_source_segments
from modules.text_analysis import analyse_transcript_chunks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("meeting", type=Path)
    parser.add_argument("--start-turn", type=int, default=0)
    parser.add_argument("--end-turn", type=int)
    args = parser.parse_args()
    meeting = json.loads(args.meeting.read_text(encoding="utf-8"))
    turns = meeting.get("meeting_transcripts", [])[args.start_turn : args.end_turn]
    transcript = "\n".join(f"{turn['speaker']}: {turn['content']}" for turn in turns)
    chunks = chunk_source_segments(build_source_segments(transcript))
    print(f"Analysing {len(chunks)} chunks / {len(transcript):,} characters...", flush=True)
    result = analyse_transcript_chunks([chunk["text"] for chunk in chunks])
    diagnostic = {
        "runtime": result.get("_runtime", {}),
        "chunks": [
            {
                "summary": analysis.get("meeting_summary"),
                "topics": analysis.get("topics", []),
                "key_points": analysis.get("key_points", []),
                "decisions": analysis.get("decisions", []),
                "actions": analysis.get("action_items", []),
                "uncertainties": analysis.get("uncertainties", []),
            }
            for analysis in result.get("analyses", [])
        ],
    }
    print(json.dumps(diagnostic, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
