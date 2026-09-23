"""Cached transcript analysis over a compact, evidence-focused meeting digest."""

from __future__ import annotations

import time
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from modules.cache_keys import stage_cache_key
from modules.evidence import build_analysis_digests
from modules.text_analysis import merge_chunk_analyses


def analyse_segments_with_cache(
    segments: list[dict[str, Any]],
    cache: dict[str, Any],
    *,
    transcript_quality: float,
    quality_basis: str,
    versions: dict[str, Any],
    configuration: dict[str, Any],
    batch_analyser: Callable[..., dict[str, Any]],
    progress_callback: Callable[[str, float], None] | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Analyse a cached, timeline-balanced digest of the full transcript."""
    chunks = build_analysis_digests(segments)
    if not chunks:
        raise ValueError("At least one transcript source segment is required.")
    keys = [
        stage_cache_key(
            "transcript_analysis_chunk",
            chunk["text"],
            versions=versions,
            configuration={
                **configuration,
                "transcript_quality": transcript_quality,
                "quality_basis": quality_basis,
            },
        )
        for chunk in chunks
    ]
    missing_positions = [index for index, key in enumerate(keys) if key not in cache]
    worker_runtime: dict[str, Any] = {
        "model_load_seconds": 0.0,
        "inference_seconds": 0.0,
        "postprocess_seconds": 0.0,
        "total_seconds": 0.0,
        "subprocess_seconds": 0.0,
    }
    if missing_positions:
        if progress_callback:
            progress_callback(
                "Qwen is analysing the most relevant evidence from the meeting",
                0.22,
            )
        batch = batch_analyser(
            [chunks[position]["text"] for position in missing_positions],
            transcript_quality=transcript_quality,
            quality_basis=quality_basis,
        )
        analyses = batch.get("analyses", [])
        if len(analyses) != len(missing_positions):
            raise RuntimeError("Qwen worker returned an unexpected number of chunk analyses.")
        for position, analysis in zip(missing_positions, analyses, strict=True):
            cache[keys[position]] = deepcopy(analysis)
        worker_runtime.update(batch.get("_runtime", {}))
    elif progress_callback:
        progress_callback("Reusing the cached Qwen meeting analysis", 0.78)

    ordered = [deepcopy(cache[key]) for key in keys]
    if progress_callback:
        progress_callback("Validating the meeting analysis against its evidence", 0.82)
    merge_started = time.perf_counter()
    result = (
        ordered[0]
        if len(ordered) == 1
        else merge_chunk_analyses(ordered, transcript_quality, quality_basis)
    )
    worker_runtime["merge_seconds"] = round(time.perf_counter() - merge_started, 3)
    stats = {
        "chunks": len(chunks),
        "hits": len(chunks) - len(missing_positions),
        "misses": len(missing_positions),
        "source_segments": len(segments),
        "selected_segments": sum(
            int(chunk.get("selected_segments", 0)) for chunk in chunks
        ),
    }
    result["_runtime"] = {**worker_runtime, "cache": stats}
    return result, stats
