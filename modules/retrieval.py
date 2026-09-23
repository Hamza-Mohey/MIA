"""Hybrid lexical and semantic retrieval over timestamped meeting turns."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from modules.config import (
    HF_CACHE_DIR,
    RETRIEVAL_DENSE_WEIGHT,
    RETRIEVAL_MAX_CHARACTERS,
    RETRIEVAL_MIN_SCORE,
    RETRIEVAL_MODEL_NAME,
    cached_model_snapshot,
)
from modules.evidence import render_source_segment

_TOKEN = re.compile(r"[\w']+")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "did",
    "do",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "the",
    "that",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _terms(text: str) -> set[str]:
    return {term.casefold() for term in _TOKEN.findall(text) if term.casefold() not in _STOP_WORDS}


def _lexical_score(query: str, passage: str) -> float:
    query_terms = _terms(query)
    passage_terms = _terms(passage)
    if not query_terms or not passage_terms:
        return 0.0
    overlap = len(query_terms & passage_terms) / len(query_terms)
    phrase_bonus = 0.15 if query.casefold() in passage.casefold() else 0.0
    return min(1.0, overlap + phrase_bonus)


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


@lru_cache(maxsize=1)
def load_embedding_model():
    """Keep the small CPU embedding model resident across questions in one app process."""
    snapshot = cached_model_snapshot(RETRIEVAL_MODEL_NAME)
    if snapshot is None:
        raise RuntimeError(
            "The retrieval model is not cached. Run `python -m "
            "modules.prefetch_retrieval_model` once while online."
        )
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        str(snapshot),
        device="cpu",
        cache_folder=str(HF_CACHE_DIR),
        local_files_only=True,
    )


def _default_embed(texts: list[str]) -> list[list[float]]:
    vectors = load_embedding_model().encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def build_retrieval_index(
    segments: list[dict[str, Any]],
    embedder: Callable[[list[str]], list[list[float]]] | None = None,
) -> dict[str, Any]:
    """Embed individual speaker turns while retaining their source metadata."""
    texts = [segment.get("text", "") for segment in segments]
    embedding_error = None
    load_seconds = 0.0
    inference_seconds = 0.0
    try:
        if not texts:
            embeddings = []
        elif embedder is not None:
            inference_started = time.perf_counter()
            embeddings = embedder(texts)
            inference_seconds = time.perf_counter() - inference_started
        else:
            load_started = time.perf_counter()
            model = load_embedding_model()
            load_seconds = time.perf_counter() - load_started
            inference_started = time.perf_counter()
            vectors = model.encode(
                texts,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            embeddings = vectors.tolist()
            inference_seconds = time.perf_counter() - inference_started
    except (ImportError, RuntimeError, ValueError) as exc:
        embeddings = []
        embedding_error = f"{type(exc).__name__}: {exc}"
    return {
        "model": RETRIEVAL_MODEL_NAME,
        "segments": segments,
        "embeddings": embeddings,
        "embedding_error": embedding_error,
        "_runtime": {
            "model_load_seconds": round(load_seconds, 3),
            "inference_seconds": round(inference_seconds, 3),
        },
    }


def retrieve_evidence(
    question: str,
    index: dict[str, Any],
    *,
    max_characters: int = RETRIEVAL_MAX_CHARACTERS,
    dense_weight: float = RETRIEVAL_DENSE_WEIGHT,
    minimum_score: float = RETRIEVAL_MIN_SCORE,
    query_embedding: list[float] | None = None,
) -> dict[str, Any]:
    """Rank turns by hybrid score, then include neighbouring context within a budget."""
    segments = index.get("segments", [])
    embeddings = index.get("embeddings", [])
    semantic_available = bool(segments and embeddings)
    embedding_error = index.get("embedding_error")
    query_embedding_seconds = 0.0
    if semantic_available and query_embedding is None:
        try:
            query_started = time.perf_counter()
            query_embedding = _default_embed([question])[0]
            query_embedding_seconds = time.perf_counter() - query_started
        except (ImportError, RuntimeError, ValueError) as exc:
            semantic_available = False
            embedding_error = f"{type(exc).__name__}: {exc}"

    scored: list[dict[str, Any]] = []
    for position, segment in enumerate(segments):
        lexical = _lexical_score(question, segment.get("text", ""))
        dense = max(0.0, _dot(query_embedding, embeddings[position])) if semantic_available else 0.0
        combined = dense_weight * dense + (1.0 - dense_weight) * lexical
        if not semantic_available:
            combined = lexical
        scored.append(
            {
                "position": position,
                "segment_id": segment["id"],
                "lexical_score": round(lexical, 6),
                "semantic_score": round(dense, 6),
                "score": round(combined, 6),
            }
        )
    scored.sort(key=lambda item: (-item["score"], item["position"]))
    best_score = scored[0]["score"] if scored else 0.0

    selected_positions: set[int] = set()
    used_characters = 0
    for candidate in scored[:4]:
        for position in (
            candidate["position"] - 1,
            candidate["position"],
            candidate["position"] + 1,
        ):
            if position < 0 or position >= len(segments) or position in selected_positions:
                continue
            rendered = render_source_segment(segments[position])
            if selected_positions and used_characters + len(rendered) + 1 > max_characters:
                continue
            selected_positions.add(position)
            used_characters += len(rendered) + 1

    selected = [segments[position] for position in sorted(selected_positions)]
    evidence_text = "\n".join(render_source_segment(segment) for segment in selected)
    return {
        "status": "supported" if best_score >= minimum_score else "low_confidence",
        "mode": "hybrid" if semantic_available else "lexical_fallback",
        "best_score": best_score,
        "selected_segments": selected,
        "evidence_text": evidence_text,
        "estimated_tokens": math.ceil(len(evidence_text) / 4),
        "ranked_candidates": scored[:10],
        "embedding_error": embedding_error,
        "query_embedding_seconds": round(query_embedding_seconds, 3),
    }
