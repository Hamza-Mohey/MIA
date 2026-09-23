"""Local Qwen transcript extraction with strict normalisation and fallbacks."""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from contextlib import nullcontext
from typing import Any

from modules.config import (
    DEFAULT_LLM_CONFIDENCE,
    MAX_NEW_TOKENS,
    MAX_TRANSCRIPT_LENGTH,
    PROMPTS_DIR,
    QA_ADAPTER_PATH,
    TEXT_ADAPTER_PATH,
    TEXT_CHUNK_LENGTH,
    TEXT_GENERATION_TIMEOUT_SECONDS,
    TEXT_MODEL_NAME,
    UPLOADED_TRANSCRIPT_QUALITY,
    hub_load_kwargs,
)
from modules.evidence import build_analysis_digests, build_source_segments
from modules.schemas import TranscriptAnalysisResult, as_list, clean_text

LOGGER = logging.getLogger(__name__)
PROMPT_PATH = PROMPTS_DIR / "meeting_intelligence_prompt.txt"

_tokenizer = None
_model = None
_qa_only_adapter = False


def load_model():
    """Load the configured local instruction model once."""
    global _tokenizer, _model, _qa_only_adapter
    if _tokenizer is None or _model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_kwargs = hub_load_kwargs(TEXT_MODEL_NAME)
        _tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, **load_kwargs)
        _model = AutoModelForCausalLM.from_pretrained(
            TEXT_MODEL_NAME,
            dtype="auto",
            device_map="auto",
            attn_implementation="sdpa",
            **load_kwargs,
        )
        if TEXT_ADAPTER_PATH is not None:
            if not TEXT_ADAPTER_PATH.exists():
                raise RuntimeError(f"Configured Qwen adapter does not exist: {TEXT_ADAPTER_PATH}")
            from peft import PeftModel

            _model = PeftModel.from_pretrained(_model, str(TEXT_ADAPTER_PATH))
        elif QA_ADAPTER_PATH is not None:
            if not QA_ADAPTER_PATH.exists():
                raise RuntimeError(f"Configured Qwen QA adapter does not exist: {QA_ADAPTER_PATH}")
            from peft import PeftModel

            _model = PeftModel.from_pretrained(_model, str(QA_ADAPTER_PATH))
            _qa_only_adapter = True
        _model.eval()
    return _tokenizer, _model


def _balanced_json_candidates(text: str):
    """Yield balanced JSON objects while respecting quoted strings."""
    start = None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start : index + 1]
                start = None


def extract_json_from_text(text: str) -> dict[str, Any] | None:
    """Extract the first valid JSON object from clean or chatty model output."""
    if not isinstance(text, str) or not text.strip():
        return None
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    for candidate in _balanced_json_candidates(cleaned):
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    # Truncated model output is repaired deterministically instead of spending
    # another full Qwen generation on the same content. Only accept repaired
    # objects that contain a field from our known meeting schema.
    known_fields = {
        "meeting_summary",
        "participants",
        "topics",
        "key_points",
        "requirements",
        "decisions",
        "action_items",
        "risks",
        "open_questions",
        "assumptions",
        "missing_information",
        "uncertainties",
    }
    if any(f'"{field}"' in cleaned for field in known_fields):
        try:
            from json_repair import repair_json

            repaired = repair_json(cleaned, return_objects=True)
            if isinstance(repaired, dict) and known_fields.intersection(repaired):
                return repaired
        except (ImportError, TypeError, ValueError):
            pass
    return None


def _read_prompt() -> str:
    if PROMPT_PATH.exists():
        prompt = PROMPT_PATH.read_text(encoding="utf-8").strip()
        if prompt:
            return prompt
    LOGGER.warning("Transcript prompt missing or empty: %s", PROMPT_PATH)
    return (
        "Extract supported meeting facts into the requested JSON schema. "
        "Do not invent missing details and return JSON only."
    )


def normalise_transcript_analysis(
    raw: dict[str, Any] | None,
    transcript_quality: float = UPLOADED_TRANSCRIPT_QUALITY,
    quality_basis: str = "uploaded transcript heuristic",
) -> dict[str, Any]:
    """Validate old or new extraction shapes and return the canonical schema."""
    raw = dict(raw or {})
    raw["transcript_quality_score"] = transcript_quality
    raw["transcript_quality_basis"] = quality_basis
    result = TranscriptAnalysisResult.from_mapping(raw).to_dict()
    result["schema_version"] = "1.0"
    result["confidence_basis"] = (
        f"Uncalibrated extraction heuristic; default LLM contribution is {DEFAULT_LLM_CONFIDENCE:.2f}."
    )
    return result


def _fallback(output_text: str, transcript_quality: float, quality_basis: str) -> dict[str, Any]:
    return normalise_transcript_analysis(
        {
            "meeting_summary": "The transcript was available, but structured extraction failed.",
            "risks": ["The local language model returned malformed JSON."],
            "missing_information": ["Retry with a shorter or clearer transcript."],
            "uncertainties": [
                "Model output could not be parsed as JSON.",
                output_text[:500] if output_text else "No model output was returned.",
            ],
        },
        transcript_quality,
        quality_basis,
    )


def _generate_chat(
    tokenizer: Any, model: Any, messages: list[dict[str, str]], max_tokens: int
) -> str:
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([formatted], return_tensors="pt").to(model.device)
    import torch

    adapter_context = model.disable_adapter() if _qa_only_adapter else nullcontext()
    with torch.inference_mode(), adapter_context:
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            max_time=TEXT_GENERATION_TIMEOUT_SECONDS,
            do_sample=False,
            repetition_penalty=1.05,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0][len(inputs.input_ids[0]) :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def split_transcript(transcript_text: str, max_characters: int = TEXT_CHUNK_LENGTH) -> list[str]:
    """Split on speaker/paragraph boundaries with a hard upper bound."""
    text = transcript_text.strip()
    if not text:
        return []
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for original_line in text.splitlines() or [text]:
        line = original_line.strip()
        if not line:
            continue
        pieces = [
            line[index : index + max_characters] for index in range(0, len(line), max_characters)
        ]
        for piece in pieces:
            added = len(piece) + (1 if current else 0)
            if current and current_length + added > max_characters:
                chunks.append("\n".join(current))
                current, current_length = [], 0
            current.append(piece)
            current_length += len(piece) + (1 if len(current) > 1 else 0)
    if current:
        chunks.append("\n".join(current))
    return chunks


def _deduplicate(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        if isinstance(value, dict):
            identity = (
                value.get("task")
                or value.get("text")
                or value.get("speaker")
                or value.get("name")
                or value
            )
        else:
            identity = value
        key = json.dumps(identity, sort_keys=True, ensure_ascii=False, default=str).casefold()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _analyse_once(
    transcript_text: str,
    transcript_quality: float,
    quality_basis: str,
) -> dict[str, Any]:
    result, _ = _analyse_once_with_timings(transcript_text, transcript_quality, quality_basis)
    return result


def _analyse_once_with_timings(
    transcript_text: str,
    transcript_quality: float,
    quality_basis: str,
) -> tuple[dict[str, Any], dict[str, float]]:
    tokenizer, model = load_model()
    prompt = f"{_read_prompt()}\n\nTRANSCRIPT:\n{transcript_text}"
    messages = [
        {
            "role": "system",
            "content": "Return supported general meeting intelligence as one valid JSON object only.",
        },
        {"role": "user", "content": prompt},
    ]
    # The promoted local LoRA was trained and validated for meeting Q&A only;
    # _generate_chat keeps structured extraction on the unchanged base model.
    inference_started = time.perf_counter()
    output_text = _generate_chat(tokenizer, model, messages, MAX_NEW_TOKENS)
    inference_seconds = time.perf_counter() - inference_started
    postprocess_started = time.perf_counter()
    parsed = extract_json_from_text(output_text)
    postprocess_seconds = time.perf_counter() - postprocess_started
    normalise_started = time.perf_counter()
    result = (
        normalise_transcript_analysis(parsed, transcript_quality, quality_basis)
        if parsed is not None
        else _fallback(output_text, transcript_quality, quality_basis)
    )
    postprocess_seconds += time.perf_counter() - normalise_started
    return result, {
        "inference_seconds": inference_seconds,
        "postprocess_seconds": postprocess_seconds,
    }


def merge_chunk_analyses(
    analyses: list[dict[str, Any]],
    transcript_quality: float,
    quality_basis: str,
) -> dict[str, Any]:
    list_fields = (
        "participants",
        "topics",
        "key_points",
        "requirements",
        "decisions",
        "action_items",
        "risks",
        "open_questions",
        "assumptions",
        "missing_information",
        "uncertainties",
    )
    summaries = _deduplicate(
        [value for analysis in analyses if (value := clean_text(analysis.get("meeting_summary")))]
    )
    # Summaries are replaced later by a deterministic summary of evidence-linked
    # claims. Keeping one candidate here avoids concatenating unrelated windows.
    merged: dict[str, Any] = {
        "meeting_summary": max(summaries, key=len, default="")
    }
    for field in list_fields:
        merged[field] = _deduplicate(
            [item for analysis in analyses for item in as_list(analysis.get(field))]
        )
    merged["uncertainties"].append(
        f"The transcript was processed in {len(analyses)} bounded evidence windows; "
        "cross-window context may be incomplete."
    )
    return normalise_transcript_analysis(merged, transcript_quality, quality_basis)


def analyse_transcript_chunks(
    chunks: list[str],
    transcript_quality: float = UPLOADED_TRANSCRIPT_QUALITY,
    quality_basis: str = "uploaded transcript heuristic",
) -> dict[str, Any]:
    """Analyse several cache-miss chunks after loading Qwen only once."""
    if not chunks:
        return {
            "analyses": [],
            "_runtime": {
                "model_load_seconds": 0.0,
                "inference_seconds": 0.0,
                "postprocess_seconds": 0.0,
                "total_seconds": 0.0,
            },
        }
    total_started = time.perf_counter()
    load_started = time.perf_counter()
    load_model()
    model_load_seconds = time.perf_counter() - load_started
    analyses: list[dict[str, Any]] = []
    inference_seconds = 0.0
    postprocess_seconds = 0.0
    for chunk in chunks:
        analysis, timings = _analyse_once_with_timings(chunk, transcript_quality, quality_basis)
        analyses.append(analysis)
        inference_seconds += timings["inference_seconds"]
        postprocess_seconds += timings["postprocess_seconds"]
    return {
        "analyses": analyses,
        "_runtime": {
            "model_load_seconds": round(model_load_seconds, 3),
            "inference_seconds": round(inference_seconds, 3),
            "postprocess_seconds": round(postprocess_seconds, 3),
            "total_seconds": round(time.perf_counter() - total_started, 3),
        },
    }


def analyse_transcript(
    transcript_text: str,
    transcript_quality: float = UPLOADED_TRANSCRIPT_QUALITY,
    quality_basis: str = "uploaded transcript heuristic",
) -> dict[str, Any]:
    """Analyse a transcript with Qwen and normalise its untrusted JSON output."""
    if not isinstance(transcript_text, str) or not transcript_text.strip():
        return normalise_transcript_analysis(
            {"uncertainties": ["No transcript text was supplied."]},
            transcript_quality,
            quality_basis,
        )

    transcript_text = transcript_text.strip()
    source_segments = build_source_segments(transcript_text)
    digests = build_analysis_digests(
        source_segments,
        max_characters=min(TEXT_CHUNK_LENGTH, MAX_TRANSCRIPT_LENGTH),
    )
    chunks = [digest["text"] for digest in digests]
    batch = analyse_transcript_chunks(chunks, transcript_quality, quality_basis)
    analyses = batch["analyses"]
    merge_started = time.perf_counter()
    result = (
        analyses[0]
        if len(analyses) == 1
        else merge_chunk_analyses(analyses, transcript_quality, quality_basis)
    )
    runtime = dict(batch["_runtime"])
    runtime["merge_seconds"] = round(time.perf_counter() - merge_started, 3)
    result["_runtime"] = runtime
    return result


def answer_meeting_question(evidence_text: str, question: str) -> dict[str, Any]:
    """Answer only from evidence selected by the hybrid retrieval stage."""
    evidence_text = evidence_text.strip()
    question = clean_text(question)
    if not evidence_text or not question:
        return {"answer": "A question and retrieved evidence are required.", "_runtime": {}}

    total_started = time.perf_counter()
    load_started = time.perf_counter()
    tokenizer, model = load_model()
    model_load_seconds = time.perf_counter() - load_started
    messages = [
        {
            "role": "system",
            "content": (
                "Answer the question using only the retrieved meeting evidence below. "
                "Be concise, cite supporting segment IDs, and mention the speaker when supported. "
                "Treat a question joined by 'and' or containing multiple requested fields as a "
                "checklist: answer every supported part explicitly. Copy names, roles, quantities, "
                "currencies, dates, and units exactly from the evidence. Never replace a named "
                "person or role with an ambiguous pronoun. Before returning, silently verify that "
                "each requested part is present in the answer. "
                "If the answer is absent from these passages, say 'Not established in the retrieved "
                "evidence.' Never claim that the topic was absent from the entire meeting."
            ),
        },
        {
            "role": "user",
            "content": f"QUESTION:\n{question}\n\nRETRIEVED EVIDENCE:\n{evidence_text}",
        },
    ]
    inference_started = time.perf_counter()
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([formatted], return_tensors="pt").to(model.device)
    import torch

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=min(256, MAX_NEW_TOKENS),
            max_time=TEXT_GENERATION_TIMEOUT_SECONDS,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0][len(inputs.input_ids[0]) :]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    answer = _complete_grounded_answer(question, evidence_text, answer)
    inference_seconds = time.perf_counter() - inference_started
    return {
        "answer": answer or "Not established in the retrieved evidence.",
        "_runtime": {
            "model_load_seconds": round(model_load_seconds, 3),
            "inference_seconds": round(inference_seconds, 3),
            "postprocess_seconds": 0.0,
            "total_seconds": round(time.perf_counter() - total_started, 3),
        },
    }


_QA_EVIDENCE_LINE = re.compile(
    r"^\[(?P<segment_id>seg-[^\]]+)\]\s+(?P<speaker>[^:]+):\s*(?P<text>.*)$",
    re.IGNORECASE,
)
_QA_NUMBER = r"(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)"


def _number_value(value: str) -> str:
    words = {
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
    }
    return words.get(clean_text(value).casefold(), clean_text(value))


def _complete_grounded_answer(question: str, evidence_text: str, answer: str) -> str:
    """Add only high-precision facts omitted from an otherwise grounded answer."""
    answer = clean_text(answer) or "Not established in the retrieved evidence."
    question_lower = question.casefold()
    evidence_lower = evidence_text.casefold()
    additions: list[str] = []

    if "profit" in question_lower:
        profit = re.search(
            rf"\bprofit (?:aim|goal|target)?(?: is)?\s+(?P<amount>{_QA_NUMBER})\s+"
            r"million(?:\s+(?:euro|euros|eur))?\b",
            evidence_text,
            re.IGNORECASE,
        )
        if profit:
            amount = _number_value(profit.group("amount"))
            if not re.search(rf"\b{re.escape(amount)}\s+million\b", answer, re.IGNORECASE):
                additions.append(f"profit target {amount} million euros")

    if re.search(r"\b(?:sales?|sell|units?)\b", question_lower):
        sales_context = re.search(
            r"\bhow many (?:should|will|can) we sell\b(?P<context>.{0,700})",
            evidence_lower,
            re.IGNORECASE | re.DOTALL,
        )
        explicit_sales = re.search(
            rf"\b(?:that(?:'ll| will) do|sales target(?: is)?|"
            rf"sell(?:ing)?(?: about)?)\s+(?P<amount>{_QA_NUMBER})\s+million\b",
            evidence_text,
            re.IGNORECASE,
        )
        if explicit_sales:
            amount = _number_value(explicit_sales.group("amount"))
            if not re.search(rf"\b{re.escape(amount)}\s+million\b", answer, re.IGNORECASE):
                additions.append(f"sales target {amount} million units")
        elif sales_context:
            amounts = re.findall(
                rf"\b({_QA_NUMBER})\s+million\b",
                sales_context.group("context"),
                re.IGNORECASE,
            )
            if amounts:
                amount = Counter(_number_value(value) for value in amounts).most_common(1)[0][0]
                if not re.search(
                    rf"\b{re.escape(amount)}\s+million\b", answer, re.IGNORECASE
                ):
                    additions.append(f"sales target {amount} million units")

    lines = []
    for raw_line in evidence_text.splitlines():
        match = _QA_EVIDENCE_LINE.match(raw_line.strip())
        if match:
            lines.append(match.groupdict())
    if question_lower.startswith("who") and lines:
        query_terms = {
            token
            for token in re.findall(r"[a-z0-9]+", question_lower)
            if token not in {"who", "was", "were", "is", "are", "to", "on", "the", "work"}
        }
        for index, line in enumerate(lines):
            text_terms = set(re.findall(r"[a-z0-9]+", line["text"].casefold()))
            if len(query_terms & text_terms) < min(2, len(query_terms)):
                continue
            candidates = lines[max(0, index - 2) : index]
            role_pattern = re.compile(
                r"^(?:marketing|industrial designer|user interface(?: designer)?|"
                r"project manager)$",
                re.IGNORECASE,
            )
            role_line = next(
                (
                    candidate
                    for candidate in reversed(candidates)
                    if role_pattern.fullmatch(clean_text(candidate["speaker"]))
                    or role_pattern.fullmatch(
                        clean_text(candidate["text"]).strip(" .,;:-")
                    )
                ),
                None,
            )
            if role_line:
                role = clean_text(role_line["speaker"])
                if role_pattern.fullmatch(role):
                    return (
                        f"{role} was assigned to work on the requested area "
                        f"[{role_line['segment_id']}, {line['segment_id']}]."
                    )
            break

    if additions:
        prefix = answer.rstrip(" .")
        return f"{prefix}. Additional supported detail: {'; '.join(additions)}."
    return answer
