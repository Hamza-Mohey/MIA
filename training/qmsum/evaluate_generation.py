"""Compare base and LoRA Qwen generation, not only teacher-forced loss."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.qmsum.train_qwen_lora import ROOT, SYSTEM_PROMPT, read_jsonl

GROUNDING_CHECKS = [
    {
        "question": "When is the customer review?",
        "transcript": (
            "SPEAKER_00: We approved moving the customer review to Friday at 10 AM.\n"
            "SPEAKER_01: I will send the revised agenda by Wednesday."
        ),
        "required_terms": ["friday", "10"],
    }
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "training" / "qmsum" / "generated" / "specific_queries.jsonl",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=ROOT / "artifacts" / "qwen_qmsum_qa_lora",
    )
    parser.add_argument("--eval-limit", type=int, default=16)
    parser.add_argument("--max-input-length", type=int, default=1792)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "training" / "qmsum" / "runs" / "generation_comparison.json",
    )
    return parser.parse_args()


def normalized_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def token_f1(reference: str, prediction: str) -> float:
    reference_tokens = normalized_tokens(reference)
    prediction_tokens = normalized_tokens(prediction)
    if not reference_tokens or not prediction_tokens:
        return float(reference_tokens == prediction_tokens)
    overlap = sum((Counter(reference_tokens) & Counter(prediction_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(reference: str, prediction: str) -> float:
    reference_tokens = normalized_tokens(reference)
    prediction_tokens = normalized_tokens(prediction)
    if not reference_tokens or not prediction_tokens:
        return float(reference_tokens == prediction_tokens)
    previous = [0] * (len(prediction_tokens) + 1)
    for reference_token in reference_tokens:
        current = [0]
        for index, prediction_token in enumerate(prediction_tokens, start=1):
            if reference_token == prediction_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    common = previous[-1]
    precision = common / len(prediction_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if common else 0.0


def generate_answer(model, tokenizer, question: str, transcript: str, args) -> str:
    user = f"QUESTION:\n{question}\n\nTRANSCRIPT:\n{transcript}"
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        prompt,
        add_special_tokens=False,
        truncation=True,
        max_length=args.max_input_length,
        return_tensors="pt",
    ).to("cuda")
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    answer_ids = generated[0][inputs.input_ids.shape[1] :]
    return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()


def evaluate_model(model, tokenizer, records, args) -> dict:
    model.eval()
    rows = []
    for record in records:
        prediction = generate_answer(
            model, tokenizer, record["query"], record["relevant_transcript"], args
        )
        rows.append(
            {
                "meeting_id": record["meeting_id"],
                "query": record["query"],
                "reference": record["answer"],
                "prediction": prediction,
                "token_f1": token_f1(record["answer"], prediction),
                "rouge_l_f1": rouge_l_f1(record["answer"], prediction),
            }
        )
    checks = []
    for check in GROUNDING_CHECKS:
        prediction = generate_answer(model, tokenizer, check["question"], check["transcript"], args)
        normalized = prediction.casefold()
        checks.append(
            {
                "question": check["question"],
                "prediction": prediction,
                "passed": all(term in normalized for term in check["required_terms"]),
            }
        )
    return {
        "mean_token_f1": sum(row["token_f1"] for row in rows) / len(rows),
        "mean_rouge_l_f1": sum(row["rouge_l_f1"] for row in rows) / len(rows),
        "grounding_checks_passed": all(check["passed"] for check in checks),
        "grounding_checks": checks,
        "examples": rows,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this configured evaluation.")
    if not args.adapter.exists():
        raise SystemExit(f"Adapter not found: {args.adapter}")
    records = [record for record in read_jsonl(args.manifest) if record["split"] == "validation"][
        : args.eval_limit
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=str(ROOT / "hf_cache"))
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        cache_dir=str(ROOT / "hf_cache"),
        low_cpu_mem_usage=True,
    ).to("cuda")
    base_result = evaluate_model(model, tokenizer, records, args)
    adapted_model = PeftModel.from_pretrained(model, str(args.adapter))
    adapted_result = evaluate_model(adapted_model, tokenizer, records, args)
    result = {
        "base_model": args.model,
        "adapter": str(args.adapter),
        "validation_examples": len(records),
        "base": base_result,
        "adapted": adapted_result,
        "adapter_improved_token_f1": (
            adapted_result["mean_token_f1"] > base_result["mean_token_f1"]
        ),
        "adapter_improved_rouge_l": (
            adapted_result["mean_rouge_l_f1"] > base_result["mean_rouge_l_f1"]
        ),
        "promotion_gate_passed": (
            adapted_result["mean_token_f1"] > base_result["mean_token_f1"]
            and adapted_result["mean_rouge_l_f1"] > base_result["mean_rouge_l_f1"]
            and adapted_result["grounding_checks_passed"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    compact = {
        "base_model": result["base_model"],
        "adapter": result["adapter"],
        "validation_examples": result["validation_examples"],
        "base": {key: value for key, value in base_result.items() if key != "examples"},
        "adapted": {key: value for key, value in adapted_result.items() if key != "examples"},
        "promotion_gate_passed": result["promotion_gate_passed"],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
