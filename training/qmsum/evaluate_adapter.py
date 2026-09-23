"""Compare Qwen base and LoRA validation loss on the same held-out records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from training.qmsum.train_qwen_lora import (
    ROOT,
    CausalCollator,
    MeetingQADataset,
    read_jsonl,
)


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
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--eval-limit", type=int, default=64)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "training" / "qmsum" / "runs" / "loss_comparison.json",
    )
    return parser.parse_args()


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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = MeetingQADataset(records, tokenizer, args.max_length)
    collator = CausalCollator(tokenizer.pad_token_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        cache_dir=str(ROOT / "hf_cache"),
        low_cpu_mem_usage=True,
    ).to("cuda")
    evaluation_args = TrainingArguments(
        output_dir=str(args.output.parent / "evaluation_cache"),
        per_device_eval_batch_size=1,
        fp16=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
    )
    base_metrics = Trainer(
        model=model,
        args=evaluation_args,
        eval_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
    ).evaluate()

    adapted_model = PeftModel.from_pretrained(model, str(args.adapter))
    adapted_metrics = Trainer(
        model=adapted_model,
        args=evaluation_args,
        eval_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
    ).evaluate()
    base_loss = float(base_metrics["eval_loss"])
    adapted_loss = float(adapted_metrics["eval_loss"])
    result = {
        "base_model": args.model,
        "adapter": str(args.adapter),
        "validation_examples": len(records),
        "base_eval_loss": base_loss,
        "adapted_eval_loss": adapted_loss,
        "absolute_loss_change": adapted_loss - base_loss,
        "relative_loss_change_percent": ((adapted_loss - base_loss) / base_loss) * 100,
        "adapter_improved_loss": adapted_loss < base_loss,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
