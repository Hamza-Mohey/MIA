"""LoRA smoke training for general-meeting Q&A using QMSum relevant spans."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[2]
SYSTEM_PROMPT = (
    "You are a factual general meeting assistant. Answer the question using only the supplied "
    "speaker-attributed transcript. If the answer is not supported, say that it was not established."
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class MeetingQADataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer: Any, max_length: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        record = self.records[index]
        user = f"QUESTION:\n{record['query']}\n\nTRANSCRIPT:\n{record['relevant_transcript']}"
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        answer = record["answer"].strip() + self.tokenizer.eos_token
        answer_ids = self.tokenizer(answer, add_special_tokens=False).input_ids
        # Always retain supervised answer tokens. Simply truncating prompt+answer
        # can fill the window with transcript text and yield an all-masked label
        # row (and therefore NaN loss).
        answer_ids = answer_ids[: max(1, self.max_length // 2)]
        prompt_budget = max(1, self.max_length - len(answer_ids))
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=prompt_budget,
        ).input_ids
        input_ids = list(prompt_ids) + list(answer_ids)
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + list(answer_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@dataclass
class CausalCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        length = max(len(item["input_ids"]) for item in features)
        result: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            padding = length - len(item["input_ids"])
            result["input_ids"].append(item["input_ids"] + [self.pad_token_id] * padding)
            result["attention_mask"].append(item["attention_mask"] + [0] * padding)
            result["labels"].append(item["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "training" / "qmsum" / "generated" / "specific_queries.jsonl",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "training" / "qmsum" / "runs" / "qwen_lora"
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--eval-limit", type=int, default=64)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this configured training run.")
    records = read_jsonl(args.manifest)
    train_records = [record for record in records if record["split"] == "train"]
    validation_records = [record for record in records if record["split"] == "validation"][
        : args.eval_limit
    ]
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=str(ROOT / "hf_cache"))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        cache_dir=str(ROOT / "hf_cache"),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=min(10, max(1, args.max_steps // 10)),
        fp16=True,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=max(5, args.max_steps),
        save_strategy="steps",
        save_steps=max(5, args.max_steps),
        logging_steps=1,
        save_total_limit=2,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=MeetingQADataset(train_records, tokenizer, args.max_length),
        eval_dataset=MeetingQADataset(validation_records, tokenizer, args.max_length),
        data_collator=CausalCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    result = trainer.train()
    final_dir = args.output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    summary = {
        **result.metrics,
        "base_model": args.model,
        "train_examples": len(train_records),
        "validation_examples": len(validation_records),
        "max_steps": args.max_steps,
        "max_length": args.max_length,
        "gpu": torch.cuda.get_device_name(0),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
