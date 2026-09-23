"""Parameter-efficient Whisper adaptation on the prepared AMI segments.

Designed for the local 8 GB RTX 3060 Ti: batch size one, gradient
checkpointing, FP16, and LoRA rather than full-model optimization.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class OffsetAudioDataset(Dataset):
    """Read only a manifest's requested WAV range instead of duplicating clips."""

    def __init__(self, records: list[dict[str, Any]], processor: WhisperProcessor):
        self.records = records
        self.processor = processor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path = ROOT / record["audio_path"]
        info = sf.info(path)
        start_frame = max(0, int(float(record["start"]) * info.samplerate))
        frame_count = max(1, int(float(record["duration"]) * info.samplerate))
        audio, sample_rate = sf.read(
            path,
            start=start_frame,
            frames=frame_count,
            dtype="float32",
            always_2d=False,
        )
        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            audio = audio.mean(axis=1)
        features = self.processor.feature_extractor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        ).input_features[0]
        labels = self.processor.tokenizer(record["text"]).input_ids
        return {"input_features": features, "labels": labels}


@dataclass
class WhisperCollator:
    processor: WhisperProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        inputs = [{"input_features": item["input_features"]} for item in features]
        batch = self.processor.feature_extractor.pad(inputs, return_tensors="pt")
        label_features = [{"input_ids": item["labels"]} for item in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "training" / "ami" / "generated" / "asr_segments.jsonl",
    )
    parser.add_argument("--model", default="openai/whisper-medium")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "training" / "ami" / "runs" / "whisper_medium_lora",
    )
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=128,
        help="Maximum validation segments per evaluation; use 0 for the complete split.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this configured training run.")
    training_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    records = read_jsonl(args.manifest)
    train_records = [record for record in records if record["split"] == "train"]
    validation_records = [record for record in records if record["split"] == "validation"]
    if args.eval_limit > 0:
        validation_records = validation_records[: args.eval_limit]
    processor = WhisperProcessor.from_pretrained(
        args.model, language="English", task="transcribe", cache_dir=str(ROOT / "hf_cache")
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model,
        dtype=training_dtype,
        cache_dir=str(ROOT / "hf_cache"),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    arguments = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=min(10, max(1, args.max_steps // 10)),
        fp16=training_dtype == torch.float16,
        bf16=training_dtype == torch.bfloat16,
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
        load_best_model_at_end=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=arguments,
        train_dataset=OffsetAudioDataset(train_records, processor),
        eval_dataset=OffsetAudioDataset(validation_records, processor),
        data_collator=WhisperCollator(processor),
        processing_class=processor.feature_extractor,
    )
    result = trainer.train()
    trainer.save_model(str(args.output_dir / "final_adapter"))
    processor.save_pretrained(str(args.output_dir / "final_adapter"))
    metrics = dict(result.metrics)
    metrics.update(
        {
            "base_model": args.model,
            "train_segments": len(train_records),
            "validation_segments": len(validation_records),
            "max_steps": args.max_steps,
            "precision": str(training_dtype).replace("torch.", ""),
            "gpu": torch.cuda.get_device_name(0),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
