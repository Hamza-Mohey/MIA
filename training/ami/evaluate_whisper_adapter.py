"""Compare Whisper base and LoRA word error rate on held-out AMI segments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from jiwer import wer
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from training.ami.train_whisper_lora import ROOT, OffsetAudioDataset, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "training" / "ami" / "generated" / "asr_segments.jsonl",
    )
    parser.add_argument("--model", default="openai/whisper-medium")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=ROOT / "training" / "ami" / "runs" / "whisper_medium_lora_20" / "final_adapter",
    )
    parser.add_argument("--eval-limit", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "training"
        / "ami"
        / "runs"
        / "whisper_medium_lora_20"
        / "wer_comparison.json",
    )
    return parser.parse_args()


def transcribe_records(model, dataset, processor) -> list[str]:
    predictions: list[str] = []
    model.eval()
    for index in range(len(dataset)):
        features = dataset[index]["input_features"].unsqueeze(0).to("cuda", dtype=torch.float16)
        with torch.inference_mode():
            generated = model.generate(input_features=features, max_new_tokens=128)
        predictions.append(
            processor.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        )
    return predictions


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this configured evaluation.")
    if not args.adapter.exists():
        raise SystemExit(f"Adapter not found: {args.adapter}")

    records = [record for record in read_jsonl(args.manifest) if record["split"] == "validation"][
        : args.eval_limit
    ]
    processor = WhisperProcessor.from_pretrained(
        args.model,
        language="English",
        task="transcribe",
        cache_dir=str(ROOT / "hf_cache"),
    )
    dataset = OffsetAudioDataset(records, processor)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.float16,
        cache_dir=str(ROOT / "hf_cache"),
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    references = [record["text"].strip() for record in records]
    base_predictions = transcribe_records(model, dataset, processor)
    adapted_model = PeftModel.from_pretrained(model, str(args.adapter))
    adapted_predictions = transcribe_records(adapted_model, dataset, processor)
    base_wer = float(wer(references, base_predictions))
    adapted_wer = float(wer(references, adapted_predictions))
    result = {
        "base_model": args.model,
        "adapter": str(args.adapter),
        "validation_segments": len(records),
        "base_wer": base_wer,
        "adapted_wer": adapted_wer,
        "absolute_wer_change": adapted_wer - base_wer,
        "relative_wer_change_percent": ((adapted_wer - base_wer) / base_wer) * 100
        if base_wer
        else None,
        "adapter_improved_wer": adapted_wer < base_wer,
        "samples": [
            {
                "reference": reference,
                "base_prediction": base_prediction,
                "adapted_prediction": adapted_prediction,
            }
            for reference, base_prediction, adapted_prediction in zip(
                references, base_predictions, adapted_predictions, strict=True
            )
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))


if __name__ == "__main__":
    main()
