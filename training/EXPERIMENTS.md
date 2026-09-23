# Experiment record

Measured locally on an NVIDIA GeForce RTX 3060 Ti. These are smoke-run results,
not final test-set claims. Test partitions were not used for training or model
selection.

## Initial AMI Qwen baseline

Three validation meetings completed in 54.865-83.437 seconds each. A fourth
meeting saturated the 8 GB GPU and was stopped after more than ten minutes.
The completed outputs averaged 0.139 lexical summary similarity and produced no
matches at the strict 0.82 fuzzy threshold for decisions, actions, or problems.
Manual review also found schema confusion and unsupported owners. This baseline
motivated bounded evidence windows, stricter prompts, and evidence validation;
the generated prediction files are no longer retained.

## 2026-07-29: Qwen QMSum LoRA

- Base: `Qwen/Qwen2.5-1.5B-Instruct`
- Trainable parameters: 4,358,144 / 1,548,072,448 (0.2815%)
- Training: 20 optimizer steps, effective batch 8, 160 observed examples
- Held-out subset: first 64 official QMSum validation query examples
- Base loss: 2.763656
- Adapted loss: 2.538304
- Relative change: -8.154% (lower is better)
- Qualitative gate: for a transcript containing a Friday 10 AM customer review
  and a Wednesday agenda deadline, the adapter incorrectly answered Wednesday
  when asked for the review time. The unchanged base correctly answered Friday
  at 10 AM.
- Decision: do not promote. Lower teacher-forced loss alone did not translate
  to better grounded generation.

The compact result is retained in
`evaluation/results/model_experiments.json`; generated run files are excluded.

## 2026-07-29: Qwen QMSum LoRA, one epoch

- Training: 137 optimizer steps, effective batch 8, one pass over 1,095 examples
- Held-out loss: 2.433 versus base 2.764
- Generation subset: first 16 official QMSum validation query examples
- Mean token F1: 0.178800 base, 0.206950 adapted (+15.7%)
- Mean ROUGE-L F1: 0.123612 base, 0.140453 adapted (+13.6%)
- Grounding regression: passed; both expected event-day and time were present
- Decision: promoted for the matching meeting-Q&A feature only. Structured JSON
  brief extraction runs with the adapter disabled.

The promoted adapter is retained at `artifacts/qwen_qmsum_qa_lora`; compact
evaluation results are in `evaluation/results/model_experiments.json`.

## 2026-07-29: Whisper Medium AMI LoRA

- Base: `openai/whisper-medium`
- Trainable parameters: 4,718,592 / 768,576,512 (0.6139%)
- Training: 20 optimizer steps, effective batch 8, 160 observed segments
- Held-out subset: first 32 official local AMI validation segments
- Base WER: 0.344418
- Adapted WER: 0.344418
- Relative change: 0.0%
- Decision: do not promote. The next-run defaults now use BF16 on supported
  GPUs and lower the learning rate from 1e-4 to 2e-5 to address the unstable
  FP16 gradient norms before evaluating a meaningfully longer checkpoint.

The compact result is retained in
`evaluation/results/model_experiments.json`; the unpromoted weights were removed.

The revised BF16/2e-5 configuration passed a one-step runtime smoke check with
finite loss (1.166), finite gradient norm (0.811), and finite two-example
validation loss (2.853). This validates the safer configuration but is not an
accuracy result and is not promoted.

Intermediate checkpoints, TensorBoard event files, and adapter weights from
non-promoted runs were removed after their metrics were recorded. They can be
recreated from the documented commands and generated datasets.

## Diarization

The Community-1 evaluation script and 8,979-turn RTTM reference are prepared,
but downloading the gated model requires the project owner to accept its Hugging
Face conditions and authenticate. No diarization accuracy claim has been made.
