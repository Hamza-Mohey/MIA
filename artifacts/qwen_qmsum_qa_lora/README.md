# Qwen QMSum meeting-Q&A LoRA

This is the promoted parameter-efficient adapter used by the application's
**Ask the meeting** feature. It is not used for structured brief extraction.

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Training data: 1,095 QMSum training queries with relevant transcript spans
- Method: LoRA, rank 16, alpha 32, one epoch (137 optimizer steps)
- Hardware: NVIDIA GeForce RTX 3060 Ti
- Trainable parameters: 4,358,144 (0.2815% of the base model)

On 16 QMSum validation queries, the adapter changed mean token F1 from 0.1788
to 0.2070 and mean ROUGE-L F1 from 0.1236 to 0.1405. It also passed the local
grounding regression used for promotion. This small validation experiment is
not a general accuracy claim.

Only `adapter_config.json` and `adapter_model.safetensors` are required. The
base model and tokenizer are downloaded separately into the ignored local
Hugging Face cache.

See `evaluation/results/model_experiments.json` and
`training/EXPERIMENTS.md` for the recorded comparison and limitations.
