# QMSum for the general-meeting direction

QMSum is used for Qwen summarization, topic location, and meeting question
answering. It contains academic, product, and committee meetings, giving much
broader coverage than the local AMI scenario subset.

Build the manifests with:

```powershell
.\.venv-training\Scripts\python.exe -m training.qmsum.build_dataset
```

The official train/validation/test split is preserved. Do not combine QMSum's
AMI-derived records with `training/ami/text_sft.jsonl`: QMSum assigns some AMI
meetings to different splits, which would contaminate evaluation.

Upstream: <https://github.com/Yale-LILY/QMSum> (MIT licence). The dataset paper
reports 1,808 query-summary pairs across 232 meetings.

Training and promotion checks:

```powershell
.\.venv-training\Scripts\python.exe -m training.qmsum.train_qwen_lora --max-steps 20
.\.venv-training\Scripts\python.exe -m training.qmsum.evaluate_adapter
.\.venv-training\Scripts\python.exe -m training.qmsum.evaluate_generation
```

An adapter is not promoted from validation loss alone. It must also improve
held-out token F1 and ROUGE-L generation and pass the grounded regression
checks in `evaluate_generation.py`.
