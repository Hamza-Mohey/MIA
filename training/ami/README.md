# AMI corpus preparation and evaluation

This package converts the local AMI subset into reproducible manifests for
speech recognition, diarization, and meeting-intelligence evaluation.

```powershell
.\.venv-training\Scripts\python.exe -m training.ami.build_dataset
```

## Local subset

The corpus contains 30 `ES` remote-control scenario meetings: the `a` and `b`
meetings from 15 projects. Together they provide 13.21 hours of 16 kHz mono
mixed-headset audio. AMI annotations provide timed words, speaker turns, formal
roles, abstractive summaries, actions, decisions, and problems.

| Split | Meetings | Projects | Intended use |
|---|---:|---:|---|
| train | 22 | 11 | Training and prompt development |
| validation | 4 | 2 | Model and threshold selection |
| test | 4 | 2 | Locked comparison |

The split follows AMI's `seen_type` metadata and keeps both meetings from a
project together to reduce scenario leakage.

## Generated manifests

- `meeting_records.jsonl`: transcripts, roles, gold outputs, and modality
  metadata for each meeting.
- `asr_segments.jsonl`: manually transcribed, forced-aligned audio ranges of at
  most 28 seconds for Whisper evaluation and training.
- `diarization_manifest.jsonl` and `diarization_reference.rttm`: speaker-time
  references for pyannote evaluation.
- `text_sft.jsonl`: transcript-to-JSON examples for cautious adapter tests.
- `corpus_audit.json`: counts, coverage, warnings, and split checks.

These files are excluded from Git because they are generated from the local
licensed corpus.

## Current findings

Long meetings are analysed through bounded, timeline-balanced evidence
windows. The 20-step Whisper Medium LoRA pilot did not improve validation WER
and was not promoted. This small, product-design-heavy subset is suitable for
evaluation and parameter-efficient experiments, not general meeting-type
training or full-model fine-tuning.

The local RTX 3060 Ti has 8 GB VRAM. Training therefore uses mixed precision,
gradient checkpointing, batch size one, and LoRA. Recorded experiment results
are in `evaluation/results/model_experiments.json` and
`training/EXPERIMENTS.md`.

## Provenance

- AMI corpus: <https://groups.inf.ed.ac.uk/ami/corpus/>
- AMI downloads: <https://groups.inf.ed.ac.uk/ami/download/>
- Meeting structure: <https://groups.inf.ed.ac.uk/ami/corpus/meetingids.shtml>
- Annotation overview: <https://groups.inf.ed.ac.uk/ami/corpus/annotation.shtml>
- Licence: Creative Commons Attribution 4.0
