# Local AI Meeting Intelligence Workspace

A local-first Streamlit application that converts meeting recordings into a
speaker-attributed transcript and a reviewable brief containing topics, key
points, decisions, action items, risks, and open questions.

## Three-model architecture

1. **Whisper Medium** (`openai/whisper-medium`) transcribes multilingual audio
   and returns timestamps. Runtime inference uses its optimized CTranslate2
   representation through Faster-Whisper in FP16 on CUDA.
2. **pyannote Community-1** (`pyannote/speaker-diarization-community-1`)
   determines who spoke when. Its exclusive diarization timeline is reconciled
   with Whisper timestamps. Users can rename anonymous speaker labels.
3. **Qwen 2.5 1.5B Instruct** (`Qwen/Qwen2.5-1.5B-Instruct`) extracts supported
   general meeting intelligence into validated JSON and answers grounded
   follow-up questions from the transcript.

A small `all-MiniLM-L6-v2` Sentence Transformer supplies semantic retrieval
for grounded Q&A. It complements the required three-stage meeting pipeline;
it does not replace Whisper, pyannote, or Qwen.

Each GPU stage runs in a separate one-shot Python worker. When Whisper,
pyannote, or Qwen finishes, its worker process exits and Windows reclaims its
complete CPU/GPU context. This is more reliable than keeping all models inside
Streamlit and calling `torch.cuda.empty_cache()`. Long transcripts are
processed through bounded evidence windows instead of being silently truncated.

## Application setup

The project uses one Python 3.11 environment for the full application and
training/evaluation utilities. Create it as described in
[training/README.md](training/README.md), then install the runtime packages:

```powershell
.\.venv-training\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-training\Scripts\python.exe -m modules.prefetch_faster_whisper
.\.venv-training\Scripts\python.exe -m modules.prefetch_retrieval_model
.\.venv-training\Scripts\python.exe -m streamlit run app.py
```

The prefetch commands download the CTranslate2 Whisper weights and the small
semantic-retrieval model once. Normal application workers use the project cache
in offline mode and do not download model files during a meeting.

## Evidence review and caching

Every extracted topic, key point, decision, action, risk, and open question is
linked to validated source-segment IDs.
The review UI can highlight those transcript turns, seek uploaded audio to the
supporting timestamp, edit the item without inference, and mark it confirmed.
Model-provided IDs must exist, form a compact local evidence window, contain an
appropriate decision/commitment cue, and pass a claim-to-evidence support
threshold. Unsupported analytical claims are omitted and recorded as
uncertainties. Decision/action/risk/question claims must also contain the
appropriate linguistic cue. The displayed quote always comes from the stored transcript.
When a reviewer opens or plays an item's evidence and then confirms it, the
result records `verification_seconds`. This supports a later within-subject
comparison of verification time with and without evidence navigation.

Transcription, diarization, evidence windows, retrieval indexes, and repeated
answers are cached only in the current Streamlit session. Cache keys include
the input plus relevant model, prompt, adapter, pipeline, and configuration
versions. Correcting a long transcript therefore sends only changed evidence
windows to Qwen. Use **Clear session cache** in System details to delete retained session
data. Nothing is persisted across sessions.

Qwen extraction uses GPU SDPA attention, inference mode, and a 512-token bounded
response. Before inference, the application removes repeated/noisy turns and
selects a 9,000-character, three-window digest that balances explicit outcome
cues, lexical centrality, and chronological coverage. This avoids feeding the
same 40-minute transcript through Qwen several times while reducing end-of-
transcript bias. Candidates from the three windows are merged deterministically;
the application then rescans every original source segment for narrow explicit
follow-ups and validates every retained claim against the complete transcript.
Truncated JSON is repaired
deterministically instead of launching a second Qwen generation. Individual
generations are capped at 65 seconds and the complete Qwen worker at 360
seconds, so a failed generation cannot silently occupy the GPU for half an
hour. These limits can be changed with `MAX_NEW_TOKENS`,
`ANALYSIS_EVIDENCE_MAX_CHARACTERS`, `TEXT_GENERATION_TIMEOUT_SECONDS`, and
`TEXT_WORKER_TIMEOUT_SECONDS`.

`WHISPER_COMPUTE_TYPE=int8_float16` enables GPU quantization for local
benchmarking. `WHISPER_BATCH_SIZE` enables Faster-Whisper batching when set
above 1. Both are opt-in because batching trades additional VRAM for speed.
Whisper defaults to `WHISPER_VAD_MODE=auto`. Long recordings with suspiciously
low words per minute are retried once without VAD. Rather than accepting the
entire noisier retry, the application filters low-confidence, silence-like,
compressed, and repeated segments, then adds only reliable turns that fill VAD
gaps. `WHISPER_VAD_MODE=on` or `off` forces a single strategy. The review screen
warns for questionable coverage and requires explicit acknowledgement only for
critically sparse transcripts.

Adapters can be enabled explicitly after they pass both held-out metrics and
task-level generation checks:

```powershell
$env:TEXT_ADAPTER_PATH="path\to\validated\qwen_adapter"
.\.venv-training\Scripts\python.exe -m streamlit run app.py
```

The one-epoch Qwen QMSum adapter passed the generation gate and is discovered
automatically at `artifacts/qwen_qmsum_qa_lora`. It is
enabled only inside **Ask the meeting**; structured brief extraction explicitly
disables it and keeps the base model. Do not enable either 20-step smoke
adapter. Whisper's held-out decoded WER was unchanged from the base model.

## Required pyannote access

Community-1 is free and CC BY 4.0, but its Hugging Face repository is gated:

1. Accept the conditions at
   <https://huggingface.co/pyannote/speaker-diarization-community-1>.
2. Create a read token at <https://huggingface.co/settings/tokens>.
3. Log in without placing the token in source code:

```powershell
.\.venv-training\Scripts\hf.exe auth login
```

An `HF_TOKEN` authenticates requests and raises rate limits; it does not remove
ISP, proxy, server, or physical bandwidth limits.

## Datasets and evaluation

- `training/ami`: 30 local AMI recordings, 13.21 hours, 8,979 ASR/diarization
  reference segments, an official project-grouped split, and gold summary/
  action/decision/problem annotations.
- `training/qmsum`: 232 general meetings across academic, product, and committee
  domains; 1,576 specific query examples plus 232 general summaries. The
  official QMSum split is kept separate from the local AMI split.
- `training/ami/run_diarization_baseline.py`: reports diarization error rate on
  held-out AMI meetings.
- `training/ami/train_whisper_lora.py`: 8 GB-friendly Whisper LoRA training.
- `training/qmsum/train_qwen_lora.py`: Qwen LoRA smoke training using supported
  transcript spans.
- `training/qmsum/evaluate_adapter.py`: compares base and adapted Qwen loss on
  identical held-out examples.
- `training/ami/evaluate_whisper_adapter.py`: compares base and adapted Whisper
  WER on identical held-out audio segments.

Rebuild and test:

```powershell
.\.venv-training\Scripts\python.exe -m training.ami.build_dataset
.\.venv-training\Scripts\python.exe -m training.qmsum.build_dataset
.\.venv-training\Scripts\python.exe -m unittest discover -s tests -v
```

The generated manifests, downloaded datasets, model cache, and Python
environment are intentionally excluded from Git. The promoted Qwen LoRA is the
only model weight stored in the repository because it is a project-produced,
runtime-used academic artifact. Dataset and model provenance is recorded in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Run the static clean-code checks:

```powershell
.\.venv-training\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv-training\Scripts\python.exe -m ruff check app.py modules evaluation training tests
.\.venv-training\Scripts\python.exe -m vulture app.py modules evaluation training tests --min-confidence 80
```

## Safety and limitations

- Speaker labels are anonymous unless the user names them.
- Overlapping speakers, noise, accents, and remote-call compression can reduce
  transcription and diarization accuracy.
- Decisions, owners, deadlines, and names must be reviewed before distribution.
- The local AMI subset is software-product-heavy and is not used alone to train
  a general meeting classifier.
- Keep attribution for AMI/pyannote CC BY 4.0 material and QMSum's MIT licence.
