# Changelog

## Unreleased

### Current architecture

- Generalized the application from software-requirements extraction to
  speaker-aware meeting intelligence.
- Upgraded runtime defaults to Whisper Medium, pyannote Community-1, and Qwen
  2.5 1.5B Instruct.
- Added one-shot subprocess workers so each heavyweight model releases its RAM
  and VRAM before the next stage starts.
- Added bounded transcript processing, grounded meeting Q&A, validated JSON
  normalization, and failure fallbacks that preserve completed transcripts.
- Added AMI and QMSum dataset builders, evaluation scripts, experiment records,
  and a promoted QMSum adapter for the matching Q&A task.
- Reworked the Streamlit interface into a full-width, high-contrast workspace.
- Added segment-validated evidence links, timestamped audio playback, inline
  review/edit/confirmation controls, and transcript highlighting.
- Added versioned session caching for transcription, diarization, Qwen evidence windows,
  retrieval indexes, and repeated answers.
- Replaced fixed keyword-only Q&A context selection with hybrid lexical and
  MiniLM semantic retrieval, neighbouring turns, and an evidence budget.
- Added separate model-load, inference, and post-processing timings plus opt-in
  Faster-Whisper GPU quantization and batching settings.
- Removed Qwen's expensive second-pass JSON repair, enabled SDPA/inference-mode
  generation, reduced redundant chunk generations, and added generation/worker
  timeouts with accurate per-stage progress text.
- Added Whisper coverage diagnostics and a hybrid VAD/no-VAD strategy that
  filters unreliable retry segments and uses only reliable gap-filling speech.
- Replaced slow full-transcript generations and recency-prone model
  reconciliation with three cached, timeline-balanced evidence windows and a
  deterministic merge.
- Grounded topics, key points, risks, and open questions in source segment IDs
  in addition to decisions and actions; unsupported claims are omitted.
- Added narrow full-transcript recovery of explicit commitments, requests, and
  recommendations, with confirmed/suggested status and grounded owner/deadline
  fields.
- Replaced free-form model summaries with summaries assembled only from claims
  that passed evidence validation.
- Added a low-transcript-coverage review gate, diarization-based participant
  fallback, and honest combined pipeline-processing timing.
- Raised Qwen's structured-output allowance from 320 to 512 tokens after the
  locked regression showed that short generations omitted later schema fields.
- Tightened action validation to reject negated or past work, conditional
  examples, interface-label quotations, petitions, and immediate meeting
  operations while retaining explicit commitments and assignments.
- Added deterministic, evidence-linked recovery for introductions, cross-turn
  role assignments, selling price, production cost, profit and sales targets,
  international-market scope, age range, and product qualities.
- Added multi-part Q&A completion that copies omitted quantities and explicit
  role owners from the already-retrieved evidence.
- Fixed transcript-only speaker counts, worker timing propagation, and noisy
  Streamlit model-package file watching.

### Cleanup

- Removed the retired TrOCR, multimodal fusion, software-routing, and
  requirements-reporting implementation.
- Removed its obsolete prompt, evaluation corpus, compatibility wrappers, and
  tests while retaining focused coverage of every active pipeline stage.
- Renamed active Python modules to standard snake_case filenames.
- Removed obsolete model caches, duplicate dataset archives, browser profiles,
  stale bytecode, and non-promoted checkpoint weights.
- Added reproducible Ruff and Vulture clean-code checks.
- Consolidated development and inference into the Python 3.11 environment.
- Removed persistent report writes; reports now remain in the Streamlit session
  and are exported only when the user requests a download.
- Reduced the AMI preparation path to the audio, transcript, summary, action,
  decision, and diarization data actually used by the current application.
- Removed duplicate QMSum representations while retaining the official raw
  train/validation/test meetings and generated benchmark records.
