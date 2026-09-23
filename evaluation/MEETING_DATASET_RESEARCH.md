# Meeting-recording dataset research and benchmark recommendation

Research checked on **9 September 2026** for this application's three-stage pipeline:

1. Whisper Medium: automatic speech recognition (ASR)
2. pyannote Community-1: speaker diarization
3. Qwen 2.5 1.5B Instruct: summary, topics, decisions, actions, risks, open questions, and meeting Q&A

## Executive conclusion

There is no credible, freely downloadable English benchmark I found that simultaneously provides all of the following at useful quality: real general-meeting audio, human verbatim transcripts, speaker-time ground truth, and human labels for summaries, decisions, action items, risks, and open questions.

The strongest defensible benchmark is therefore a **three-track composite**:

1. **NOTSOFAR-1 eval-small** for Whisper and pyannote: real office meetings, short recordings, human transcription/speaker ground truth, and an established speaker-attributed ASR metric.[^1]
2. **ICSI + QMSum joined by meeting ID** for the closest thing to an English end-to-end benchmark: real research meetings with audio and human transcription from ICSI, plus human-written query summaries from QMSum.[^3][^5]
3. **MeetingBank held-out segments** for Qwen and realistic long-form meeting intelligence: city-council speech paired with official/professional minutes, but not reliable human ASR/diarization ground truth.[^6]

The QMSum test meetings Bed008, Bmr014, Bro019, covid_4, ES2004a, ES2011a,
Bed016 and ES2002a were used during diagnosis or regression work and must not be
presented as an untouched final benchmark. A final headline result should lock
different meetings before running or tuning the pipeline.

## Ranked dataset shortlist

| Rank | Dataset | Real audio | Human transcript / speakers | Human summary or intelligence labels | Download burden | Best use here |
|---|---|---:|---:|---:|---:|---|
| 1 | ICSI + QMSum | Yes | Yes | Human query summaries and QA-style targets | About 120 MB per ICSI mixed meeting | Closest end-to-end English benchmark |
| 2 | NOTSOFAR-1 | Yes | Yes | No summary/action gold | About 1 GB if only one single-channel device for all 80 eval-small meetings; 20 GB for every device | Primary ASR + diarization benchmark |
| 3 | MeetingBank | Yes | Speaker-tagged transcripts, but not trustworthy gold for WER/DER | Professional minutes and 6,892 segment summaries | 198 GB for the complete audio repository; much smaller if clips are cut from source URLs | Primary general-meeting summarization benchmark |
| 4 | GAP Corpus | Yes | Transcripts and interaction annotations | No app-shaped summary/action gold | 1.9 GB audio | Small, easy speech/diarization pilot |
| 5 | CHiME-6 | Yes | Full transcripts, speaker labels, times | No meeting-summary gold | Large and licence application required | Very hard noisy/overlap stress test |
| 6 | SUMM-RE | Yes, French | Manual dev/test transcripts; individual speaker tracks | Public HF release does not include the full project's meeting summaries | 93.8 GB complete | Optional multilingual test |
| 7 | AISHELL-4 / AliMeeting | Yes, Mandarin | Yes | No English meeting-intelligence labels | Tens of GB | Optional multilingual speech benchmark |

## 1. NOTSOFAR-1: use this for the speech half of the application

This is the best match for how the app actually receives audio: a single distant microphone in a real meeting room. The current open release contains 237 meetings, averaging six minutes, recorded in 30 conference rooms with four to eight attendees and 35 unique speakers. The public `240629.1_eval_small_with_GT` subset contains 80 meetings and ground truth. It is explicitly designed for resource-constrained evaluation and is identical to the original challenge evaluation set.[^1]

It supports:

- WER for Whisper
- DER/JER and speaker-count error for pyannote
- speaker-attributed WER (`tcpWER`) for the combined Whisper + pyannote result
- latency, real-time factor, peak RAM, and peak VRAM for the application

It does **not** provide human summaries, decisions, or action items, so it should not be used to score Qwen.

The complete eval-small directory is about 20 GB because it includes close-talk, multiple single-channel devices, and multichannel arrays.[^2] The app only needs one distant single-channel recording per meeting. Selective Hugging Face download patterns can reduce the useful benchmark to roughly one device plus JSON ground truth instead of downloading every microphone:

```python
from huggingface_hub import snapshot_download

root = "benchmark-datasets/eval_set/240629.1_eval_small_with_GT"
snapshot_download(
    repo_id="microsoft/NOTSOFAR",
    repo_type="dataset",
    local_dir="datasets/notsofar_eval_small",
    allow_patterns=[
        f"{root}/MTG/*/sc_meetup_0/ch0.wav",
        f"{root}/MTG/*/gt_transcription.json",
        f"{root}/MTG/*/gt_meeting_metadata.json",
    ],
)
```

For a tiny first test, download only meeting `MTG_32000`. Its single-channel WAV is about 11.6 MB and the ground-truth transcript is about 213 KB:

```python
from huggingface_hub import snapshot_download

meeting = "benchmark-datasets/eval_set/240629.1_eval_small_with_GT/MTG/MTG_32000"
snapshot_download(
    repo_id="microsoft/NOTSOFAR",
    repo_type="dataset",
    local_dir="datasets/notsofar_smoke",
    allow_patterns=[
        f"{meeting}/sc_meetup_0/ch0.wav",
        f"{meeting}/gt_transcription.json",
        f"{meeting}/gt_meeting_metadata.json",
    ],
)
```

The data is CC BY 4.0. The official repository notes that its full baseline environment is Linux-oriented, but the WAV and JSON files themselves can be processed by this Windows application without installing the NOTSOFAR baseline.[^1]

## 2. ICSI + QMSum: use this for the closest end-to-end test

ICSI contains about 70 hours of real, naturally occurring English research-group meetings. Unlike the disliked AMI scenario meetings, these were recurring meetings that would have taken place regardless of recording. ICSI provides a single mixed WAV, individual close-talk tracks, orthographic transcription, and dialogue-act annotations; audio and core annotations are CC BY 4.0.[^3][^4]

QMSum contains 1,808 human-annotated query-summary pairs over 232 meetings and includes all 59 ICSI meetings as its academic domain.[^5] Meeting IDs match directly between the two resources.

The local QMSum archive's official **academic test split** contains exactly these nine ICSI meetings:

```text
Bed003, Bed008, Bed016,
Bmr006, Bmr014, Bmr023,
Bro004, Bro019, Bro027
```

These nine should be the first full end-to-end benchmark because they were not used for QMSum training. Download the ICSI **Headset mix** WAV for each one and the core-plus-contributed annotation ZIP from the official ICSI page. The site reports an average of about 120 MB per mixed meeting, so all nine should be around 1.1 GB rather than a corpus-scale download.[^4]

The audio files also have stable direct URLs. For example:

```powershell
New-Item -ItemType Directory -Force datasets\icsi_qmsum_test | Out-Null
$ids = 'Bed003','Bed008','Bed016','Bmr006','Bmr014','Bmr023','Bro004','Bro019','Bro027'
foreach ($id in $ids) {
    Invoke-WebRequest `
        "https://groups.inf.ed.ac.uk/ami/ICSIsignals/NXT/$id.interaction.wav" `
        -OutFile "datasets\icsi_qmsum_test\$id.wav"
}
```

Use the joined records as follows:

- ICSI audio versus ICSI orthographic transcript: Whisper WER
- ICSI speaker-time annotations versus pyannote output: DER/JER
- ICSI gold speaker transcript versus the app's speaker transcript: speaker-attributed WER
- QMSum general query answer versus Qwen summary: ROUGE-L, BERTScore, and human factuality scoring
- QMSum specific queries versus “Ask the meeting”: token F1 / ROUGE-L plus answer-grounding checks

ICSI also has a published extension with actionable-item labels over 22 meetings and 21,000 speaker turns, with 10 actionable intent types.[^15] It is relevant to the app, but the current Microsoft publication page does not expose a simple standalone download. Treat it as an optional annotation lead, not a dependency for the first benchmark.

## 3. MeetingBank: use what is already in the workspace, but use it correctly

MeetingBank contains 1,366 city-council meetings, more than 3,579 hours of video, professional minutes, agendas, and 6,892 segment-level summarization examples. The mean full meeting is 2.6 hours and over 28,000 transcript tokens.[^6]

The workspace already contains:

- the 253 MB master metadata file with 1,250 locally packaged meeting records;
- official train/validation/test JSONL splits;
- timestamped segment transcripts and reference summaries;
- QMSum separately;
- **zero MeetingBank audio files**.

The complete MeetingBank audio repository is approximately 198 GB and CC BY-NC-SA 4.0.[^7] Downloading all of it is unnecessary and poorly matched to the machine's RAM and runtime limits.

Instead, construct a held-out clip set from the local `Metadata/Splits/test.json` and the matching segment `startTime`, `endTime`, and `URLs.Video` in `Metadata/MeetingBank.json`. Cut only 3–12 minute agenda items from the original video URL:

```powershell
ffmpeg -ss 147 -i "http://archive-media.granicus.com:443/OnDemand/longbeach/example.mp4" `
    -t 242 -vn -ac 1 -ar 16000 meetingbank_segment.wav
```

Recommended sample design:

- 30 held-out segments total
- five segments from each of the six cities
- 3–12 minutes per segment
- a mixture of agenda items, contracts, resolutions, public hearings, and reports
- include both short/no-action cases and segments with explicit votes or commitments

MeetingBank's transcript text visibly contains ASR errors, and its numeric speaker IDs are not a verified diarization reference. Therefore:

- use it for Qwen summary/content evaluation and realistic end-to-end demonstrations;
- do not report its transcript as human gold WER;
- do not report its speaker IDs as gold DER without manual verification;
- manually annotate decisions/actions for a smaller 15–20 segment subset if those fields are central to the report.

## 4. Smaller or specialist alternatives

### GAP Corpus

The GAP Corpus provides 28 English small-group meetings, transcripts, annotations, task data, and a separate 1.9 GB WAV download under CC BY-NC 4.0.[^8] It is easy to obtain and manageable on this computer. It is suitable for a smoke/pilot evaluation of ASR and diarization, but it uses a winter-survival ranking exercise, so it has the same scenario/task limitation that made AMI unattractive.

### CHiME-6

CHiME-6 contains 20 real English dinner parties recorded in homes, with four participants per session, multiple distant arrays, full transcriptions, speaker labels, and utterance times. The official split is about 40.5 hours train, 4.5 hours development, and 5.2 hours evaluation.[^9] It is excellent for difficult overlap, movement, and background-noise testing. It is not a formal meeting-intelligence dataset, and access requires a licence application; the free non-commercial licence is intended for not-for-profit organizations and requires a separate local approver.[^9]

### SUMM-RE

SUMM-RE is a strong French meeting-style corpus: 210 train, 36 development, and 37 test conversations, each about 20 minutes with usually three or four speakers. Development and test have manually aligned transcripts, and individual speaker audio tracks allow a mixed diarization test. It is CC BY-SA 4.0, but the current Hugging Face release is 93.8 GB and explicitly says it is only an extract of the full corpus; the public schema does not include the full project's meeting-summary labels.[^10]

### AISHELL-4 and AliMeeting

AISHELL-4 supplies 120 hours of real Mandarin meetings with four to eight speakers, an eight-channel array, transcription, and speaker activity under CC BY-SA 4.0.[^11] AliMeeting similarly offers 118.75 hours of real Mandarin meetings with far-field and headset channels.[^12] Both are valuable only if the final report includes a multilingual robustness section; neither should replace the English primary benchmark.

### Four-meeting Hugging Face conversational sample

`jml2026/conversational-speech-dataset` is a convenient 135.9 MB English sample with four meetings, mixed and per-speaker WAVs, speaker IDs, and word timestamps under CC BY-NC 4.0.[^13] It can prove the import/evaluation code works before larger downloads, but four meetings are too few for a credible final performance claim.

### Paid exact-match option

Defined.ai advertises 99.5K hours of English corporate meetings spanning interviews, one-to-ones, standups, and multiple industries, with audio/video, transcripts, metadata, action items, and recaps.[^14] This is the closest commercial match to the app's complete schema, but access is quote/sample based. It is not recommended for a student project unless the university already has access.

## Datasets to avoid as the primary benchmark

- **NSF-QA:** derived from NOTSOFAR, but its questions, answers, and per-speaker summaries were generated by Gemini rather than humans. It is useful as weak-label supplementary data, not ground truth.
- **MeetingBank full audio download:** 198 GB and mostly very long meetings; use timestamped clips instead.
- **Purely synthetic meeting-summary or action-item corpora:** useful for unit tests and prompt regression, but they cannot support a claim about performance on real meetings.
- **Audio-only meeting sets without transcripts/timestamps:** they can support demos or listening-based review, but not repeatable WER/DER benchmarks.
- **AMI as the sole benchmark:** it is richly annotated, but the scenario meetings are narrow and the user has already found them unrepresentative. Keeping a few AMI examples only as a historical comparison is reasonable.

## Recommended final benchmark protocol

### Speech benchmark

- NOTSOFAR eval-small: all 80 meetings using one single-channel device, or a fixed stratified subset of 20 if runtime is prohibitive.
- Report WER, DER, JER, speaker-count mean absolute error, and speaker-attributed WER.
- Report wall-clock time, real-time factor (`processing seconds / audio seconds`), peak RAM, and peak VRAM.

### End-to-end benchmark

- Nine ICSI meetings in the QMSum academic test split.
- Score ASR/diarization against ICSI and summary/Q&A against QMSum.
- Run two Qwen conditions: gold ICSI transcript and app-generated transcript. Their difference measures how much speech errors damage meeting intelligence.

### General-meeting intelligence benchmark

- Thirty MeetingBank test segments, stratified across cities and item types.
- Automatic metrics: ROUGE-L and BERTScore for summary; JSON-schema success rate; evidence/quote support rate.
- Human rubric: factuality, coverage, unsupported claims, useful decisions, correct action owner, correct due date, and appropriate “none found” behavior.
- Manually double-annotate 15–20 clips for decisions/actions/open questions. This small gold layer is more defensible than treating machine-produced labels from another model as truth.

### Contamination controls

- Never tune prompts or adapters on NOTSOFAR eval-small, QMSum test, or MeetingBank test.
- Choose and save the meeting/segment manifest before looking at results.
- Record model IDs, adapter IDs, decoding settings, hardware, and code commit for every run.
- Compare the same fixed manifest for all model/configuration versions.

## What to do next

1. Download the single `MTG_32000` NOTSOFAR smoke meeting and make the evaluator compute WER and DER from its JSON ground truth.
2. Download `Bmr006.interaction.wav` and join it to the existing QMSum `Bmr006` test record; validate one complete audio-to-summary run.
3. Once both adapters work, download the remaining eight ICSI/QMSum test meetings and the single-channel NOTSOFAR subset.
4. Build the 30-clip MeetingBank manifest without downloading the 198 GB audio repository.
5. Freeze the manifests and run the final benchmark matrix.

## Sources

[^1]: Microsoft, [NOTSOFAR-1 Challenge repository and current recorded-meeting release](https://github.com/microsoft/NOTSOFAR1-Challenge).
[^2]: Microsoft, [NOTSOFAR eval-small-with-ground-truth files](https://huggingface.co/datasets/microsoft/NOTSOFAR/tree/main/benchmark-datasets/eval_set/240629.1_eval_small_with_GT).
[^3]: University of Edinburgh/ICSI, [ICSI Meeting Corpus overview](https://groups.inf.ed.ac.uk/ami/icsi/).
[^4]: University of Edinburgh/ICSI, [ICSI audio and annotation download page](https://groups.inf.ed.ac.uk/ami/icsi/download/).
[^5]: Yale-LILY, [QMSum official repository and data schema](https://github.com/Yale-LILY/QMSum).
[^6]: MeetingBank authors, [MeetingBank official dataset site](https://meetingbank.github.io/).
[^7]: MeetingBank authors, [MeetingBank Audio repository](https://huggingface.co/datasets/huuuyeah/MeetingBank_Audio).
[^8]: University of the Fraser Valley, [GAP Corpus official site](https://sites.google.com/view/gap-corpus/home).
[^9]: CHiME organizers, [CHiME-6 dataset description](https://chimechallenge.github.io/chime6/track1_data.html) and [licensing page](https://chimechallenge.github.io/chime6/download.html).
[^10]: LINAGORA and Aix-Marseille University, [SUMM-RE dataset card](https://huggingface.co/datasets/linagora/SUMM-RE).
[^11]: OpenSLR, [AISHELL-4](https://www.openslr.org/111/).
[^12]: OpenSLR, [AliMeeting](https://www.openslr.org/119/).
[^13]: Hugging Face, [conversational speech sample dataset](https://huggingface.co/datasets/jml2026/conversational-speech-dataset).
[^14]: Defined.ai, [commercial Meeting Recordings dataset](https://defined.ai/datasets/meeting-recordings).
[^15]: Microsoft Research, [AIMU: Actionable Items for Meeting Understanding](https://www.microsoft.com/en-us/research/publication/aimu/).
