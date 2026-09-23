# Manual Application Evaluation — 20 September 2026

## Scope

This evaluation exercised the application as a user and tested the current models on
meetings that were not part of the recent Bed016/ES2002a development loop. It covers:

- the Streamlit sample workflow;
- five official QMSum test transcripts from different domains;
- one complete AMI audio-to-brief run;
- three grounded business-meeting questions;
- automated reliability and static-quality checks.

The machine used an AMD Ryzen 5 5600X, 16 GB RAM and an NVIDIA RTX 3060 Ti with
8 GB VRAM. No additional recordings were downloaded because the repository already
contains 30 AMI recordings and 232 QMSum meetings, including unused official test
meetings. Adding another source would not address the defects found here.

## Test 1: manual Streamlit workflow

The built-in weekly-planning sample contains one decision, one assigned action with a
deadline, and one open question. The application correctly extracted all three:

- decision: move the customer review to Friday at 10:00;
- action: `SPEAKER_01` sends the revised agenda by Wednesday;
- open question: the meeting room remains unconfirmed.

Evidence navigation opened the exact transcript turn. Confirmation changed the review
state and recorded 5.01 seconds between opening the evidence and confirming the item.
The question "When is the customer review?" was answered correctly as Friday at 10:00.

The first analysis took 59.12 seconds: 42.15 seconds loading Qwen and 15.96 seconds
performing inference. Loading the same sample and generating the same brief again
completed visibly in under one second, confirming that session caching works.

Observed UI/runtime defects:

1. The overview displayed the speaker count as `—` although `SPEAKER_00` and
   `SPEAKER_01` were present in the uploaded transcript.
2. System details displayed `text_analysis_seconds: 0` even though the worker correctly
   reported 58.11 seconds.
3. Streamlit's source watcher produced thousands of non-fatal Transformers image-module
   import traces because `torchvision` is not installed. The application continued to
   work, but the terminal output obscures genuine errors.

## Test 2: five held-out QMSum transcripts

The current structured-analysis pipeline was run on official test meetings Bed008,
Bmr014, Bro019, covid_4 and ES2004a. Human transcripts were used so this test isolates
Qwen and deterministic post-processing from Whisper.

| Meeting | Domain | Token F1 | ROUGE-L F1 | Extracted actions | Main observation |
| --- | --- | ---: | ---: | ---: | --- |
| Bed008 | Academic | 7.25% | 5.80% | 0 | Produced one broad topic and missed the detailed belief-net discussion. |
| Bmr014 | Research administration | 3.36% | 3.36% | 12 | Most actions were descriptive, conditional, negated or discussion fragments. |
| Bro019 | Technical research | 12.08% | 9.40% | 3 | Missed the technical results; all three actions were poor extractions. |
| covid_4 | Parliamentary committee | 18.92% | 12.84% | 16 | Petitions, policy statements and meeting operations were treated as follow-up tasks. |
| ES2004a | Product design | 1.54% | 1.54% | 0 | Created a false product requirement from a microphone/setup comment. |
| **Mean** | | **8.63%** | **6.59%** | **6.2** | Good citation validity but weak semantic extraction. |

Every retained claim contained an existing source-segment ID, producing a 100% evidence
link rate. This does not imply semantic correctness. ES2004a's only requirement was
"Product must be standing up there", linked to "Am I supposed to be standing up
there?" during room setup. The citation existed, but the interpretation was false.

A strict manual action audit defined an action as a specific future follow-up task, not
a past activity, petition, policy description, hypothetical, question, negated proposal
or immediate meeting operation. Only approximately four to five of the 31 extracted
actions from Bmr014, Bro019 and covid_4 were plausible under this definition. Estimated
precision was therefore approximately 13–16%, with additional duplication in covid_4.

These meetings averaged 29.29 seconds each when Qwen remained loaded in one evaluation
process. The first meeting included 6.80 seconds of model loading; subsequent meetings
used the resident model and took approximately 28 seconds.

## Test 3: complete ES2011a audio pipeline

ES2011a was selected because it is an official QMSum test meeting, has local AMI audio,
is business/product focused, and was not one of the recent tuning examples.

| Measurement | Result |
| --- | ---: |
| Audio duration | 18.56 minutes |
| Complete processing time | 254.21 seconds |
| Real-time factor | 0.228 (about 4.4× faster than real time) |
| Whisper time | 86.53 seconds |
| Diarization time | 71.81 seconds |
| Qwen analysis time | 91.58 seconds |
| Normalised WER | 35.14% |
| Reference/generated words | 2,826 / 2,122 |
| Substitutions/deletions/insertions | 177 / 760 / 56 |
| Expected/detected main speakers | 4 / 4 |
| Words assigned to unknown speaker | 0.05% |
| Summary token F1 | 11.90% |
| Summary ROUGE-L F1 | 7.14% |

The subprocess architecture completed without exhausting 16 GB system RAM or 8 GB
VRAM. A live GPU snapshot during processing showed 6.38 GB used. The models loaded from
the local cache; no download occurred.

Whisper's main error was deletion: 760 reference words were missing. The coverage
heuristic nevertheless labelled the transcript `good` because it contained 108 words
per minute and speech timestamps covered 91.9% of the audio. This shows that time
coverage and word rate cannot reliably detect missing semantic content.

The brief captured the €25 selling price and one user-interface assignment. It missed:

- the €50 million profit target;
- the four-million-unit sales target;
- the international market;
- most portability, size, weight, usability and button constraints;
- the Industrial Designer's working-design assignment;
- Marketing's user-requirements assignment.

The introduction parser recovered `Krista — Industrial Designer`, but produced the name
`the` for both the Marketing Expert and Project Manager. It also failed to recover the
User Interface Designer's name and role. Against the clearly introduced named people,
only one of three names was correct; three of four roles were represented.

The final action `User Interface: Work on design` was grounded in the two-word fragment
"User interface design". It is directionally related to the meeting, but too vague and
does not represent all three explicit conclusion assignments.

## Test 4: grounded question answering

Three business questions were asked against the human ES2011a transcript. All three
retrieval results contained the necessary answer, but only one generated answer was
complete.

| Question | Retrieval | Answer result |
| --- | --- | --- |
| What was the selling price goal? | Correct evidence selected | Passed: €25 |
| What profit and sales targets were stated? | Included €50m and four million | Failed: Qwen omitted four million |
| Who was assigned to user requirements? | Included Marketing and the assignment | Failed: Qwen answered only "they" |

Exact-term pass rate was therefore 33.3%. This test isolates a generation/interpretation
failure: retrieval succeeded, but Qwen did not use all of the supplied evidence.

## Test 5: implementation reliability

- 58/58 automated tests passed in 1.10 seconds.
- Ruff reported no lint errors in active source and tests.
- Vulture reported no dead code at the configured 80% confidence threshold.
- All source and test modules compiled successfully.
- Streamlit started and completed model-backed user interactions successfully.

These checks demonstrate stable orchestration and validation logic. They do not measure
model comprehension, which is why the manual tests found failures despite the clean
test suite.

## Overall judgement

The project is a credible research prototype, not yet a reliable autonomous meeting
assistant. Its strongest properties are local execution, recoverable model isolation,
session caching, evidence traceability, diarization speaker-count recovery, clear review
controls and deterministic rejection of some unsupported output.

Its principal weakness is semantic precision. A valid transcript citation currently
proves only that the source exists, not that the category or interpretation is correct.
The deterministic recovery layer is the most urgent problem because it can override a
cautious Qwen result and manufacture many false actions or requirements from genuine
but irrelevant transcript lines.

## Prioritised changes before final evaluation

1. Disable broad deterministic action recovery by default. Retain only explicit future
   commitments or assignments with a valid actor, action verb and non-negated clause.
2. Reject setup language, quotations, petitions, past-tense reports, conditional offers,
   policy descriptions and immediate meeting operations from actions/requirements.
3. Fix introduction parsing so articles such as `the` cannot become names and test the
   patterns against Chiara, Stephanie and Krista in ES2011a.
4. Improve conclusion-assignment parsing across adjacent speaker turns and role labels.
5. Add financial-target recovery for multiple values in one evidence window rather than
   stopping after the first price.
6. Require Q&A answers to cover each requested subpart and copy supported names/numbers
   from evidence before generation finishes.
7. Add an evidence-category entailment check: a cited segment should have to support the
   claimed category, not merely share words with the claim.
8. Replace the current coverage gate with a reference-free missing-speech diagnostic
   using long-gap distribution, speech-activity duration and repeated low-confidence
   regions.
9. Fix speaker-count display and timing fields, and disable Streamlit file watching for
   model packages to keep the terminal usable.
10. Repeat this exact manifest after changes. Do not tune on these meetings and then call
    them held out; create a new locked final manifest for the final report.

## Post-fix regression evaluation

The defects above were then used to guide pipeline version 4.4. Consequently, this is
a regression evaluation, not a new held-out result. The same five transcripts were
rerun end-to-end through Qwen and evidence validation after increasing the structured
output budget from 320 to 512 tokens and tightening category validation.

| Meeting | Token F1 before | Token F1 after | ROUGE-L before | ROUGE-L after |
| --- | ---: | ---: | ---: | ---: |
| Bed008 | 7.25% | 5.93% | 5.80% | 5.93% |
| Bmr014 | 3.36% | 11.20% | 3.36% | 6.40% |
| Bro019 | 12.08% | 14.29% | 9.40% | 12.99% |
| covid_4 | 18.92% | 13.17% | 12.84% | 11.52% |
| ES2004a | 1.54% | 18.06% | 1.54% | 10.32% |
| **Mean** | **8.63%** | **12.53%** | **6.59%** | **9.43%** |

This is a 45.2% relative increase in mean token F1 and a 43.1% relative increase in
mean ROUGE-L. More importantly for practical use, action output fell from 31 mostly
false items to two candidates in the measured rerun. A final post-filter removed the
remaining Bmr014 email fragment, leaving only covid_4's explicit commitment to check
what happened and provide a solution next time. That candidate is reviewable rather
than silently treated as confirmed truth.

ES2004a changed from the false requirement "Product must be standing up there" to a
grounded set containing product qualities, a €25 selling price, a €50 million profit
target, international scope and all-age usability. A final deterministic regression
also recovers "production cost's at twelve fifty" as €12.50 and records its stated
relationship as 50% of the selling price. This final addition was unit- and
transcript-tested after the model rerun, so it is not included in the table's summary
score.

The larger response budget increased mean resident-model analysis time from 29.29 to
44.59 seconds for these long transcripts. This is an intentional quality trade-off:
generation remains bounded by the 65-second per-window and 360-second worker limits,
and identical repeated analyses still return from the versioned session cache without
model inference.

The three ES2011a Q&A cases were replayed against the same retrieved evidence. The
selling-price answer remained correct. Evidence-bound completion added the omitted
four-million-unit sales target to the €50 million profit answer and replaced the vague
pronoun answer with the explicit Marketing role and source segment IDs. Required-fact
coverage therefore improved from 1/3 to 3/3 on this regression set; this small result
must not be presented as a general Q&A accuracy estimate.

The final implementation check passed 68/68 automated tests. Ruff, Vulture at 80%
confidence, and Python bytecode compilation all completed without findings. The fixes
also cover the previously observed transcript-only speaker count, zero analysis timing,
and terminal trace flooding. A genuinely held-out final manifest is still required for
an unbiased headline result, and the remaining summary scores show that the 1.5B model
is useful as a review-assisted extractor rather than a fully autonomous meeting analyst.
