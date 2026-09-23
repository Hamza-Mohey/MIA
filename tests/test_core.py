from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.metrics import extraction_metrics
from modules.analysis_workflow import analyse_segments_with_cache
from modules.audio_transcription import (
    _merge_transcription_passes,
    temporary_audio_file,
    transcribe_audio,
)
from modules.cache_keys import stage_cache_key
from modules.evidence import (
    build_analysis_digest,
    build_source_segments,
    link_analysis_evidence,
)
from modules.general_meeting import (
    build_general_context,
    build_grounded_summary,
    generate_general_report,
    run_general_pipeline,
)
from modules.isolated_inference import (
    run_model_worker,
    transcribe_audio_with_speakers_isolated,
)
from modules.retrieval import build_retrieval_index, retrieve_evidence
from modules.speaker_diarization import (
    assign_speakers_to_segments,
    format_speaker_transcript,
)
from modules.text_analysis import (
    _analyse_once,
    _complete_grounded_answer,
    extract_json_from_text,
    normalise_transcript_analysis,
    split_transcript,
)


class FakeUpload:
    name = "meeting.wav"

    def getvalue(self):
        return b"fake audio bytes"


class DiarizationTests(unittest.TestCase):
    def test_assigns_speaker_with_greatest_overlap(self):
        transcription = [
            {"timestamp": (0.5, 2.5), "text": "First point"},
            {"timestamp": (3.1, 4.0), "text": "Second point"},
        ]
        turns = [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 5.0, "speaker": "SPEAKER_01"},
        ]
        result = assign_speakers_to_segments(transcription, turns)
        self.assertEqual(result[0]["speaker"], "SPEAKER_00")
        self.assertEqual(result[1]["speaker"], "SPEAKER_01")
        self.assertIn("SPEAKER_00: First point", format_speaker_transcript(result))

    def test_unknown_speaker_when_no_turn_overlaps(self):
        result = assign_speakers_to_segments(
            [{"timestamp": (10.0, 11.0), "text": "Outside"}],
            [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
        )
        self.assertEqual(result[0]["speaker"], "SPEAKER_UNKNOWN")


class JsonTests(unittest.TestCase):
    def test_production_prompt_has_no_content_bearing_demo_facts(self):
        prompt = Path("prompts/meeting_intelligence_prompt.txt").read_text(encoding="utf-8")
        self.assertNotIn("revised agenda", prompt.casefold())
        self.assertNotIn("friday at 10", prompt.casefold())

    def test_clean_json(self):
        value = extract_json_from_text('{"topics": ["Budget"]}')
        self.assertEqual(value["topics"], ["Budget"])

    def test_json_with_extra_text(self):
        value = extract_json_from_text('Result follows:\n```json\n{"topics": ["Login"]}\n```\nDone')
        self.assertEqual(value["topics"], ["Login"])

    def test_invalid_json_fallback_normalises(self):
        self.assertIsNone(extract_json_from_text("{not valid}"))
        result = normalise_transcript_analysis({"action_items": "Review notes"})
        self.assertEqual(result["action_items"][0]["task"], "Review notes")

    def test_long_transcript_is_bounded_without_losing_lines(self):
        text = "\n".join(["SPEAKER_00: " + "x" * 30, "SPEAKER_01: " + "y" * 30])
        chunks = split_transcript(text, max_characters=45)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 45 for chunk in chunks))
        self.assertIn("SPEAKER_01", "\n".join(chunks))

    def test_truncated_model_json_is_repaired_without_a_second_generation(self):
        with (
            patch("modules.text_analysis.load_model", return_value=(object(), object())),
            patch(
                "modules.text_analysis._generate_chat",
                return_value=('{"meeting_summary": "Budget review", "topics": ["Budget"]'),
            ) as generate,
        ):
            result = _analyse_once("SPEAKER_00: Review the budget.", 0.95, "test")

        self.assertEqual(generate.call_count, 1)
        self.assertEqual(result["meeting_summary"], "Budget review")
        self.assertEqual(result["topics"][0]["text"], "Budget")


class GeneralMeetingTests(unittest.TestCase):
    def test_context_keeps_general_meeting_evidence(self):
        context = build_general_context(
            {
                "meeting_summary": "A weekly planning meeting.",
                "participants": [{"speaker": "SPEAKER_00", "name": None}],
                "decisions": ["Move the review to Friday."],
                "action_items": [{"task": "Send the agenda", "owner": "SPEAKER_00"}],
                "open_questions": ["Which room is available?"],
            },
            diarization={"model": "pyannote/test", "speaker_count": 2},
        )
        self.assertEqual(context["meeting_mode"], "general")
        self.assertEqual(context["decisions"][0]["text"], "Move the review to Friday.")
        self.assertEqual(context["quality"]["speaker_count"], 2)
        self.assertIn("General Meeting", generate_general_report(context))

    def test_diarization_speakers_are_used_when_qwen_omits_participants(self):
        context = build_general_context(
            {"meeting_summary": "A meeting."},
            diarization={
                "model": "pyannote/test",
                "speaker_count": 2,
                "speakers": ["SPEAKER_00", "SPEAKER_01"],
            },
        )

        self.assertEqual(
            [item["speaker"] for item in context["participants"]],
            ["SPEAKER_00", "SPEAKER_01"],
        )

    def test_unknown_speaker_is_not_counted_as_a_participant(self):
        context = build_general_context(
            {
                "participants": [
                    {"speaker": "SPEAKER_UNKNOWN", "name": None},
                    {"speaker": "SPEAKER_00", "name": None},
                ]
            },
            diarization={
                "model": "pyannote/test",
                "speaker_count": 1,
                "speakers": ["SPEAKER_00"],
            },
        )

        self.assertEqual(
            [item["speaker"] for item in context["participants"]],
            ["SPEAKER_00"],
        )

    def test_pipeline_is_dependency_injectable(self):
        result = run_general_pipeline(
            "SPEAKER_00: We approved Friday.",
            text_analyser=lambda *_args, **_kwargs: {
                "meeting_summary": "Friday was approved.",
                "decisions": ["Meet on Friday."],
            },
        )
        self.assertEqual(result["schema_version"], "3.0")
        self.assertIn("Meet on Friday", result["report_markdown"])


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.audio_result = {
            "segments": [
                {"text": "We approved the Friday launch.", "timestamp": [2.0, 4.5]},
                {
                    "text": "Sarah, please circulate the agenda by Wednesday.",
                    "timestamp": [4.5, 8.0],
                },
            ]
        }
        self.segments = build_source_segments(
            "SPEAKER_00: We approved the Friday launch.\n"
            "SPEAKER_01: Sarah, please circulate the agenda by Wednesday.",
            self.audio_result,
        )

    def test_source_segments_keep_stable_ids_and_audio_times(self):
        self.assertEqual([item["id"] for item in self.segments], ["seg-0001", "seg-0002"])
        self.assertEqual(self.segments[1]["speaker"], "SPEAKER_01")
        self.assertEqual(self.segments[1]["start"], 4.5)
        self.assertEqual(self.segments[1]["end"], 8.0)

    def test_claimed_evidence_is_replaced_with_actual_source_text(self):
        linked = link_analysis_evidence(
            {
                "decisions": [],
                "action_items": [
                    {
                        "task": "Circulate the agenda",
                        "evidence": "A fabricated quotation",
                        "evidence_segment_ids": ["seg-0002", "seg-does-not-exist"],
                    }
                ],
            },
            self.segments,
        )["action_items"][0]

        self.assertEqual(linked["evidence_segment_ids"], ["seg-0002"])
        self.assertEqual(linked["evidence"], "Sarah, please circulate the agenda by Wednesday.")
        self.assertEqual(linked["model_evidence"], "A fabricated quotation")
        self.assertEqual(linked["evidence_start_seconds"], 4.5)
        self.assertEqual(linked["evidence_status"], "linked")

    def test_unverifiable_evidence_is_not_displayed_as_a_source_quote(self):
        linked = link_analysis_evidence(
            {
                "decisions": [
                    {
                        "text": "Purchase a new office building",
                        "evidence": "The purchase was unanimously approved.",
                        "evidence_segment_ids": ["seg-9999"],
                    }
                ],
                "action_items": [],
            },
            self.segments,
        )

        self.assertEqual(linked["decisions"], [])
        self.assertEqual(linked["evidence_rejections"][0]["type"], "decision")
        self.assertIn("unsupported analytical", linked["uncertainties"][0])

    def test_overbroad_model_citations_are_pruned_to_relevant_segments(self):
        segments = [
            {
                "id": f"seg-{index:04d}",
                "index": index - 1,
                "speaker": "SPEAKER_00",
                "text": text,
                "start": float(index),
                "end": float(index + 1),
            }
            for index, text in enumerate(
                [
                    "The weather is warm.",
                    "The room is available.",
                    "Sarah will circulate the revised proposal tomorrow.",
                    "Lunch starts at noon.",
                    "The projector is working.",
                ],
                start=1,
            )
        ]
        linked = link_analysis_evidence(
            {
                "decisions": [],
                "action_items": [
                    {
                        "task": "Circulate the revised proposal",
                        "evidence_segment_ids": [item["id"] for item in segments],
                    }
                ],
            },
            segments,
        )["action_items"][0]

        self.assertLessEqual(len(linked["evidence_segment_ids"]), 3)
        self.assertIn("seg-0003", linked["evidence_segment_ids"])

    def test_existing_segment_id_does_not_validate_an_unsupported_decision(self):
        linked = link_analysis_evidence(
            {
                "decisions": [
                    {
                        "text": "Adopt a digital marketing strategy",
                        "evidence_segment_ids": ["seg-0002"],
                    }
                ],
                "action_items": [],
            },
            self.segments,
        )

        self.assertEqual(linked["decisions"], [])
        self.assertTrue(linked["uncertainties"])

    def test_action_owner_and_deadline_are_grounded_in_evidence(self):
        linked = link_analysis_evidence(
            {
                "decisions": [],
                "action_items": [
                    {
                        "task": "Send the agenda to Sarah",
                        "owner": "SPEAKER_01",
                        "deadline": "Wednesday",
                        "evidence_segment_ids": ["seg-0002"],
                    }
                ],
            },
            self.segments,
        )["action_items"][0]

        self.assertEqual(linked["owner"], "Sarah")
        self.assertEqual(linked["task"], "Send the agenda")
        self.assertEqual(linked["deadline"], "Wednesday")

    def test_unlinked_key_point_is_removed_like_an_unlinked_decision(self):
        linked = link_analysis_evidence(
            {
                "key_points": [
                    {
                        "text": "The team adopted a digit ritual",
                        "evidence_segment_ids": ["seg-9999"],
                    }
                ]
            },
            self.segments,
        )

        self.assertEqual(linked["key_points"], [])
        self.assertEqual(linked["evidence_rejections"][0]["type"], "key point")

    def test_supported_topic_paraphrase_uses_topic_specific_threshold(self):
        segments = build_source_segments(
            "SPEAKER_00: We are going to look at people's brains in a more direct way."
        )
        linked = link_analysis_evidence(
            {
                "topics": [
                    {
                        "text": "Brain imaging research",
                        "evidence_segment_ids": ["seg-0001"],
                    }
                ]
            },
            segments,
        )

        self.assertEqual(len(linked["topics"]), 1)
        self.assertEqual(linked["topics"][0]["evidence_status"], "linked")

    def test_wrong_topic_citation_is_repaired_from_actual_source(self):
        segments = build_source_segments(
            "SPEAKER_00: We discussed the budget.\n"
            "SPEAKER_01: We may start looking at people's brains directly with fMRI.\n"
            "SPEAKER_02: The meeting is finished."
        )
        linked = link_analysis_evidence(
            {
                "topics": [
                    {
                        "text": "Brain imaging research",
                        "evidence_segment_ids": ["seg-0001"],
                    }
                ]
            },
            segments,
        )["topics"][0]

        self.assertTrue(linked["citation_repaired"])
        self.assertIn("seg-0002", linked["evidence_segment_ids"])

    def test_character_similarity_without_token_overlap_does_not_ground_claim(self):
        linked = link_analysis_evidence(
            {
                "topics": [
                    {
                        "text": "Brain imaging",
                        "evidence_segment_ids": ["seg-0001"],
                    }
                ]
            },
            build_source_segments("SPEAKER_00: Sorry, I agree with that."),
        )

        self.assertEqual(linked["topics"], [])

    def test_unsupported_distinctive_term_is_not_grounded_by_generic_overlap(self):
        linked = link_analysis_evidence(
            {
                "key_points": [
                    {
                        "text": "BibTeX ensures functionality across projections",
                        "evidence_segment_ids": ["seg-0001"],
                    }
                ]
            },
            build_source_segments(
                "SPEAKER_00: The functionality should interact with all projections."
            ),
        )

        self.assertEqual(linked["key_points"], [])

    def test_hypothetical_should_statement_is_not_recovered_as_action(self):
        linked = link_analysis_evidence(
            {"action_items": []},
            build_source_segments(
                "SPEAKER_00: If you introduce a construction, it should interact with projections."
            ),
        )

        self.assertEqual(linked["action_items"], [])

    def test_concrete_suggestion_is_kept_as_suggested_action(self):
        segments = build_source_segments(
            "SPEAKER_00: You could always send me comments by electronic mail."
        )
        linked = link_analysis_evidence(
            {
                "action_items": [
                    {
                        "task": "Send comments by electronic mail",
                        "evidence_segment_ids": ["seg-0001"],
                    }
                ]
            },
            segments,
        )["action_items"][0]

        self.assertEqual(linked["commitment_status"], "suggested")
        self.assertEqual(linked["evidence_status"], "linked")

    def test_explicit_follow_up_is_recovered_when_qwen_omits_it(self):
        segments = build_source_segments(
            "SPEAKER_00: You might want to double-check the author names."
        )
        linked = link_analysis_evidence({"action_items": []}, segments)

        self.assertEqual(len(linked["action_items"]), 1)
        self.assertEqual(linked["action_items"][0]["commitment_status"], "suggested")
        self.assertEqual(
            linked["action_items"][0]["source"], "deterministic_cue_recovery"
        )

    def test_split_action_fragments_are_merged_and_cleaned(self):
        segments = build_source_segments(
            "SPEAKER_00: Yeah, you may have received a note asking you to send me the\n"
            "SPEAKER_00: you to send me the current formalism that you presented."
        )
        linked = link_analysis_evidence({"action_items": []}, segments)

        self.assertEqual(len(linked["action_items"]), 1)
        self.assertEqual(
            linked["action_items"][0]["task"],
            "Send me the current formalism that you presented",
        )
        self.assertEqual(
            linked["action_items"][0]["evidence_segment_ids"],
            ["seg-0001", "seg-0002"],
        )

    def test_grounded_open_question_is_preserved(self):
        segments = build_source_segments(
            "SPEAKER_00: Which lexical example should we include in the proposal?"
        )
        linked = link_analysis_evidence(
            {
                "open_questions": [
                    {
                        "text": "Which lexical example should be included?",
                        "evidence_segment_ids": ["seg-0001"],
                    }
                ]
            },
            segments,
        )

        self.assertEqual(len(linked["open_questions"]), 1)

    def test_analysis_digest_prioritises_outcomes_and_limits_size(self):
        transcript = "\n".join(
            [
                f"SPEAKER_00: Background discussion item {index} about the product design."
                for index in range(24)
            ]
            + [
                "SPEAKER_01: I will send the revised budget to Sarah tomorrow.",
                "SPEAKER_02: The supplier delay is a risk for the launch date.",
                "SPEAKER_00: Should we move the launch to next week?",
            ]
        )
        digest = build_analysis_digest(
            build_source_segments(transcript), max_characters=900
        )

        self.assertLessEqual(len(digest["text"]), 900)
        self.assertIn("send the revised budget", digest["text"])
        self.assertIn("supplier delay is a risk", digest["text"])
        self.assertGreater(digest["total_segments"], digest["selected_segments"])

    def test_analysis_digest_omits_repeated_calibration_noise(self):
        transcript = "\n".join(
            ["SPEAKER_00: one two three four five microphone test"] * 6
            + ["SPEAKER_01: We agreed to use the revised proposal for the launch."]
        )
        digest = build_analysis_digest(
            build_source_segments(transcript), max_characters=500
        )

        self.assertIn("agreed to use the revised proposal", digest["text"])
        self.assertNotIn("microphone test", digest["text"])

    def test_purpose_and_presentation_replace_preagenda_small_talk(self):
        segments = build_source_segments(
            "SPEAKER_00: I just had the most frustrating meeting of my career.\n"
            "SPEAKER_01: We are talking about Robert's thesis proposal today.\n"
            "SPEAKER_01: I am going to present a little talk at EML about our work."
        )
        linked = link_analysis_evidence(
            {
                "topics": [
                    {
                        "text": "Meeting frustration",
                        "evidence_segment_ids": ["seg-0001"],
                    }
                ]
            },
            segments,
        )
        topics = [item["text"] for item in linked["topics"]]

        self.assertNotIn("Meeting frustration", topics)
        self.assertIn("Robert's thesis proposal", topics)
        self.assertIn("EML presentation", topics)

        digest = build_analysis_digest(segments, max_characters=600)
        self.assertNotIn("most frustrating meeting", digest["text"])
        self.assertIn("Robert's thesis proposal", digest["text"])

    def test_repeated_words_do_not_create_a_synthetic_topic(self):
        lines = ["SPEAKER_00: We are discussing the quarterly launch plan today."]
        for index in range(15):
            if index in {1, 7, 13}:
                lines.append(
                    "SPEAKER_01: Customer retention and onboarding remain connected priorities."
                )
            else:
                lines.append(f"SPEAKER_02: Operational update number {index}.")
        segments = build_source_segments("\n".join(lines))

        linked = link_analysis_evidence({"topics": []}, segments)
        self.assertFalse(
            any(
                item.get("source") == "deterministic_recurrent_theme"
                for item in linked["topics"]
            )
        )

    def test_introductions_recover_names_roles_and_speaker_labels(self):
        segments = build_source_segments(
            "SPEAKER_03: I'm Laura, and I'm the project manager.\n"
            "SPEAKER_00: Hi, I'm David, and I'm supposed to be an industrial designer.\n"
            "SPEAKER_02: And I'm Andrew, and I'm our marketing expert.\n"
            "SPEAKER_01: I'm Craig, and I'm user interface."
        )

        linked = link_analysis_evidence({}, segments)

        self.assertEqual(
            [(item["speaker"], item["name"], item["role"]) for item in linked["participants"]],
            [
                ("SPEAKER_03", "Laura", "Project Manager"),
                ("SPEAKER_00", "David", "Industrial Designer"),
                ("SPEAKER_02", "Andrew", "Marketing Expert"),
                ("SPEAKER_01", "Craig", "User Interface"),
            ],
        )

    def test_my_name_introduction_never_treats_article_as_name(self):
        linked = link_analysis_evidence(
            {},
            build_source_segments(
                "SPEAKER_01: My name is Chiara, and I'm the marketing expert.\n"
                "SPEAKER_02: I forgot to say I'm the project manager.\n"
                "SPEAKER_00: I'm Stephanie and I am the user interface designer."
            ),
        )

        participants = {
            item["speaker"]: (item.get("name"), item.get("role"))
            for item in linked["participants"]
        }
        self.assertEqual(participants["SPEAKER_01"], ("Chiara", "Marketing Expert"))
        self.assertEqual(participants["SPEAKER_00"], ("Stephanie", "User Interface"))
        self.assertNotIn("SPEAKER_02", participants)

    def test_roll_call_reconciles_a_close_asr_name(self):
        segments = build_source_segments(
            "SPEAKER_00: I'm David, and I'm the industrial designer.\n"
            "SPEAKER_02: I'm Andrew, and I'm the marketing expert.\n"
            "SPEAKER_01: I'm Greg, and I'm future interface.\n"
            "SPEAKER_03: So that's David, Andrew, and Craig, isn't it?"
        )

        linked = link_analysis_evidence({}, segments)

        reconciled = next(item for item in linked["participants"] if item["speaker"] == "SPEAKER_01")
        self.assertEqual(reconciled["name"], "Craig")
        self.assertEqual(reconciled["name_reconciled_from"], "Greg")

    def test_business_requirements_are_grounded_and_finance_is_normalised(self):
        segments = build_source_segments(
            "SPEAKER_03: The product is supposed to be original, trendy and user-friendly.\n"
            "SPEAKER_03: We are selling this remote control for 25 euro.\n"
            "SPEAKER_03: We do not want it to cost any more than 1250 euro.\n"
            "SPEAKER_03: That is 50% of the selling price.\n"
            "SPEAKER_03: We are selling this on an international scale."
        )

        linked = link_analysis_evidence({}, segments)
        requirements = [item["text"] for item in linked["requirements"]]

        self.assertTrue(any("original, trendy" in item for item in requirements))
        self.assertIn("Selling price: €25.", requirements)
        self.assertTrue(any("€12.50" in item for item in requirements))
        self.assertIn("International market.", requirements)

    def test_setup_language_is_not_a_requirement_and_all_targets_are_recovered(self):
        segments = build_source_segments(
            "Project Manager: Am I supposed to be standing up there?\n"
            "Project Manager: Our selling price goal is twenty five Euro and profit aim is fifty million Euro.\n"
            "Project Manager: We hope to sell this internationally.\n"
            "Marketing: How many should we sell then?\n"
            "Marketing: two million, no, more, four million.\n"
            "Industrial Designer: Two million.\n"
            "Marketing: Four million.\n"
            "Marketing: If profit for each is twelve fifty, that'll do four million."
        )

        requirements = [
            item["text"] for item in link_analysis_evidence({}, segments)["requirements"]
        ]

        self.assertFalse(any("standing" in item.casefold() for item in requirements))
        self.assertIn("Selling price: €25.", requirements)
        self.assertIn("Revenue target: €50 million.", requirements)
        self.assertIn("Sales target: 4 million units.", requirements)
        self.assertIn("International market.", requirements)

    def test_meeting_style_business_constraints_are_recovered(self):
        segments = build_source_segments(
            "Project Manager: We want it to be original, trendy, appealing to a wide market, "
            "recognisable, and user-friendly while still doing something different.\n"
            "Project Manager: The selling price at twenty five Euros.\n"
            "Project Manager: The production cost's at twelve fifty.\n"
            "Project Manager: The market range is international.\n"
            "Project Manager: It must be accessible and usable by all age groups."
        )

        requirements = [
            item["text"] for item in link_analysis_evidence({}, segments)["requirements"]
        ]

        self.assertTrue(any("original" in item and "user-friendly" in item for item in requirements))
        self.assertIn("Selling price: €25.", requirements)
        self.assertIn("Maximum production cost: €12.50.", requirements)
        self.assertIn("International market.", requirements)
        self.assertIn("Usable across all age groups.", requirements)

    def test_wrap_up_role_assignments_are_recovered(self):
        segments = build_source_segments(
            "SPEAKER_03: Just to wrap up, the next meeting is in thirty minutes.\n"
            "SPEAKER_03: As the industrial designer, you're going to be working on the physical design.\n"
            "SPEAKER_03: Marketing executive, you'll be thinking about the requirements."
        )

        linked = link_analysis_evidence({}, segments)
        actions = {(item.get("owner"), item["task"]) for item in linked["action_items"]}

        self.assertIn(("Industrial Designer", "Work on the physical design"), actions)
        self.assertIn(("Marketing Executive", "Determine the requirements"), actions)

    def test_wrap_up_assignments_can_span_adjacent_speakers(self):
        segments = build_source_segments(
            "Project Manager: We should start wrapping up before the next meeting.\n"
            "Project Manager: The industrial designer is going to be looking more into the working design.\n"
            "Industrial Designer: Yes.\n"
            "Project Manager: What does UI stand for?\n"
            "User Interface: User interface design, so technical function.\n"
            "Project Manager: And marketing?\n"
            "Marketing: Marketing.\n"
            "Project Manager: So we'll be working on the user requirements."
        )

        actions = {
            (item.get("owner_role"), item["task"])
            for item in link_analysis_evidence({}, segments)["action_items"]
        }

        self.assertIn(("Industrial Designer", "Work on the working design"), actions)
        self.assertIn(("User Interface", "Define the technical functions"), actions)
        self.assertIn(("Marketing", "Determine the user requirements"), actions)

    def test_role_assignment_owner_resolves_to_introduced_name(self):
        segments = build_source_segments(
            "SPEAKER_00: Hi, I'm David, and I'm the industrial designer.\n"
            "SPEAKER_03: Some planning discussion.\n"
            "SPEAKER_03: Just to wrap up, the next meeting is tomorrow.\n"
            "SPEAKER_03: As the industrial designer, you're going to be working on the physical design."
        )

        linked = link_analysis_evidence({}, segments)

        self.assertEqual(linked["action_items"][0]["owner"], "David")
        self.assertEqual(linked["action_items"][0]["owner_role"], "Industrial Designer")

    def test_questions_and_in_meeting_operations_are_not_actions(self):
        segments = build_source_segments(
            "SPEAKER_03: I'll just check if there's nothing else.\n"
            "SPEAKER_03: Is that something we'd want to include, do you think?"
        )

        linked = link_analysis_evidence(
            {
                "action_items": [
                    {"task": segments[0]["text"], "evidence_segment_ids": ["seg-0001"]},
                    {"task": segments[1]["text"], "evidence_segment_ids": ["seg-0002"]},
                ]
            },
            segments,
        )

        self.assertEqual(linked["action_items"], [])

    def test_conditional_examples_and_interface_descriptions_are_not_actions(self):
        segments = build_source_segments(
            "Speaker: Certainly provide.\n"
            "Speaker: Send it through email you're thinking.\n"
            'Speaker: Just put the button on the web page which says "send me scripts".\n'
            "Speaker: Wanna hear this in context if you need that, issue them a password.\n"
            "The Chair: Check what happened with that interruption."
        )

        linked = link_analysis_evidence(
            {
                "action_items": [
                    {
                        "task": segment["text"],
                        "evidence_segment_ids": [segment["id"]],
                    }
                    for segment in segments
                ]
            },
            segments,
        )

        self.assertEqual(linked["action_items"], [])

    def test_negated_past_and_proceeding_statements_are_not_actions(self):
        segments = build_source_segments(
            "Professor: I don't think we can send the transcript through email.\n"
            "Researcher: What I did was review the earlier experiment.\n"
            "Member: Thank you, Mr. Chair. I am presenting a petition.\n"
            "Chair: We'll check what happened, but I stopped the clock."
        )
        linked = link_analysis_evidence(
            {
                "action_items": [
                    {"task": segment["text"], "evidence_segment_ids": [segment["id"]]}
                    for segment in segments
                ]
            },
            segments,
        )

        self.assertEqual(linked["action_items"], [])


class GroundedSummaryTests(unittest.TestCase):
    def test_summary_uses_only_validated_claims(self):
        result = build_grounded_summary(
            {
                "topics": [{"text": "Launch planning", "evidence_status": "linked"}],
                "key_points": [
                    {
                        "text": "The Friday launch was approved",
                        "evidence_status": "linked",
                        "evidence_start_seconds": 12.0,
                    }
                ],
            }
        )

        self.assertIn("Launch planning", result)
        self.assertIn("Friday launch was approved", result)

    def test_summary_prioritises_requirements_decisions_and_assignments(self):
        result = build_grounded_summary(
            {
                "topics": [
                    {
                        "text": "Remote control design",
                        "source": "deterministic_purpose_recovery",
                        "evidence_status": "linked",
                    }
                ],
                "requirements": [
                    {"text": "Selling price: €25.", "evidence_status": "linked"}
                ],
                "decisions": [
                    {"text": "Combine several devices", "evidence_status": "linked"}
                ],
                "action_items": [
                    {
                        "task": "Work on the physical design",
                        "owner": "Industrial Designer",
                        "commitment_status": "confirmed",
                        "evidence_status": "linked",
                        "source": "deterministic_role_assignment",
                    }
                ],
            }
        )

        self.assertIn("Remote control design", result)
        self.assertIn("Selling price", result)
        self.assertIn("Combine several devices", result)
        self.assertIn("Industrial Designer", result)


class CacheTests(unittest.TestCase):
    def test_cache_key_changes_with_model_prompt_or_configuration(self):
        base = stage_cache_key(
            "analysis",
            "same transcript",
            versions={"model": "v1", "prompt": "p1"},
            configuration={"tokens": 100},
        )
        changed_model = stage_cache_key(
            "analysis",
            "same transcript",
            versions={"model": "v2", "prompt": "p1"},
            configuration={"tokens": 100},
        )
        changed_prompt = stage_cache_key(
            "analysis",
            "same transcript",
            versions={"model": "v1", "prompt": "p2"},
            configuration={"tokens": 100},
        )
        changed_configuration = stage_cache_key(
            "analysis",
            "same transcript",
            versions={"model": "v1", "prompt": "p1"},
            configuration={"tokens": 200},
        )
        self.assertEqual(len({base, changed_model, changed_prompt, changed_configuration}), 4)

    def test_analysis_digest_is_cached_and_invalidated_when_changed(self):
        first_digest = {
            "index": 0,
            "segment_ids": ["seg-0001"],
            "text": "first digest",
            "total_segments": 10,
            "selected_segments": 4,
        }
        second_digest = {**first_digest, "index": 1, "text": "second digest"}
        edited_digest = {**second_digest, "text": "edited digest"}
        calls: list[list[str]] = []

        def analyser(chunks, **_kwargs):
            calls.append(chunks)
            return {
                "analyses": [
                    normalise_transcript_analysis({"meeting_summary": chunk}) for chunk in chunks
                ],
                "_runtime": {"inference_seconds": 0.01},
            }

        cache = {}
        arguments = {
            "segments": [{"id": f"seg-{index:04d}"} for index in range(10)],
            "cache": cache,
            "transcript_quality": 0.9,
            "quality_basis": "test",
            "versions": {"model": "v1", "prompt": "p1"},
            "configuration": {"tokens": 100},
            "batch_analyser": analyser,
        }
        with patch(
            "modules.analysis_workflow.build_analysis_digests",
            side_effect=[
                [first_digest, second_digest],
                [first_digest, second_digest],
                [first_digest, edited_digest],
            ],
        ):
            _, first_stats = analyse_segments_with_cache(**arguments)
            _, repeat_stats = analyse_segments_with_cache(**arguments)
            _, edited_stats = analyse_segments_with_cache(**arguments)

        expected_miss = {
            "chunks": 2,
            "hits": 0,
            "misses": 2,
            "source_segments": 10,
            "selected_segments": 8,
        }
        self.assertEqual(first_stats, expected_miss)
        self.assertEqual(
            repeat_stats,
            {**expected_miss, "hits": 2, "misses": 0},
        )
        self.assertEqual(
            edited_stats,
            {**expected_miss, "hits": 1, "misses": 1},
        )
        self.assertEqual(
            calls,
            [["first digest", "second digest"], ["edited digest"]],
        )


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.segments = [
            {
                "id": "seg-0001",
                "speaker": "SPEAKER_00",
                "text": "The launch date is still being discussed.",
                "start": 0.0,
                "end": 2.0,
            },
            {
                "id": "seg-0002",
                "speaker": "SPEAKER_01",
                "text": "Mariam will follow up with the vendor.",
                "start": 2.0,
                "end": 5.0,
            },
            {
                "id": "seg-0003",
                "speaker": "SPEAKER_00",
                "text": "She will report back tomorrow.",
                "start": 5.0,
                "end": 7.0,
            },
        ]

    def test_semantic_match_finds_different_wording_and_keeps_neighbours(self):
        index = build_retrieval_index(
            self.segments,
            embedder=lambda _texts: [[0.0, 1.0], [1.0, 0.0], [0.2, 0.8]],
        )
        result = retrieve_evidence(
            "Who is responsible for contacting the supplier?",
            index,
            query_embedding=[1.0, 0.0],
        )

        self.assertEqual(result["mode"], "hybrid")
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["ranked_candidates"][0]["segment_id"], "seg-0002")
        self.assertIn("seg-0001", [item["id"] for item in result["selected_segments"]])
        self.assertIn("seg-0003", [item["id"] for item in result["selected_segments"]])

    def test_low_score_is_retrieval_failure_not_proof_of_absence(self):
        index = build_retrieval_index(
            self.segments,
            embedder=lambda _texts: [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        )
        result = retrieve_evidence(
            "Was quantum encryption selected?",
            index,
            query_embedding=[0.0, 1.0],
        )

        self.assertEqual(result["status"], "low_confidence")
        self.assertEqual(result["best_score"], 0.0)

    def test_multi_part_answer_recovers_supported_sales_target(self):
        answer = _complete_grounded_answer(
            "What profit and sales targets were stated?",
            "\n".join(
                [
                    "[seg-0087] Project Manager: Profit aim fifty million Euro.",
                    "[seg-0089] Marketing: How many should we sell then?",
                    "[seg-0090] Marketing: two million, no, more, four million.",
                    "[seg-0092] Marketing: Four million.",
                ]
            ),
            "The profit target was 50 million euros.",
        )

        self.assertIn("sales target 4 million units", answer)

    def test_multi_part_answer_handles_implicit_sales_calculation(self):
        answer = _complete_grounded_answer(
            "What profit and sales targets were stated?",
            "\n".join(
                [
                    "[seg-0087] Project Manager: Profit aim fifty million Euro.",
                    "[seg-0095] Marketing: if profit for each is twelve fifty, "
                    "that'll do four million.",
                ]
            ),
            "The profit target was 50 million euros.",
        )

        self.assertIn("sales target 4 million units", answer)

    def test_who_answer_replaces_ambiguous_role_with_grounded_role(self):
        answer = _complete_grounded_answer(
            "Who was assigned to work on user requirements?",
            "\n".join(
                [
                    "[seg-0273] Marketing: Marketing.",
                    "[seg-0274] Project Manager: So we'll be working on the user requirements.",
                ]
            ),
            "The project manager said that they would be working on user requirements.",
        )

        self.assertTrue(answer.startswith("Marketing was assigned"))

    def test_who_answer_uses_role_speaker_immediately_before_assignment(self):
        answer = _complete_grounded_answer(
            "Who was assigned to work on user requirements?",
            "\n".join(
                [
                    "[seg-0273] Marketing: Marketing. Oh it's written here.",
                    "[seg-0274] Project Manager: So we'll be working on the user requirements.",
                ]
            ),
            "The project manager said that they would work on user requirements.",
        )

        self.assertTrue(answer.startswith("Marketing was assigned"))


class OrchestrationTests(unittest.TestCase):
    def test_hybrid_merge_rejects_duplicates_and_unbounded_tail(self):
        base = [
            {"text": "Hello team", "timestamp": (0.0, 10.0)},
            {"text": "Closing discussion", "timestamp": (40.0, 50.0)},
        ]
        recovery = [
            {"text": "Hello team everyone", "timestamp": (1.0, 9.0)},
            {"text": "Important missing discussion", "timestamp": (20.0, 30.0)},
            {"text": "Hallucinated tail", "timestamp": (70.0, 80.0)},
        ]

        merged, recovered = _merge_transcription_passes(base, recovery)

        self.assertEqual(recovered, 1)
        self.assertEqual(
            [segment["text"] for segment in merged],
            ["Hello team", "Important missing discussion", "Closing discussion"],
        )

    def test_model_worker_subprocess_entry_point(self):
        with self.assertRaisesRegex(RuntimeError, "Unknown model worker stage"):
            run_model_worker("invalid-test-stage", {})

    def test_temporary_audio_cleanup(self):
        with temporary_audio_file(FakeUpload()) as path:
            self.assertTrue(os.path.exists(path))
        self.assertFalse(os.path.exists(path))

    def test_faster_whisper_returns_timestamped_segments(self):
        captured = {}

        class Segment:
            text = " Meeting started. "
            start = 0.25
            end = 1.5

        class Information:
            language = "en"
            language_probability = 0.99

        class FakeWhisper:
            def transcribe(self, audio, **kwargs):
                captured["audio"] = audio
                captured["kwargs"] = kwargs
                return iter([Segment()]), Information()

        with (
            tempfile.NamedTemporaryFile(suffix=".wav") as audio_file,
            patch(
                "modules.audio_transcription.load_whisper_model",
                return_value=FakeWhisper(),
            ),
        ):
            result = transcribe_audio(audio_file.name)

        self.assertTrue(captured["audio"].endswith(".wav"))
        self.assertTrue(captured["kwargs"]["vad_filter"])
        self.assertEqual(result["transcript"], "Meeting started.")
        self.assertEqual(result["segments"][0]["timestamp"], (0.25, 1.5))
        self.assertEqual(result["engine"], "faster-whisper/CTranslate2")

    def test_low_coverage_vad_output_is_retried_without_vad(self):
        calls: list[bool] = []

        class Segment:
            def __init__(self, text: str, start: float):
                self.text = text
                self.start = start
                self.end = start + 10.0

        class Information:
            language = "en"
            language_probability = 0.99
            duration = 600.0

        class FakeWhisper:
            def transcribe(self, _audio, **kwargs):
                calls.append(kwargs["vad_filter"])
                if kwargs["vad_filter"]:
                    segments = [Segment("one two three four five", 0.0)]
                else:
                    segments = [Segment("six seven eight nine ten", 20.0)]
                return iter(segments), Information()

        with (
            tempfile.NamedTemporaryFile(suffix=".wav") as audio_file,
            patch(
                "modules.audio_transcription.load_whisper_model",
                return_value=FakeWhisper(),
            ),
        ):
            result = transcribe_audio(audio_file.name)

        self.assertEqual(calls, [True, False])
        self.assertEqual(result["transcription_strategy"], "hybrid")
        self.assertEqual(result["coverage"]["word_count"], 10)
        self.assertEqual(len(result["coverage"]["attempts"]), 2)
        self.assertTrue(result["coverage"]["low_coverage"])

    def test_isolated_audio_orchestration_combines_worker_results(self):
        transcription = {
            "transcript": "Meeting started.",
            "segments": [{"text": "Meeting started.", "timestamp": [0.0, 1.0]}],
            "model": "openai/whisper-medium",
        }
        diarization = {
            "model": "pyannote/test",
            "speakers": ["SPEAKER_00"],
            "exclusive_turns": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
        }
        with (
            patch(
                "modules.isolated_inference.transcribe_audio_isolated",
                return_value=transcription,
            ),
            patch(
                "modules.isolated_inference.run_model_worker",
                return_value=diarization,
            ),
        ):
            result = transcribe_audio_with_speakers_isolated("meeting.wav", num_speakers=1)

        self.assertIn("SPEAKER_00: Meeting started.", result["transcript"])
        self.assertEqual(result["models"], ["openai/whisper-medium", "pyannote/test"])

    def test_diarization_failure_preserves_isolated_transcript(self):
        transcription = {
            "transcript": "Meeting started.",
            "segments": [{"text": "Meeting started.", "timestamp": [0.0, 1.0]}],
            "model": "openai/whisper-medium",
        }
        with (
            patch(
                "modules.isolated_inference.transcribe_audio_isolated",
                return_value=transcription,
            ),
            patch(
                "modules.isolated_inference.run_model_worker",
                side_effect=RuntimeError("403 gated model access"),
            ),
        ):
            result = transcribe_audio_with_speakers_isolated("meeting.wav")

        self.assertEqual(result["transcript"], "Meeting started.")
        self.assertEqual(result["plain_transcript"], "Meeting started.")
        self.assertIsNone(result["diarization"])
        self.assertIn("403", result["diarization_error"])


class EvaluationMetricTests(unittest.TestCase):
    def test_normalised_and_fuzzy_extraction(self):
        exact = extraction_metrics(["Submit budget report"], ["Submit budget report"])
        fuzzy = extraction_metrics(
            ["Submit the budget report"], ["Submit budget report"], fuzzy=True
        )
        self.assertEqual(exact["f1"], 1.0)
        self.assertGreater(fuzzy["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
