from __future__ import annotations

import unittest

from streamlit.testing.v1 import AppTest


def _result_fixture() -> dict:
    evidence = {
        "id": "seg-0001",
        "index": 0,
        "speaker": "Sarah",
        "text": "Sarah, please circulate the agenda by Wednesday.",
        "start": 4.5,
        "end": 8.0,
        "origin": "audio",
    }
    action = {
        "task": "Circulate the agenda",
        "owner": "Sarah",
        "deadline": "Wednesday",
        "evidence": evidence["text"],
        "evidence_segment_ids": [evidence["id"]],
        "evidence_start_seconds": evidence["start"],
        "evidence_end_seconds": evidence["end"],
        "evidence_status": "linked",
        "review_status": "draft",
    }
    context = {
        "meeting_summary": "The team reviewed the agenda.",
        "participants": [],
        "topics": ["Agenda"],
        "key_points": [],
        "decisions": [],
        "action_items": [action],
        "risks": [],
        "open_questions": [],
        "uncertainties": [],
        "quality": {},
    }
    return {
        "schema_version": "3.0",
        "meeting_context": context,
        "text_analysis": {"action_items": [dict(action)], "decisions": []},
        "source_segments": [evidence],
        "transcript": "Sarah: Sarah, please circulate the agenda by Wednesday.",
        "report_markdown": "# Meeting brief",
        "models": {},
        "timings": {},
        "cache": {},
    }


class AppUITests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file("app.py")
        self.app.session_state["meeting_result"] = _result_fixture()
        self.app.run(timeout=30)
        self.assertEqual(list(self.app.exception), [])

    def test_evidence_view_and_confirmation_are_interactive(self):
        view = next(button for button in self.app.button if button.label == "View transcript")
        view.click().run(timeout=30)
        self.assertEqual(list(self.app.exception), [])
        self.assertTrue(any("Evidence inspector" in heading.value for heading in self.app.markdown))

        confirm = next(button for button in self.app.button if button.label == "Confirm")
        confirm.click().run(timeout=30)
        self.assertEqual(list(self.app.exception), [])
        action = self.app.session_state["meeting_result"]["meeting_context"]["action_items"][0]
        self.assertEqual(action["review_status"], "confirmed")
        self.assertIn("verification_seconds", action)


if __name__ == "__main__":
    unittest.main()
