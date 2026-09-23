import tempfile
import unittest
from pathlib import Path

from training.ami.build_dataset import (
    _tokens_for_href,
    join_tokens,
    read_abstractive,
    read_meeting_metadata,
    read_words,
)
from training.ami.evaluate_text_outputs import score_predictions

NITE_ROOT = 'xmlns:nite="http://nite.sourceforge.net/"'


class AMIDatasetTests(unittest.TestCase):
    def test_join_tokens_formats_punctuation(self):
        tokens = [
            {"text": "Hello", "punctuation": False},
            {"text": ",", "punctuation": True},
            {"text": "world", "punctuation": False},
            {"text": "!", "punctuation": True},
        ]
        self.assertEqual(join_tokens(tokens), "Hello, world!")

    def test_word_ranges_keep_nonlexical_ids_but_omit_their_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "words.xml"
            path.write_text(
                f"""<?xml version="1.0"?>
<nite:root {NITE_ROOT}>
  <w nite:id="m.A.words0" starttime="1" endtime="1.2">Good</w>
  <vocalsound nite:id="m.A.words1" starttime="1.2" endtime="1.3" type="laugh"/>
  <w nite:id="m.A.words2" starttime="1.3" endtime="1.5">morning</w>
  <w nite:id="m.A.words3" starttime="1.5" endtime="1.5" punc="true">.</w>
</nite:root>""",
                encoding="utf-8",
            )
            ordered, by_id = read_words(path)
            tokens = _tokens_for_href("words.xml#id(m.A.words0)..id(m.A.words3)", ordered, by_id)
            self.assertEqual(len(tokens), 4)
            self.assertEqual(join_tokens(tokens), "Good morning.")

    def test_official_seen_type_controls_split_and_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meetings.xml"
            path.write_text(
                f"""<?xml version="1.0"?>
<nite:root {NITE_ROOT}>
  <meeting nite:id="one" observation="ES2002a" duration="10" seen_type="training">
    <speaker nxt_agent="A" role="PM" global_name="speaker-1" channel="0"/>
  </meeting>
  <meeting nite:id="two" observation="ES2003a" duration="10" seen_type="development"/>
  <meeting nite:id="three" observation="ES2004a" duration="10"/>
</nite:root>""",
                encoding="utf-8",
            )
            metadata = read_meeting_metadata(path)
            self.assertEqual(metadata["ES2002a"]["split"], "train")
            self.assertEqual(metadata["ES2003a"]["split"], "validation")
            self.assertEqual(metadata["ES2004a"]["split"], "test")
            self.assertEqual(metadata["ES2002a"]["speakers"]["A"]["role"], "PM")

    def test_abstractive_sections_map_to_current_task(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.xml"
            path.write_text(
                f"""<?xml version="1.0"?>
<nite:root {NITE_ROOT}>
  <abstract><sentence> A short summary. </sentence></abstract>
  <actions><sentence>Pat will test it.</sentence></actions>
  <decisions><sentence>Use the blue case.</sentence></decisions>
  <problems><sentence>Battery life is unknown.</sentence></problems>
</nite:root>""",
                encoding="utf-8",
            )
            result = read_abstractive(path)
            self.assertEqual(result["abstract"], ["A short summary."])
            self.assertEqual(result["actions"], ["Pat will test it."])
            self.assertEqual(result["decisions"], ["Use the blue case."])
            self.assertEqual(result["problems"], ["Battery life is unknown."])

    def test_prediction_scorer_respects_split(self):
        meetings = [
            {
                "meeting_id": "ES2002a",
                "split": "train",
                "gold": {
                    "meeting_summary_sentences": ["Train"],
                    "decisions": [],
                    "actions": [],
                    "problems": [],
                },
            },
            {
                "meeting_id": "ES2003a",
                "split": "validation",
                "gold": {
                    "meeting_summary_sentences": ["Use blue."],
                    "decisions": ["Use blue."],
                    "actions": ["Pat will test."],
                    "problems": [],
                },
            },
        ]
        predictions = [
            {
                "meeting_id": "ES2003a",
                "prediction": {
                    "meeting_summary": "Use blue.",
                    "decisions": ["Use blue."],
                    "action_items": [{"task": "Pat will test."}],
                    "risks": [],
                },
            }
        ]
        report = score_predictions(meetings, predictions, "validation")
        self.assertEqual(report["expected_meetings"], 1)
        self.assertEqual(report["summary_similarity"], 1.0)
        self.assertEqual(report["category_macro"]["actions"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
