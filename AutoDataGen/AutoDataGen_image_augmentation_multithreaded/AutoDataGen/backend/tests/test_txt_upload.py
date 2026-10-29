import unittest
from unittest.mock import patch

import pandas as pd

from app.main import read_dataset
from app.workflow.graph import _run_text_only_augmentation, run_workflow


class TxtUploadTests(unittest.TestCase):
    def test_txt_reader_creates_one_text_column_and_skips_blank_lines(self):
        df = read_dataset("sentences.txt", b"First sentence.\r\n\r\nSecond sentence.\n")
        self.assertEqual(list(df.columns), ["text"])
        self.assertEqual(df["text"].tolist(), ["First sentence.", "Second sentence."])

    def test_txt_reader_rejects_empty_file(self):
        with self.assertRaises(Exception):
            read_dataset("empty.txt", b"\n  \n")

    def test_txt_output_is_plain_lines_without_csv_header(self):
        original = pd.DataFrame({"text": ["First sentence.", "Second sentence."]})
        augmented = pd.DataFrame({"text": ["First sentence.", "Second sentence.", "A paraphrase."]})
        metadata = {"original_rows": 2, "generated_rows": 1, "augmented_rows": 3,
                    "method": "TextBackTranslation", "text_columns": ["text"],
                    "generated_text_examples": [], "note": "test"}
        with patch("app.workflow.graph.augment_unlabeled_text_column", return_value=(augmented, metadata)):
            result = _run_text_only_augmentation(original, "text", "test-run", "sentences.txt", "original")
        self.assertEqual(result["download_suffix"], ".txt")
        self.assertEqual(result["dataset_bytes"].decode("utf-8"),
                         "First sentence.\nSecond sentence.\nA paraphrase.\n")
        self.assertIsNone(result["evaluation"])

    def test_txt_always_uses_text_only_route_even_for_short_lines(self):
        df = pd.DataFrame({"text": ["yes", "no"]})
        with patch("app.workflow.graph._run_text_only_augmentation", return_value={"task_type": "text_augmentation"}) as run_text:
            result = run_workflow(df, "text", "test-run", "tiny.txt")
        run_text.assert_called_once()
        self.assertEqual(result["task_type"], "text_augmentation")


if __name__ == "__main__":
    unittest.main()
