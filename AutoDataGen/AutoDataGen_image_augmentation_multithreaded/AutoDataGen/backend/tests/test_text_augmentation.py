import unittest
from unittest.mock import patch

import pandas as pd

from app.agents.execution_agent import _text_columns, augment_training_data
from app.agents.strategy_agent import choose_strategy


class TextAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "message": [
                "This message offers a wonderful prize and asks you to click this link now.",
                "Congratulations you have won a special reward please respond quickly.",
                "Please review the attached meeting notes before tomorrow morning.",
                "The team will discuss the quarterly project plan during the meeting.",
                "Claim your exclusive gift today by following the provided link.",
                "I will send the notes after the project meeting this afternoon.",
                "You have received a reward click the link to claim your prize.",
                "We should review the schedule and prepare the report for tomorrow.",
                "Limited offer available now please claim the gift using this link.",
                "The meeting agenda includes project updates and next week planning.",
            ],
            "source": ["sms"] * 10,
            "label": ["spam", "spam", "ham", "ham", "spam", "ham", "spam", "ham", "spam", "ham"],
        })

    def test_detects_natural_language_column(self):
        self.assertIn("message", _text_columns(self.df, "label"))
        self.assertNotIn("source", _text_columns(self.df, "label"))

    def test_ollama_text_variants_append_rows_and_preserve_labels(self):
        variants = [
            {"id": 0, "variant": "This note presents a fantastic prize and asks you to open this link now."},
            {"id": 1, "variant": "Well done, you won a special reward; please reply promptly."},
        ]
        with patch("app.agents.execution_agent.llm_enabled", return_value=True), patch(
            "app.agents.execution_agent.ask_json", return_value={"variants": variants}
        ):
            out, meta = augment_training_data(self.df, "label", {"strategy": "TextSynonymReplacement"})
        self.assertEqual(meta["generated_rows"], 2)
        self.assertEqual(len(out), len(self.df) + 2)
        self.assertEqual(out.iloc[len(self.df):]["label"].tolist(), ["spam", "spam"])
        self.assertEqual(out.iloc[len(self.df):]["source"].tolist(), ["sms", "sms"])
        self.assertTrue(out.iloc[len(self.df):]["__autodatagen_source"].str.startswith("synthetic_text_").all())

    def test_ollama_disabled_does_not_claim_generated_rows(self):
        with patch("app.agents.execution_agent.llm_enabled", return_value=False):
            out, meta = augment_training_data(self.df, "label", {"strategy": "TextBackTranslation"})
        self.assertEqual(meta["generated_rows"], 0)
        self.assertEqual(len(out), len(self.df))
        self.assertIn("Ollama disabled", meta["used_strategy"])

    def test_strategy_agent_selects_text_augmentation_for_text_dataset(self):
        data = pd.concat([self.df, self.df], ignore_index=True)
        with patch("app.agents.strategy_agent._enrich_plan", side_effect=lambda plan, analysis: plan):
            # Avoid relying on a running Ollama server; strategy is still deterministic when enabled.
            with patch("app.agents.ollama_agent.llm_enabled", return_value=True):
                plan = choose_strategy(data, "label", {"column_types": {"numeric": []}})
        self.assertEqual(plan["strategy"], "TextSynonymReplacement")
        self.assertIn("message", plan["text_features"])


if __name__ == "__main__":
    unittest.main()
