import unittest

import pandas as pd

from app.agents.execution_agent import _categorical_candidates, augment_training_data


class CategoricalAugmentationTests(unittest.TestCase):
    def test_numeric_range_category_is_inferred(self):
        values = pd.Series(["0-2 years", "3-5 years"])
        self.assertIn("6-8 years", _categorical_candidates(values))

    def test_composite_category_uses_observed_components(self):
        values = pd.Series(["Data Analyst", "Data Scientist", "Software Engineer", "Software Developer"])
        candidates = _categorical_candidates(values)
        self.assertIn("Data Engineer", candidates)
        self.assertNotIn("Data Astronaut", candidates)

    def test_new_categories_are_reported_and_original_rows_preserved(self):
        df = pd.DataFrame({
            "role": ["Data Analyst", "Data Scientist", "Software Engineer", "Software Developer"] * 10,
            "experience": ["0-2 years", "3-5 years", "0-2 years", "3-5 years"] * 10,
            "salary": [50000, 65000, 70000, 60000] * 10,
            "target": ["A", "A", "B", "B"] * 10,
        })
        out, metadata = augment_training_data(df, "target", {"strategy": "SimilarityOverSampler"})
        self.assertEqual(len(out), len(df) + metadata["generated_rows"])
        self.assertGreater(metadata["generated_rows"], 0)
        self.assertTrue(metadata["generated_new_categorical_values"])
        pd.testing.assert_frame_equal(
            out.iloc[:len(df)].drop(columns=["__autodatagen_source"]).reset_index(drop=True),
            df.reset_index(drop=True),
            check_dtype=False,
        )

    def test_new_category_generation_can_be_disabled(self):
        df = pd.DataFrame({
            "role": ["Data Analyst", "Data Scientist", "Software Engineer", "Software Developer"] * 10,
            "target": ["A", "A", "B", "B"] * 10,
        })
        _out, metadata = augment_training_data(
            df, "target", {"strategy": "SimilarityOverSampler", "allow_new_categories": False}
        )
        self.assertEqual(metadata["generated_new_categorical_values"], {})


if __name__ == "__main__":
    unittest.main()
