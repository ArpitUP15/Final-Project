import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.agents.orchestrator_agent import orchestrate_iterations


class ConstantClassifier:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class OrchestratorFallbackTests(unittest.TestCase):
    def test_rows_are_generated_even_when_no_candidate_improves_accuracy(self):
        # Balanced target plus a constant model ensures candidates cannot improve
        # validation accuracy; the dataset augmentation should still be produced.
        df = pd.DataFrame({
            "feature_a": np.arange(100, dtype=float),
            "feature_b": np.arange(100, dtype=float) % 7,
            "Label": np.array([0, 1] * 50),
        })
        with patch("app.agents.orchestrator_agent.make_model", return_value=ConstantClassifier()):
            result = orchestrate_iterations(
                df, "Label", {"strategy": "SimilarityOverSampler"},
                max_iterations=1,
            )

        self.assertTrue(result["iteration_search"]["baseline_fallback_selected"])
        self.assertEqual(result["augmentation"]["used_strategy"], "SimilarityOverSampler")
        self.assertGreater(result["augmentation"]["generated_rows"], 0)
        self.assertEqual(
            len(result["augmented_train"]),
            result["augmentation"]["original_training_rows"] + result["augmentation"]["generated_rows"],
        )


if __name__ == "__main__":
    unittest.main()
