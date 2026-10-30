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
    def test_unaugmented_baseline_keeps_test_accuracy_unchanged(self):
        from app.agents.evaluation_agent import evaluate, split_original

        labels = np.array([0, 1] * 50)
        df = pd.DataFrame({
            "signal": labels,
            "feature_a": np.arange(100, dtype=float),
            "Label": labels,
        })
        X_train, _, y_train, _ = split_original(df, "Label")
        train = X_train.copy()
        train["Label"] = y_train.to_numpy()
        train["__autodatagen_source"] = "original"

        evaluation = evaluate(
            df, "Label", train, n_estimators=120, model_name="RandomForest"
        )

        self.assertEqual(
            evaluation["baseline"]["accuracy"],
            evaluation["augmented"]["accuracy"],
        )

    def test_unaugmented_baseline_is_kept_when_no_candidate_passes_accuracy_guard(self):
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
        self.assertEqual(result["augmentation"]["generated_rows"], 0)
        self.assertEqual(result["strategy"]["strategy"], "NoResampling")
        self.assertEqual(
            len(result["augmented_train"]),
            result["augmentation"]["original_training_rows"],
        )

    def test_candidate_must_not_regress_accuracy_on_any_validation_split(self):
        class SplitAwareClassifier:
            def __init__(self, model_name):
                self.model_name = model_name

            def fit(self, X, y):
                return self

            def predict(self, X):
                predictions = X["truth"].to_numpy().copy()
                split = int(X["split"].iloc[0])
                if self.model_name == "RandomForest":
                    predictions[:2] = 1 - predictions[:2]
                elif split == 0:
                    predictions[:1] = 1 - predictions[:1]
                elif split == 1:
                    predictions[:3] = 1 - predictions[:3]
                return predictions

        df = pd.DataFrame({
            "feature_a": np.arange(100, dtype=float),
            "Label": np.array([0, 1] * 50),
        })
        splits = []
        for split in range(3):
            y_val = pd.Series(np.array([0, 1] * 5))
            X_val = pd.DataFrame({
                "feature_a": np.arange(10, dtype=float),
                "split": [split] * 10,
                "truth": y_val.to_numpy(),
            })
            X_fit = pd.DataFrame({
                "feature_a": np.arange(20, dtype=float),
                "split": [split] * 20,
                "truth": np.array([0, 1] * 10),
            })
            y_fit = pd.Series(np.array([0, 1] * 10))
            splits.append((X_fit, X_val, y_fit, y_val))

        with patch("app.agents.orchestrator_agent._validation_splits", return_value=splits), \
             patch("app.agents.orchestrator_agent.make_model",
                   side_effect=lambda X, **kwargs: SplitAwareClassifier(kwargs["model_name"])), \
             patch("app.agents.orchestrator_agent.augment_training_data",
                   side_effect=lambda train, target, config: (train.copy(), {})):
            result = orchestrate_iterations(
                df, "Label", {"strategy": "NoResampling"},
                max_iterations=1,
            )

        candidate = result["iteration_search"]["history"][0]
        self.assertEqual(candidate["validation_splits"], 3)
        self.assertGreater(candidate["validation_accuracy"], candidate["baseline_validation_accuracy"])
        self.assertFalse(candidate["accuracy_preserved"])
        self.assertEqual(candidate["status"], "rejected_accuracy_regression")
        self.assertTrue(result["iteration_search"]["baseline_fallback_selected"])
        self.assertEqual(result["strategy"]["strategy"], "NoResampling")


if __name__ == "__main__":
    unittest.main()
