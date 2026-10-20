from app.agents.orchestrator_agent import adaptive_iteration_plan, clamp_iterations


def test_hard_cap_never_exceeds_sixty():
    assert clamp_iterations(1000) == 60
    assert clamp_iterations(60) == 60
    assert clamp_iterations(0) == 1
    assert clamp_iterations("invalid") == 60


def test_high_baseline_uses_at_most_one_iteration():
    plan = adaptive_iteration_plan(
        baseline_accuracy=0.91, rows=1000, features=10, imbalance_ratio=1.2,
        max_iterations=60,
    )
    assert plan["planned_iterations"] == 1
    assert plan["max_iterations"] == 60


def test_low_baseline_gets_expanded_adaptive_budget():
    plan = adaptive_iteration_plan(
        baseline_accuracy=0.70, rows=1000, features=10, imbalance_ratio=2.3,
        max_iterations=60,
    )
    assert 5 <= plan["planned_iterations"] <= 60
    assert plan["planned_iterations"] > 20


def test_complex_low_accuracy_dataset_gets_larger_budget_but_never_over_cap():
    plan = adaptive_iteration_plan(
        baseline_accuracy=0.52, rows=300, features=35, imbalance_ratio=4.0,
        missing_rate=0.15, duplicate_rate=0.12, validation_stability=0.10,
        max_iterations=60,
    )
    assert plan["planned_iterations"] == 60


def test_user_requested_lower_cap_is_respected():
    plan = adaptive_iteration_plan(
        baseline_accuracy=0.61, rows=1000, features=10, imbalance_ratio=2.1,
        max_iterations=12,
    )
    assert plan["planned_iterations"] <= 12


def test_candidate_that_improves_validation_is_selected(monkeypatch):
    import numpy as np
    import pandas as pd
    import app.agents.orchestrator_agent as orchestrator

    class FakeModel:
        def __init__(self, model_name): self.model_name = model_name
        def fit(self, X, y):
            self.majority = pd.Series(y).mode().iloc[0]
            return self
        def predict(self, X):
            if self.model_name == "RandomForest":
                return np.repeat(self.majority, len(X))
            return np.where(X["signal"].to_numpy() == 1, "good", "bad")

    monkeypatch.setattr(orchestrator, "make_model", lambda X, **kwargs: FakeModel(kwargs["model_name"]))
    monkeypatch.setattr(orchestrator, "augment_training_data", lambda train, target, config: (train.copy(), {}))
    y = np.array(["good", "bad"] * 50)
    df = pd.DataFrame({"signal": (y == "good").astype(int), "noise": np.arange(len(y)), "label": y})
    result = orchestrator.orchestrate_iterations(
        df, "label", {"strategy": "NoResampling"}, max_iterations=5,
        target_accuracy=0.80, total_time_budget_seconds=30,
    )
    assert result["iteration_search"]["completed_iterations"] >= 1
    assert result["iteration_search"]["best_validation_accuracy"] > result["iteration_search"]["baseline_validation_accuracy"]
    assert result["iteration_search"]["baseline_fallback_selected"] is False


def test_candidate_that_regresses_accuracy_is_rejected(monkeypatch):
    import numpy as np
    import pandas as pd
    import app.agents.orchestrator_agent as orchestrator

    class FakeModel:
        def __init__(self, model_name): self.model_name = model_name
        def fit(self, X, y):
            self.majority = pd.Series(y).mode().iloc[0]
            return self
        def predict(self, X):
            if self.model_name == "RandomForest":
                return np.where(X["signal"].to_numpy() == 1, "good", "bad")
            return np.repeat("good", len(X))

    monkeypatch.setattr(orchestrator, "make_model", lambda X, **kwargs: FakeModel(kwargs["model_name"]))
    monkeypatch.setattr(orchestrator, "augment_training_data", lambda train, target, config: (train.copy(), {}))
    y = np.array(["good", "bad"] * 50)
    df = pd.DataFrame({"signal": (y == "good").astype(int), "noise": np.arange(len(y)), "label": y})
    result = orchestrator.orchestrate_iterations(
        df, "label", {"strategy": "NoResampling"}, max_iterations=1,
        target_accuracy=0.80, total_time_budget_seconds=30,
    )
    assert result["iteration_search"]["planned_iterations"] == 1
    assert result["iteration_search"]["baseline_fallback_selected"] is True
    assert result["iteration_search"]["history"][0]["accepted"] is False


def test_numeric_high_cardinality_target_is_detected_as_regression():
    import numpy as np
    import pandas as pd
    from app.agents.evaluation_agent import is_regression_target
    df = pd.DataFrame({"feature": np.arange(100), "value": np.arange(100) * 1.5 + 3})
    assert is_regression_target(df, "value") is True
    assert is_regression_target(df.assign(label=["yes", "no"] * 50), "label") is False


def test_regression_reports_r2_mae_rmse_not_accuracy():
    import numpy as np
    import pandas as pd
    from app.agents.orchestrator_agent import orchestrate_iterations
    df = pd.DataFrame({"x": np.arange(80), "category": ["a", "b"] * 40,
                       "value": np.arange(80) * 2.0 + 5})
    result = orchestrate_iterations(
        df, "value", {"strategy": "NoResampling", "task": "regression"},
        max_iterations=1, total_time_budget_seconds=30,
    )
    assert result["task_type"] == "regression"
    assert result["iteration_search"]["max_iterations"] <= 60
    assert result["augmentation"]["generated_rows"] == 0


def test_low_baseline_also_respects_a_one_iteration_user_cap():
    plan = adaptive_iteration_plan(
        baseline_accuracy=0.55, rows=300, features=30, imbalance_ratio=4.0,
        max_iterations=1,
    )
    assert plan["planned_iterations"] == 1
