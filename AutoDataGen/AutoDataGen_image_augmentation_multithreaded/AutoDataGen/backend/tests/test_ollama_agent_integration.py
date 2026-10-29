import pandas as pd
from unittest.mock import patch

from app.agents.analysis_agent import analyze_dataframe
from app.agents.strategy_agent import choose_strategy


def test_analysis_returns_deterministic_profile_when_ollama_unavailable():
    df = pd.DataFrame({"feature": [1, 2, 3, 4] * 10, "label": ["a", "b"] * 20})
    with patch("app.agents.ollama_agent.ask_json", return_value=None):
        result = analyze_dataframe(df)
    assert result["shape"] == {"rows": 40, "columns": 2}
    assert result["llm_insights"]["source"] == "fallback"


def test_strategy_rejects_unsupported_ollama_recommendation():
    df = pd.DataFrame({"feature": [1, 2, 3, 4] * 10, "label": ["a", "b"] * 20})
    analysis = analyze_dataframe(df)
    with patch("app.agents.strategy_agent.ask_json", return_value={
        "preferred_strategy": "ExecuteArbitraryCode", "rationale": "invalid",
        "augmentation_cautions": [],
    }):
        plan = choose_strategy(df, "label", analysis)
    assert plan["strategy"] == "NoResampling"
    assert plan["llm_status"] == "used"
    assert "llm_preferred_strategy" not in plan


def test_strategy_accepts_only_an_allowlisted_ollama_recommendation():
    df = pd.DataFrame({"feature": [1, 2, 3, 4] * 10, "label": ["a", "b"] * 20})
    analysis = analyze_dataframe(df)
    with patch("app.agents.strategy_agent.ask_json", return_value={
        "preferred_strategy": "SimilarityOverSampler", "rationale": "Candidate for mixed tabular data.",
        "augmentation_cautions": ["Validate generated categories."],
    }):
        plan = choose_strategy(df, "label", analysis)
    assert plan["strategy"] == "SimilarityOverSampler"
    assert plan["llm_status"] == "used"
