"""Async API contract tests; stub LangGraph only when it is not installed in the test runner."""
import importlib.util
import sys
import time
import types

if importlib.util.find_spec("langgraph") is None:
    langgraph_module = types.ModuleType("langgraph")
    graph_module = types.ModuleType("langgraph.graph")
    class _Compiled:
        def invoke(self, state):
            return state
    class _StateGraph:
        def __init__(self, *_args, **_kwargs): pass
        def add_node(self, *_args, **_kwargs): pass
        def add_edge(self, *_args, **_kwargs): pass
        def compile(self): return _Compiled()
    graph_module.StateGraph = _StateGraph
    graph_module.START = "START"
    graph_module.END = "END"
    sys.modules["langgraph"] = langgraph_module
    sys.modules["langgraph.graph"] = graph_module

if importlib.util.find_spec("langchain_core") is None:
    lc_module = types.ModuleType("langchain_core")
    messages_module = types.ModuleType("langchain_core.messages")
    class _Message:
        def __init__(self, *args, **kwargs): pass
    messages_module.HumanMessage = _Message
    messages_module.SystemMessage = _Message
    sys.modules["langchain_core"] = lc_module
    sys.modules["langchain_core.messages"] = messages_module

from fastapi.testclient import TestClient
import app.database as database
import app.main as main


def test_pipeline_submit_returns_202_and_polls_to_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "runs.sqlite3")
    captured = {}
    def fake_workflow(df, target_column, run_id, filename, output_format, **kwargs):
        captured["max_iterations"] = kwargs["max_iterations"]
        kwargs["progress_callback"]({"stage": "optimization", "completed_iterations": 1,
                                      "planned_iterations": 1, "best_validation_score": 0.72})
        return {
            "run_id": run_id, "filename": filename, "target_column": target_column,
            "status": "completed", "dataset_bytes": b"a,b\n1,2\n", "download_suffix": ".csv",
            "iteration_search": {"completed_iterations": 1, "planned_iterations": 1},
        }
    monkeypatch.setattr(main, "run_workflow", fake_workflow)
    with TestClient(main.app) as client:
        response = client.post(
            "/api/runs",
            files={"file": ("tiny.csv", b"x,label\n1,a\n2,b\n3,a\n4,b\n5,a\n6,b\n", "text/csv")},
            data={"target_column": "label", "output_format": "csv", "max_iterations": "999"},
        )
        assert response.status_code == 202
        job = response.json()
        assert job["status"] == "queued"
        deadline = time.time() + 5
        result = None
        while time.time() < deadline:
            result_response = client.get(f"/api/runs/{job['run_id']}")
            assert result_response.status_code == 200
            result = result_response.json()
            if result["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.05)
        assert result["status"] == "completed"
        assert captured["max_iterations"] == 60
        assert client.get(f"/api/runs/{job['run_id']}/download").status_code == 200
