"""Single-column corpora must work from the dedicated text workspace."""
import io
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.database as database
import app.main as main


@pytest.mark.parametrize("suffix", ["csv", "xlsx"])
def test_short_single_column_corpus_uses_async_text_workflow(tmp_path, monkeypatch, suffix):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "text-runs.sqlite3")
    monkeypatch.setenv("OLLAMA_ENABLED", "false")
    frame = pd.DataFrame({"utterance": ["yes", "no", "yes", "no"]})
    buffer = io.BytesIO()
    if suffix == "csv":
        frame.to_csv(buffer, index=False)
    else:
        frame.to_excel(buffer, index=False, engine="openpyxl")

    with TestClient(main.app) as client:
        response = client.post(
            "/api/runs",
            files={"file": (f"corpus.{suffix}", buffer.getvalue())},
            data={"target_column": "utterance", "output_format": "original"},
        )
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        deadline = time.monotonic() + 5
        while True:
            result = client.get(f"/api/runs/{run_id}").json()
            if result["status"] in {"completed", "failed", "cancelled"}:
                break
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert result["status"] == "completed", result
        assert result["task_type"] == "text_augmentation"
        assert result["evaluation"] is None
        # With the local model disabled, zero generation must be reported honestly.
        assert result["augmentation"]["generated_rows"] == 0
        assert client.get(f"/api/runs/{run_id}/download").status_code == 200


def test_single_numeric_column_still_requires_table_features(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "table-runs.sqlite3")
    with TestClient(main.app) as client:
        response = client.post(
            "/api/runs",
            files={"file": ("values.csv", b"value\n1\n2\n3\n")},
            data={"target_column": "value"},
        )
        assert response.status_code == 400
