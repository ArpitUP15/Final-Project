import os
import unittest
from unittest.mock import patch

from app.agents.ollama_agent import ask_json, llm_enabled, ollama_status


class OllamaAgentTests(unittest.TestCase):
    def test_disabled_status_does_not_call_server(self):
        with patch.dict(os.environ, {"OLLAMA_ENABLED": "false"}, clear=False):
            status = ollama_status()
        self.assertFalse(status["enabled"])
        self.assertFalse(status["available"])
        self.assertIn("disabled", status["message"].lower())

    def test_disabled_llm_returns_none(self):
        with patch.dict(os.environ, {"OLLAMA_ENABLED": "false"}, clear=False):
            self.assertFalse(llm_enabled())
            self.assertIsNone(ask_json("system", {"example": 1}))

    def test_enabled_flag_is_read_from_environment(self):
        with patch.dict(os.environ, {"OLLAMA_ENABLED": "true"}, clear=False):
            self.assertTrue(llm_enabled())


if __name__ == "__main__":
    unittest.main()
