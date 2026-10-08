"""Cloud configuration must never fall back to a laptop's backend."""

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from ui.app import configured_url

ROOT = Path(__file__).resolve().parents[2]


class CloudTests(unittest.TestCase):
    def test_missing_cloud_backend_shows_setup_without_login_or_network(self):
        with patch.dict(os.environ, {}, clear=True), patch("socket.socket.connect", side_effect=AssertionError("Offline")):
            app = AppTest.from_file(ROOT / "ui/cloud_app.py").run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Backend setup required" in item.value for item in app.info))
        self.assertFalse(app.text_input)
        self.assertFalse(app.chat_input)

    def test_cloud_requires_public_https_and_local_entrypoint_keeps_local_default(self):
        for value in ("http://127.0.0.1:8001", "https://localhost", "https://localhost.",
                      "https://127.0.0.2", "https://[::1]", "https://10.0.0.1", "https://api.internal"):
            with self.subTest(value=value), patch.dict(os.environ, {"RAG_API_URL": value}), self.assertRaises(ValueError):
                configured_url(cloud=True)
        with patch.dict(os.environ, {"RAG_API_URL": "https://rag-api.onrender.com/"}):
            self.assertEqual(configured_url(cloud=True), "https://rag-api.onrender.com")
        with patch.dict(os.environ, {}, clear=True), patch("ui.app.st.secrets", {}):
            self.assertEqual(configured_url(), "http://127.0.0.1:8001")

    def test_cloud_entrypoint_reads_streamlit_secret_and_allows_login(self):
        with patch.dict(os.environ, {}, clear=True), patch("ui.app.st.secrets", {"RAG_API_URL": "https://rag-api.onrender.com"}), patch("socket.socket.connect", side_effect=AssertionError("Offline")):
            app = AppTest.from_file(ROOT / "ui/cloud_app.py").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state.rag_session.backend_url, "https://rag-api.onrender.com")
        self.assertEqual(len(app.text_input), 2)
        self.assertTrue(app.chat_input[0].disabled)
