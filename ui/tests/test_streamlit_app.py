"""Exercise actual Streamlit reruns with mocked HTTP responses and no sockets."""

import copy
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from ui.api import ApiError
from ui.session import MAX_TURNS
from test_api import ANSWER, QUESTION

ROOT = Path(__file__).resolve().parents[2]


class StreamlitTests(unittest.TestCase):
    def setUp(self):
        for context in (patch.dict(os.environ, {"RAG_API_URL": "http://127.0.0.1:8001"}),
                        patch("socket.socket.connect", side_effect=AssertionError("Offline UI tests")),
                        patch("socket.socket.connect_ex", side_effect=AssertionError("Offline UI tests"))):
            context.start()
            self.addCleanup(context.stop)

    def app(self):
        app = AppTest.from_file(ROOT / "streamlit_app.py", default_timeout=10).run()
        self.assertEqual(len(app.exception), 0)
        return app

    def sign_in(self, app, username="bob"):
        with patch("ui.api.RagApi.login", return_value=username + "_PRIVATE_TOKEN") as login:
            app.text_input(key="login_username").set_value(username)
            app.text_input(key="login_password").set_value("private-password")
            app.button(key="FormSubmitter:login_form-Sign in").click().run()
            login.assert_called_once_with(username, "private-password")
        self.assertEqual(len(app.exception), 0)
        return app

    def ask(self, app, answer=None):
        with patch("ui.api.RagApi.ask", return_value=copy.deepcopy(answer or ANSWER)) as ask:
            app.chat_input(key="question_input").set_value(QUESTION).run()
            ask.assert_called_once_with(QUESTION, "bob_PRIVATE_TOKEN")
        self.assertEqual(len(app.exception), 0)

    def test_login_required_and_failed_login_keeps_chat_disabled(self):
        app = self.app()
        self.assertTrue(app.chat_input[0].disabled)
        with patch("ui.api.RagApi.login", side_effect=ApiError("Incorrect username or password.", 401)):
            app.text_input(key="login_username").set_value("bob")
            app.text_input(key="login_password").set_value("wrong-password")
            app.button(key="FormSubmitter:login_form-Sign in").click().run()
        self.assertTrue(app.chat_input[0].disabled)
        self.assertFalse(app.session_state.rag_session.token)
        self.assertEqual(app.text_input(key="login_password").value, "")

    def test_answer_citations_and_cache_route_are_displayed_without_token(self):
        app = self.sign_in(self.app())
        answer = {**ANSWER, "route": "cache", "cache_status": "hit"}
        self.ask(app, answer)
        self.assertEqual(len(app.chat_message), 2)
        self.assertIn("data/acme/salaries.txt · chunk 0", [t.value for t in app.text])
        self.assertTrue(any("Cached answer" in c.value for c in app.caption))
        displayed = str([element.value for name in ("text", "caption", "markdown", "error") for element in getattr(app, name)])
        self.assertNotIn("PRIVATE_TOKEN", displayed)
        self.assertNotIn("private-password", displayed)

    def test_graph_evidence_preserves_fact_and_citation(self):
        app = self.sign_in(self.app())
        answer = {**ANSWER, "route": "graph", "graph_evidence": [{"subject": "role:ceo", "relation": "annual_salary",
                  "target": "$240,000 per year", "source": "data/acme/salaries.txt", "chunk_id": 0,
                  "evidence": "The Acme CEO earns $240,000 per year."}]}
        self.ask(app, answer)
        self.assertIn("Graph evidence · 1 fact", [e.label for e in app.expander])
        self.assertIn("role:ceo → annual_salary → $240,000 per year", [e.value for e in app.text])
        self.assertIn(answer["graph_evidence"][0]["evidence"], [e.value for e in app.text])

    def test_logout_and_next_account_cannot_see_previous_answers(self):
        app = self.sign_in(self.app())
        self.ask(app)
        app.button(key="sign_out").click().run()
        self.assertFalse(app.session_state.rag_session.token)
        self.assertEqual(app.session_state.rag_session.turns, [])
        self.assertEqual(len(app.chat_message), 0)
        self.sign_in(app, "alice")
        self.assertEqual(len(app.chat_message), 0)
        self.assertEqual(app.session_state.rag_session.username, "alice")

    def test_401_clears_token_and_history_then_requests_login(self):
        app = self.sign_in(self.app())
        self.ask(app)
        with patch("ui.api.RagApi.ask", side_effect=ApiError("Your session expired or access was revoked. Sign in again.", 401)):
            app.chat_input(key="question_input").set_value(QUESTION).run()
        self.assertFalse(app.session_state.rag_session.token)
        self.assertEqual(app.session_state.rag_session.turns, [])
        self.assertEqual(len(app.chat_message), 0)
        self.assertTrue(app.chat_input[0].disabled)
        self.assertTrue(any("expired" in w.value for w in app.warning))

    def test_service_failure_does_not_add_an_answer_or_log_out(self):
        app = self.sign_in(self.app())
        with patch("ui.api.RagApi.ask", side_effect=ApiError("The service is busy or unavailable.", 503)):
            app.chat_input(key="question_input").set_value(QUESTION).run()
        self.assertEqual(app.session_state.rag_session.turns, [])
        self.assertTrue(app.session_state.rag_session.token)
        self.assertEqual(len(app.error), 1)

    def test_separate_browser_sessions_do_not_share_tokens_or_answers(self):
        first = self.sign_in(self.app())
        self.ask(first)
        second = self.app()
        self.assertFalse(second.session_state.rag_session.token)
        self.assertEqual(second.session_state.rag_session.turns, [])
        self.assertIsNot(first.session_state.rag_session, second.session_state.rag_session)

    def test_backend_change_invalidates_session_and_conversation(self):
        app = self.sign_in(self.app())
        self.ask(app)
        with patch.dict(os.environ, {"RAG_API_URL": "https://other.example.com"}):
            app.run()
        self.assertFalse(app.session_state.rag_session.token)
        self.assertEqual(len(app.chat_message), 0)

    def test_clear_conversation_retains_login_and_history_is_bounded(self):
        app = self.sign_in(self.app())
        self.ask(app)
        for index in range(MAX_TURNS + 2):
            app.session_state.rag_session.remember(str(index), copy.deepcopy(ANSWER), 1)
        app.run()
        self.assertEqual(len(app.session_state.rag_session.turns), MAX_TURNS)
        app.button(key="clear_chat").click().run()
        self.assertEqual(app.session_state.rag_session.turns, [])
        self.assertTrue(app.session_state.rag_session.token)


if __name__ == "__main__":
    unittest.main()
