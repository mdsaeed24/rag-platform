"""Run with: .venv-ui/bin/python -m streamlit run streamlit_app.py"""

import os
import ipaddress
from urllib.parse import urlsplit
from time import perf_counter

import streamlit as st

from ui.api import ApiError, RagApi, backend_url
from ui.session import session_for


def configured_url(*, cloud=False):
    default = None if cloud else "http://127.0.0.1:8001"
    value = os.environ.get("RAG_API_URL")
    if value is None:
        try:
            value = st.secrets.get("RAG_API_URL", default)
        except FileNotFoundError:
            value = default
    value = backend_url(value)
    if cloud:
        parts = urlsplit(value)
        hostname = parts.hostname.rstrip(".").lower()
        try:
            public = ipaddress.ip_address(hostname).is_global
        except ValueError:
            public = "." in hostname and not hostname.endswith((".localhost", ".local", ".internal"))
        if parts.scheme != "https" or not public:
            raise ValueError("A public HTTPS backend is required")
    return value


def show_answer(turn):
    answer = turn["response"]
    with st.chat_message("user"):
        st.text(turn["question"])
    with st.chat_message("assistant"):
        # Plain text avoids rendering remote images or HTML supplied in answers.
        st.text(answer["answer"])
        labels = {"rag": "Retrieved answer", "cache": "Cached answer", "graph": "Graph answer",
                  "graph_cache": "Cached graph answer", "abstain": "Insufficient authorized evidence"}
        st.caption(f"{labels[answer['route']]} · {turn['elapsed_ms'] / 1000:.2f}s")
        st.caption(f"Tenant: {answer['tenant_id']} · Role: {answer['role']} · Cache: {answer['cache_status']}")
        if answer["sources"]:
            with st.expander(f"Sources · {len(answer['sources'])}"):
                for source in answer["sources"]:
                    st.text(f"{source['source']} · chunk {source['chunk_id']}")
        if answer["graph_evidence"]:
            count = len(answer["graph_evidence"])
            with st.expander(f"Graph evidence · {count} {'fact' if count == 1 else 'facts'}"):
                for edge in answer["graph_evidence"]:
                    st.text(f"{edge['subject']} → {edge['relation']} → {edge['target']}")
                    st.text(edge["evidence"])
                    st.caption(f"{edge['source']} · chunk {edge['chunk_id']}")


def main(*, cloud=False):
    st.set_page_config(page_title="RAG Platform", page_icon="📚", layout="wide")
    st.title("Company knowledge, with evidence")
    st.caption("Ask about company policies and compensation. Answers follow your account's access permissions.")
    try:
        api = RagApi(configured_url(cloud=cloud))
    except ValueError:
        # Clear any previous identity even when new operator configuration is invalid.
        if "rag_session" in st.session_state:
            st.session_state.rag_session.clear()
        if cloud:
            st.info("Backend setup required. Configure RAG_API_URL in the app settings with your hosted FastAPI HTTPS address.")
        else:
            st.error("The backend address is not configured correctly. Set RAG_API_URL to HTTPS, or localhost for local use.")
        st.stop()
    session = session_for(st.session_state, api.url)
    # Explicitly remove password widget state before recreating the form.
    # clear_on_submit also resets the browser form, but is insufficient alone
    # for server-side widget state immediately after a failed login.
    if st.session_state.pop("clear_login_fields", False):
        st.session_state.login_username = ""
        st.session_state.login_password = ""
    with st.sidebar:
        st.header("RAG Platform")
        if st.button("Check connection", key="check_connection"):
            try:
                if api.ready():
                    st.success("Ready to answer")
                else:
                    st.warning("The backend is running but its model or storage is not ready.")
            except ApiError as exc:
                st.error(str(exc))
        if session.token:
            st.text("Signed in as " + session.username)
            if st.button("Sign out", key="sign_out"):
                session.clear()
                st.rerun()
            if st.button("Clear conversation", key="clear_chat"):
                session.turns.clear()
                st.rerun()
            st.caption("This session keeps the last 20 questions. Each question is answered independently.")
        else:
            st.subheader("Sign in")
            with st.form("login_form", clear_on_submit=True):
                username = st.text_input("Username", max_chars=128, key="login_username")
                password = st.text_input("Password", type="password", max_chars=1024, key="login_password")
                submitted = st.form_submit_button("Sign in", type="primary")
            login_error = st.session_state.pop("login_error", None)
            if login_error:
                st.error(login_error)
            if submitted:
                st.session_state.clear_login_fields = True
                if not username.strip() or not password:
                    st.session_state.login_error = "Enter your username and password."
                else:
                    try:
                        token = api.login(username.strip(), password)
                    except ApiError as exc:
                        session.clear()
                        st.session_state.login_error = str(exc)
                    else:
                        session.sign_in(username.strip(), token)
                st.rerun()
        with st.expander("Demo accounts"):
            st.text("alice / alice123 — Acme employee\nbob / bob123 — Acme HR\ncarol / carol123 — Globex HR")
            st.caption("Fictional demo data. Salary access is restricted to HR accounts.")

    notice = st.session_state.pop("auth_notice", None)
    if notice:
        st.warning(notice)
    if not session.token:
        st.info("Sign in using the sidebar to start asking questions.")
        st.chat_input("Sign in to ask a question", disabled=True)
        return
    if not session.turns:
        st.subheader("Start with a question")
        examples = ("How many days of annual leave do employees receive?", "What does the CEO earn?",
                    "Compare CEO and software engineer salaries.")
        for column, question in zip(st.columns(3), examples):
            with column.container(border=True):
                st.text(question)
        st.caption("Repeat an authorized question to see cache reuse. Graph questions show the facts supporting the answer.")
    for turn in session.turns:
        show_answer(turn)
    question = st.chat_input("Ask a company question", max_chars=4000, key="question_input")
    if question and question.strip():
        question = question.strip()
        start = perf_counter()
        try:
            with st.spinner("Finding an answer in your authorized documents…"):
                answer = api.ask(question, session.token)
        except ApiError as exc:
            if exc.status == 401:
                session.clear()
                st.session_state.auth_notice = str(exc)
                st.rerun()
            st.error(str(exc))
        else:
            session.remember(question, answer, (perf_counter() - start) * 1000)
            st.rerun()
