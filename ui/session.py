"""One browser session's token and bounded, in-memory conversation."""

from dataclasses import dataclass, field

MAX_TURNS = 20


@dataclass
class ChatSession:
    backend_url: str
    token: str = field(default="", repr=False)
    username: str = ""
    turns: list = field(default_factory=list, repr=False)

    def clear(self):
        self.token = ""
        self.username = ""
        self.turns.clear()

    def sign_in(self, username, token):
        self.clear()
        self.username, self.token = username, token

    def remember(self, question, response, elapsed_ms):
        self.turns.append({"question": question, "response": response, "elapsed_ms": elapsed_ms})
        del self.turns[:-MAX_TURNS]


def session_for(state, url):
    session = state.get("rag_session")
    if session is None or session.backend_url != url:
        if session is not None:
            session.clear()
        session = state["rag_session"] = ChatSession(url)
    return session
