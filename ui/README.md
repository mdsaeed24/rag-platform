# Streamlit interface

The interface signs in through FastAPI and sends only a question and bearer
token to `/ask`. The backend derives tenant and role and enforces document
permissions. Streamlit never imports the embedding model, opens Qdrant, or calls
DeepSeek directly. It needs the backend address, not provider or JWT secrets.

## Run locally

From the project root, create a separate UI environment:

```sh
python3.14 -m venv .venv-ui
.venv-ui/bin/python -m pip install -r ui/requirements.txt
```

Start the backend in one terminal:

```sh
.venv/bin/python -m deploy.run serve
```

Start the interface in another:

```sh
.venv-ui/bin/python -m streamlit run streamlit_app.py
```

Open <http://127.0.0.1:8501>. Sign in with an existing demo account:

| Account | Password | Access |
| --- | --- | --- |
| alice | alice123 | Acme employee; salary requests abstain |
| bob | bob123 | Acme HR |
| carol | carol123 | Globex HR |

Try an annual-leave question, a CEO salary question, and a salary comparison.
Answers show their source/chunk citations and answer route. Graph answers also
show each relationship, supporting sentence, and citation. Repeating an
authorized question can show cache reuse. Submitting questions uses the
backend's configured provider and can incur API charges; signing in and checking
connection do not call the answer provider.

## Configuration and sessions

The default API address is `http://127.0.0.1:8001`. Override it with the
`RAG_API_URL` environment variable, or copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml` and edit the address. The environment variable takes
precedence. The UI does not load the backend's `.env`. Remote API addresses must
use HTTPS; local HTTP is allowed for localhost, 127.0.0.1, and ::1. Only operator
configuration selects this address, and redirects are never followed. Requests
have a five-second connection timeout and a 60-second HTTP I/O timeout, without
automatic retries; this is not a total wall-clock deadline.

Tokens and the last 20 question/answer pairs stay in the current Streamlit
session's memory. They are never stored in a shared Streamlit cache, URL, or
download. Sign-out, a new login, a backend-address change, and a 401 response
clear the token and conversation. Clear conversation retains the login. Password
form state clears after every login attempt. Closing/reloading a browser can
start a new session. Chat history is for display; each question is sent
independently without previous conversation context.

Historical displayed answers are snapshots; the backend checks current access
on each new request, and previously seen content cannot be remotely retracted.
The UI displays answers and evidence as plain text, with no embedded HTML or
remote images. API error bodies and tokens are not shown. Model and service
failures display a retry message and are not saved as answers.

## Validate

```sh
.venv-ui/bin/python -m unittest discover -s ui/tests -v
.venv-ui/bin/python -m pip --isolated --no-cache-dir check
```

The 17 tests cover HTTP payloads and bearer headers, redirect blocking, sanitized
errors, invalid response shapes, login, citations, graph evidence, session
isolation, account switching, token expiry, backend changes, and bounded history.
Streamlit AppTest exercises actual app reruns with mocked API responses and
blocked socket connections. These tests do not measure live answer quality.
Backend regression and deployment checks remain separate; run them with `.venv`
as described in the main README.

An additional nine-case integration smoke runs the actual UI HTTP client through
an in-process FastAPI transport, with real cached embeddings, temporary fictional
vectors, mocked completions, and blocked sockets:

```sh
.venv/bin/python -m ui.integration_smoke
```

This command uses the backend environment, never opens the workspace's vector
store, and writes `dist/ui-integration-smoke.json`. It covers all five answer
routes, employee/cross-tenant abstention, invalid-token handling, readiness, and
tenant isolation of the mocked outbound prompts. It requires no running server.

## Publishing later

This increment provides a local interface. See `GITHUB.md` for the public source
repository and automated checks. Cloud hosting remains a separate step.
A hosted Streamlit app needs an accessible hosted API; its
localhost is not your laptop. Validate dependencies and provision the model,
vector storage, and secrets on the selected backend host. The current native
deployment lock targets macOS ARM64. The checked-in Streamlit configuration binds
locally and disables usage telemetry; a hosting platform may need an address
override. Replace/restrict public demo credentials and apply public access and
usage controls before exposing a provider-backed app. Never commit
`.streamlit/secrets.toml`; it and `.venv-ui/` are gitignored.
