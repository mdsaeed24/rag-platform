# Permission-aware RAG platform with caching, routing, and GraphRAG

Project 2 adds authenticated tenant and role isolation to the existing FastAPI,
MiniLM (384 dimensions), local Qdrant, and DeepSeek RAG pipeline. Documents and
the existing vector collection were retained; ingestion was not rerun during
this security work. Project 3 adds permission-aware answer caching and explicit
routing between cached answers, fresh RAG generation, and abstention. Project 4
adds evidence-backed graph paths for salary/policy joins and role comparisons.

## Run locally

Use the existing `.venv`, or install `requirements.txt` in a new virtual
environment. Configure `DEEPSEEK_API_KEY` and a random `JWT_SECRET_KEY` in `.env`
using `.env.example` as a guide. The application fails startup if the JWT secret
is missing or too short. The workspace's existing DeepSeek setting was preserved
and a random JWT secret was added without displaying it.

```sh
.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8001
```

Use <http://127.0.0.1:8001/docs>. Port 8000 was occupied by an unrelated application
when this workspace was inspected. Keep one worker with this local Qdrant setup;
separate processes cannot concurrently own its persistent directory. Queries
within one process serialize storage access. Stop the RAG server before ingestion
or a separate process that accesses the same collection.

`POST /login` accepts JSON with `username` and `password`. Demo credentials remain
`alice / alice123` (Acme employee), `bob / bob123` (Acme HR), and
`carol / carol123` (Globex HR). Passwords are stored as salted PBKDF2-SHA256 hashes.
These are public demo credentials; replace them before exposing the service.

`POST /ask` accepts only `{"question": "What does the CEO earn?"}` and an
`Authorization: Bearer <access_token>` header. Login again after this update:
the signing key changed and new tokens require a `token_version` claim.

| User | Permitted model context for the salary question |
| --- | --- |
| Bob | Acme handbook and salary chunks, including the $240,000 CEO salary |
| Alice | Acme handbook retrieval only; the router abstains before generation |
| Carol | Globex handbook and salary chunks, including the $310,000 CEO salary |

The authorization tests inspect the actual context passed to the generator.
DeepSeek generation is mocked in these tests, so they do not establish live
answer correctness or live RAG latency.

A separate live smoke test passed all three authenticated `/ask` cases against
the existing vectors and real DeepSeek calls: Bob received the Acme CEO salary
of $240,000, Alice received the standard unknown answer with no salary context,
and Carol received the Globex CEO salary of $310,000. Outbound prompts were
checked before each call for tenant and role isolation. The HR answers included
the correct salary source and chunk 0. Details are saved in
`eval/live_authenticated_checks.json`; this small smoke test is separate from
the full answer evaluation and does not prove absence of every possible leak.

## Streamlit interface

A local Streamlit interface provides login, chat, source/chunk citations, answer
routes, cache status, and graph facts with supporting evidence. It calls the
existing FastAPI endpoints; identity and authorization remain in the backend.
The UI uses a separate environment so the backend dependency lock stays intact.

```sh
python3.14 -m venv .venv-ui
.venv-ui/bin/python -m pip install -r ui/requirements.txt
# In one terminal:
.venv/bin/python -m deploy.run serve
# In another terminal:
.venv-ui/bin/python -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open <http://127.0.0.1:8501>. See [the interface guide](ui/README.md) for demo
accounts, API configuration, session behavior, and the 20 offline interface tests.
Submitting a question uses the configured backend provider. See [the GitHub
guide](GITHUB.md) for the public source repository, fresh-clone setup, automated
checks, and which files remain local. See [cloud deployment](CLOUD.md) for the prepared free demo backend and Streamlit Community Cloud setup. Account setup and a successful hosted deployment are still required.

## Native deployment package

Deployment tools now include a 72-package dependency lock for the current
Python 3.14/macOS ARM64 environment, a preflight check, a single-worker launcher,
an isolated offline smoke test, and an allowlisted source release archive.
Secrets, documents, vectors, model caches, logs, and raw evaluation reports are
provisioned separately and are excluded from the archive.

```sh
.venv/bin/python -m deploy.run check
.venv/bin/python -m deploy.smoke
.venv/bin/python -m deploy.package
.venv/bin/python -m deploy.run serve
```

The launcher binds to `127.0.0.1:8001`, forces one worker without reload, and
holds a process-lifetime lock to reject a second launcher in the same directory.
It requires the existing cached model and initialized vectors; startup never
downloads the model or reingests documents. Preflight and smoke make no real
provider calls. The smoke uses temporary fictional vectors and mocked answers.

See [the deployment runbook](deploy/README.md) for installation, runtime assets,
lock updates, shutdown, and release replacement. A clean Python 3.14.6/macOS
ARM64 virtual environment successfully downloaded and installed all 72 pinned
packages, passed `pip check`, all 193 regression tests and the full offline plan,
and passed all 11 smoke checks from the extracted release. The existing model
cache was reused; another host, model provisioning, and containers remain
untested. No public service is published. Installation details and wheel digests
are recorded in `eval/fresh_install_results.json` in the development workspace.

## Final offline verification

[FINAL_RESULTS.md](FINAL_RESULTS.md) records the latest full validation and the
remaining work. The current run passed 201 regression tests, all 84 cases across
the five existing evaluation suites, all 11 deployment smoke cases, and 300
measured requests across ten pipeline scenarios. Stored payload fingerprints
were unchanged. No real DeepSeek calls were made.

From the full development workspace, stop any process sharing `qdrant_data/` and
run the fixed offline plan:

```sh
.venv/bin/python -m eval.final_validation --iterations 30 --acl-iterations 200
```

The runner uses dummy provider credentials and blocks socket connections. It
runs storage-owning phases sequentially, rejects incomplete/failed/live reports,
checks that code inputs and stored payloads remain unchanged, and writes
`eval/final_results.json` plus `FINAL_RESULTS.md`. Detailed fresh reports and
private logs are under `dist/final-validation/`. Earlier baseline/live reports
remain intact. The source deployment archive includes only the aggregate
Markdown summary, not evaluation tools, tests, or raw reports.

The pipeline benchmark uses real embeddings, persisted vectors, and mocked
completions. It measures in-process work, excluding HTTP transport, login,
startup, cache priming, and warmups; audit writes are mocked. Its median/p95
values describe this four-point local corpus. Cache hits avoid mocked generation
while still retrieving and checking permissions. These results establish neither
live answer accuracy nor production latency, load capacity, or cost savings.
Live GraphRAG answer checks still require explicit payload/destination approval.

## Security behavior

Identity comes from an HS256 JWT with a required expiration, user ID, tenant,
role, and token version. Claims must match the current server-side user record.
Body identity fields are rejected with HTTP 422. Missing, forged, expired,
incomplete, or revoked credentials return HTTP 401 with `WWW-Authenticate: Bearer`.

Qdrant requires both tenant equality and role membership during retrieval.
Before generation, every retrieved chunk is checked against `DOCUMENT_ACLS`,
including source, tenant, allowed roles, classification, chunk ID, and content
shape. A failed check rejects the whole batch with HTTP 403 and skips generation.
`generate_answer()` also requires tenant/role keyword arguments and repeats the
check immediately before constructing context, protecting direct callers.
An empty retrieval returns the standard unknown answer without an LLM call.

For user revocation, disable `active`, change the user's tenant/role, or increment
`token_version` in the server-side record. Existing tokens are rechecked on each
request and again after retrieval. Removing or restricting a document's ACL
blocks stale vector metadata before generation. When editing the source files,
restart the process to load the new configuration. Reingest after changing ACLs
to synchronize stored payloads; until then, mismatches fail closed.

The user directory and ACLs are deliberately still local configuration. Runtime
changes exist only within that process; durable revocation across restarts and
multiple services requires shared persistent identity/policy storage. Requests
already sent to DeepSeek cannot be recalled by subsequent revocation.

Audit events go to `logs/audit.jsonl`, rotating at 5 MB with three backups. They
contain UTC timestamps, verified identity when available, query, retrieved point
and source/chunk IDs, decision, reason, and retrieval/validation timings. They
exclude bearer tokens, passwords, API keys, and chunk text. Queries themselves
may be sensitive; keep these local logs private. This is a local audit trail,
not a tamper-resistant production logging service.

`.env`, `.venv`, `qdrant_data`, logs, and Python caches are gitignored. No Git
repository was present at inspection, and no commits were created.

## Health, admission limits, and metrics

`GET /health/live` returns `200 {"status":"alive"}` when the application can
respond. `GET /health/ready` returns `200 {"status":"ready"}` only when the
loaded embedding model reports 384 dimensions and the existing local Qdrant
collection is accessible, nonempty, and configured for 384-dimensional cosine
vectors. Otherwise it returns `503 {"status":"not_ready"}` without storage
details. Both endpoints are public and make no DeepSeek calls. Readiness reads
collection metadata, never document content, and does not create missing storage.
It waits at most one second for this process's storage lock; a busy or externally
locked collection can temporarily fail readiness. This is local readiness,
not proof of provider availability, embedding quality, or correct document ACLs.

These settings are loaded from `.env` at startup:

| Setting | Default | Behavior |
| --- | --- | --- |
| `MAX_REQUEST_BODY_BYTES` | `16384` | Maximum actual streamed HTTP body bytes; larger requests return 413 |
| `MAX_INFLIGHT_ASK` | `4` | Maximum simultaneous `POST /ask` requests |
| `MAX_INFLIGHT_LOGIN` | `4` | Separate maximum simultaneous `POST /login` requests |
| `METRICS_TOKEN` | Empty | Disables `/metrics` with 404 until configured |

Limits must be positive integers. Body size is enforced before JSON parsing,
including chunked bodies and misleading `Content-Length` headers. The concurrency
caps reject excess requests immediately with HTTP 503 and `Retry-After: 1`;
there is no admission queue. Slots cover body receipt through response completion
and are released after errors or early disconnects. These checks run before JWT
validation, so oversized requests or requests exceeding capacity can receive
413/503 before authentication. Accepted requests retain all existing JWT,
retrieval, ACL, and cache checks. Health checks and metrics have no ask/login
admission slot requirement.

To enable `GET /metrics`, set an independent random `METRICS_TOKEN` of at least
32 characters without surrounding whitespace, then restart. Generate it with
`python -c 'import secrets; print(secrets.token_urlsafe(48))'` and keep it private.
Send it as `Authorization: Bearer <METRICS_TOKEN>` to the metrics endpoint.
Application login JWTs do not grant metrics access, and the metrics token does
not grant document access. Missing or incorrect metrics credentials return 401.

The endpoint exports Prometheus text format with:

- `rag_http_requests_total`, labeled by fixed endpoint and status code.
- `rag_http_request_duration_seconds_sum` and `_count`, for average latency.
- `rag_inflight_requests` and `rag_inflight_limit`, for ask/login capacity.
- `rag_answers_total`, counting completed answers by `rag`, `graph`, `cache`,
  `graph_cache`, or `abstain` after final authorization checks.

Metrics contain no queries, document content, source names, identities, tokens,
or provider error bodies. Unknown URL paths share one `other` label, keeping
metric cardinality bounded. HTTP timings include body receipt and application
processing; they are not model-only latency. Failures are counted by HTTP status
without incrementing completed-answer counts. Metrics and limits are process
local and reset on restart. No metrics backend or dashboard is deployed.

The caps bound simultaneous work, not requests per minute, aggregate LLM spend,
or slow-client duration. A production reverse proxy still needs body-read
timeouts and rate limits. The local Qdrant setup still requires one worker.

Offline tests cover concurrency rejection/recovery, independent login capacity,
streamed body limits, disconnects, metrics authentication/privacy, all answer
routes, failure counts, and temporary Qdrant readiness checks:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python -m unittest discover -s tests -p 'test_operations.py' -v
```

## Answer provider failure handling

Ordinary RAG and GraphRAG use `LLM_TIMEOUT_SECONDS=30` by default. Configure a
finite positive number in `.env` and restart the server to change it. Invalid
values fail startup. SDK automatic retries are disabled, so each uncached
generation attempt makes at most one provider request. This is an HTTP I/O
timeout for connection/read/write/pool operations, not an overall `/ask`
deadline; embedding, retrieval, and slow responses that keep delivering data
can take additional time.

| Provider failure | `/ask` response |
| --- | --- |
| I/O timeout | 504, `Answer provider timed out` |
| Connection failure, rate limit, or provider HTTP 5xx | 503, `Answer provider temporarily unavailable` |
| Other provider HTTP errors, including provider authentication failure | 502, `Answer provider could not complete the request` |
| Invalid schema, empty content, or incomplete/filtered completion | 502, `Answer provider returned an invalid response` |

These errors return only a fixed `detail`, without a partial answer, evidence,
provider response body, or credentials. They do not become unknown answers or
cache entries. A later successful request can generate and cache normally.
Existing authorized cache hits and router abstentions still skip the provider.
JWT and document authorization errors retain their existing 401/403 behavior.
Failures in local retrieval and unexpected programming errors are outside this
provider error handling.

Provider failures add an audit event with `authorization_decision=error`, a
fixed generation reason, verified identity, query, document IDs, route, cache
status, and elapsed generation time. The earlier context authorization event
does not indicate successful generation. Provider exception messages and bodies
are excluded from the audit trail.

The full offline suite has 201 passing tests. Provider failure tests cover both
generation routes, recovery, cache exclusion, malformed/truncated completions,
and an in-memory SDK transport that verifies
the configured timeout and a single attempt. No DeepSeek calls are needed:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python -m unittest discover -s tests -p 'test_generation.py' -v
```

Existing offline evaluations also passed after this change: GraphRAG 16/16,
graph caching 14/14, ordinary caching 15/15, routing 17/17, and adversarial
authorization 22/22. They use existing vectors and mocked completions; no live
provider reliability, latency, or answer quality is established by these runs.

## Semantic answer caching: Project 3, first increment

Caching is enabled by default. Set `ANSWER_CACHE_ENABLED=false` in `.env` and
restart the server to disable answer reuse. The router may still abstain. No new packages,
collection rebuild, or new service is required. Entries exist only in server
memory, expire after five minutes, and are capped at 128 using least-recently-used
eviction. Restarting clears all entries. There is no disk cache or cross-process
sharing, and simultaneous cold requests may still make duplicate LLM calls.

Every request still verifies the JWT, embeds the question once, runs Qdrant's
tenant-and-role filter, and validates current retrieved chunks against the ACL.
Only then may a cached answer replace the DeepSeek generation step. Cache hits
therefore save LLM calls; they do not skip retrieval or authorization. Authorization
is checked again before responding, including when revocation happens during
generation or cache lookup.

Entries are isolated by verified user ID, tenant, role, and token version. The
full current chunk payloads and point IDs must match the cached context, so
changed retrieved text, citations, ACL metadata, added chunks, or removed chunks
cause a miss. Current scores and chunk order may differ; the response always
returns the current question and retrieval citations. Source-file edits still
require ingestion to update the vector corpus. Current ACL changes are enforced
immediately when loaded by the process, even against warmed cache entries.

An identical question after case/whitespace normalization gets an exact hit.
Semantic reuse requires cosine similarity of at least 0.95 and a matching
conservative question signature. That signature preserves content-word order,
company names, job titles, numbers, negation, units, and question form. It only
normalizes a few question-framing words and salary synonyms. This intentionally
misses many valid paraphrases. It is a limited equivalence heuristic, not proof
that any two questions have the same meaning; authorization comes from verified
identity, filtered retrieval, and ACL checks. Empty-context answers, empty
answers, the standard unknown answer, upstream errors, and answers rejected by
a final revocation check are not cached.

`POST /ask` keeps its existing response fields and adds:

| Field | Values |
| --- | --- |
| `route` | `rag`, `cache`, `graph`, `graph_cache`, or `abstain` |
| `routing_reason` | `supported_context`, `cached_answer`, `connected_evidence`, `insufficient_context`, `no_context`, or `router_disabled` |
| `retrieval_top_score` | Highest authorized retrieval score, or `null` without usable scores |
| `cache_status` | `miss`, `exact_hit`, `semantic_hit`, or `bypass` |
| `cache_similarity` | Match cosine similarity, or `null` on a miss/bypass |
| `graph_edge_count` | Number of evidence-backed graph relationships used; otherwise zero |
| `graph_evidence` | Cited graph relationships used as model context; an empty list for other routes |

For Bob, `What does the CEO earn?` first generates an answer (`miss`). Repeating
it gives an `exact_hit`. `How much does the CEO earn?` gives a `semantic_hit` with
the existing model (similarity 0.974743). Alice and Carol cannot reuse Bob's
answer. Asking about Globex now causes abstention. Asking about an engineering
manager, negation, or monthly pay forces a cache miss. Audit events include the cache decision and lookup time without
logging cached answer text.

Run the integration evaluation against existing vectors with real embeddings
and a mocked completion endpoint:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python eval/evaluate_cache.py
```

All 15 integration scenarios passed; three answer reuses avoided three completion
calls. Results are saved in `eval/cache_results.json`. No live DeepSeek calls were
made for this feature, so these figures establish routing and isolation rather
than live answer correctness or production latency gains. Run this script and
other scripts that access the persistent collection sequentially with the RAG
server stopped, since local Qdrant permits one owner at a time.

The full offline regression suite now has 201 passing tests. The adversarial runner
explicitly disables answer caching so cases continue to exercise either fresh
generation or the abstention router. The existing live Project 2 reports are
historical evidence from before caching, routing, and GraphRAG were introduced.

## Query routing: Project 3, second increment

The request order is JWT verification, embedding, ACL-filtered Qdrant retrieval,
ACL validation, routing, and then cache lookup or generation. The router is a
local cost optimization. It never changes identity, grants permission, or removes
unauthorized chunks after retrieval. An unauthorized chunk still fails the
entire batch with HTTP 403 before the router runs, even for unrelated questions.

By default, the router returns `abstain` when no context is retrieved, the best
score is below 0.25, the question names another configured tenant, or a salary
question has no salary-related evidence in the permitted context. Abstentions
return HTTP 200 with the same standard unknown answer, no source citations,
`cache_status: bypass`, and zero completion calls. They do not read or populate
the answer cache. Authorized retrieval IDs and the route remain in audit logs.
Ordinary paid-leave and expense questions are not treated as salary questions.

Otherwise, cache reuse is attempted; a miss takes the existing RAG path. Both
cache hits and generated answers retain the final JWT and ACL rechecks before
response. The client cannot select a route or override server configuration
through request bodies.

Configure routing in `.env` and restart:

```dotenv
QUERY_ROUTING_ENABLED=true
ROUTING_MIN_SCORE=0.25
```

Set `QUERY_ROUTING_ENABLED=false` to restore the pre-router behavior for nonempty
retrieval. Empty retrieval still returns the unknown answer without generation.
Disable `ANSWER_CACHE_ENABLED` separately if fresh generation is also required.
Both switches leave JWT and ACL checks in place.

The 0.25 threshold is based on a small local corpus check: a relevant Acme
expense-approval question scored about 0.28, while the tested unrelated questions
scored near zero. A score above the threshold does not prove that an answer
exists; for example, a password-policy question can still resemble a handbook
that lacks that policy. The LLM's grounding instructions remain necessary.
Company/topic checks use simple English patterns and can also over-abstain,
including a question that mentions another company only to exclude it. These
rules require broader calibration for a different corpus or language.

Run routing integration checks sequentially against the existing collection:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python eval/adversarial.py --cases eval/routing_cases.json --output eval/routing_results.json
```

All 17 scenarios passed with real embeddings and Qdrant and mocked completions:
nine supported questions reached RAG, and eight questions abstained with zero
completion calls. The cache evaluation still passes all 15 scenarios. Its three
hits avoid three completion calls; routing now also skips generation for Alice's
salary questions and Bob's cross-tenant question. These results establish route
selection and call avoidance, not live answer correctness or production savings.
No new live DeepSeek requests were made for this increment.

## GraphRAG: Project 4, first increment

Combined salary/policy questions can now use a request-local typed evidence
graph. For example, ask as Bob:

```json
{"question": "What is the software engineer salary and annual leave policy?"}
```

The graph connects `organization:acme -> has_salary_entry ->
role:acme:software_engineer -> annual_salary -> $110,000 per year` and joins
the organization's annual-leave evidence from the handbook. Every edge carries
its original source, chunk ID, and exact evidence sentence. DeepSeek receives
these cited relationships and sentences, preserving qualifiers such as
"Full-time"; a salary entry does not establish benefit eligibility.

The request still begins with verified identity, filtered vector retrieval,
batch authorization, and the existing router. A supported combined request then
expands through a separate Qdrant scroll with the same tenant-and-role filter.
All expanded chunks must pass current ACL validation before graph construction.
The selected evidence is rebuilt and authorized at the model boundary, and user
and document revocation are checked before returning an answer. Unauthorized
expansion fails the entire request with HTTP 403; no post-retrieval filtering
rescues a denied batch.

The initial increment recognized one job (CEO, engineering manager, or software
engineer) plus annual leave, remote work, remote availability hours, or expense
approval. Salary wording includes salary/compensation/wages and some earnings
phrases. Extraction uses the current English corpus sentence shapes. Missing,
ambiguous, or conflicting required evidence causes the standard unknown answer
without a completion call. For example, Globex has no remote availability hours
in its handbook, so that combined question abstains. Alice cannot obtain salary
facts through a graph join. Queries outside these patterns use existing routing.

Graph expansion is bounded to 64 permitted chunks per request; exceeding the
budget abstains rather than using a partial graph. No graph is persisted or
shared between identities. No graph database or new dependency is required,
and existing ingestion and vectors are retained. This is a first local graph
increment, with no community detection, graph embeddings, or global summaries.
The corpus currently has four points, so no retrieval improvement or production
scaling claim is made.

Fresh responses use `route: "graph"`, `routing_reason: "connected_evidence"`,
and `cache_status: "miss"` when graph answer caching is enabled. Reused graph
answers use `route: "graph_cache"`, as described below. Their source scores are
`null` because scroll expansion is not ranked
vector search; `retrieval_top_score` still describes the initial vector search.
Audit records include graph duration and edge count, without evidence text.
Set `GRAPH_RAG_ENABLED=false` in `.env` and restart to use the existing pipeline.

Run the offline integration checks sequentially with other persistent-Qdrant
scripts and with the server stopped:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python eval/evaluate_graph.py
```

The initial ten cases passed using real embeddings and existing vectors, with
only the completion endpoint mocked. Four graph requests assembled the expected cited
facts across two documents; six requests abstained with zero completion calls.
The runner checks actual outgoing graph context against authorized evidence;
the expanded suite described below now writes `eval/graph_results.json`. Unit
tests also cover expansion beyond the seed set, conflicting facts, unauthorized expansion, bounded retrieval,
cache bypass, direct model-call checks, and revocation during expansion,
prompt construction, and generation. No new live DeepSeek calls were made, so
live answer correctness remains unverified for the graph path.

## GraphRAG: Project 4, second increment

The graph now supports comparisons involving two or all three supported job
roles, with or without policy questions. For example, Bob can ask:

```json
{"question": "Compare CEO and software engineer salaries."}
```

Both organization-to-role-to-salary paths must be present and unambiguous.
Different salaries for different roles are valid; competing salary values for
the same role cause the entire request to abstain. A missing requested role
also prevents generation, so a partial comparison cannot reach the model.
Repeated mentions of the same role do not create extra paths.

Salary-only comparisons select just the salary document and the requested
roles' evidence. Adding annual leave, remote work, availability hours, or expense
approval requires the corresponding handbook evidence too. Eligibility
qualifiers and source/chunk citations are preserved. Single-role salary questions
keep the existing RAG/cache route. Graph comparisons use the same filtered
expansion, authorization/revocation checks, 64-chunk budget, and graph cache rules
as graph joins. Employee and cross-tenant comparisons still abstain.

The expanded `eval/evaluate_graph.py` suite passed all 16 cases: nine graph
requests and seven abstentions. It checks exact source/edge counts and expected
salary values in the actual outbound context, including the absence of
unrelated salary facts. New unit tests cover two-role and three-role traversal,
equal salaries for different roles, missing/conflicting role evidence,
salary-only citations, and employee/cross-tenant comparisons. That increment
brought the regression suite to 102 passing tests. These checks use mocked DeepSeek completions; they
establish evidence selection and isolation, not live comparison-answer quality
or arithmetic accuracy. No new dependencies or ingestion changes were needed.

## GraphRAG: Project 4, answer evaluation

The graph evaluator now has an opt-in live mode. The default still runs all
16 cases with real embeddings/Qdrant and mocked completions, writing
`eval/graph_results.json`. Live results use separate filenames so they cannot
silently replace offline evidence or earlier Project 2 reports.

Run a small live smoke test with the RAG server stopped:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python eval/evaluate_graph.py --live --smoke
```

This selects Bob's salary/leave join, Carol's two-role salary comparison, and
Alice's salary comparison abstention. It can make two real DeepSeek calls with
only the authenticated HR users' permitted graph evidence; Alice must make
none. Each call has a 30-second timeout and retries disabled. Results go to
`eval/graph_live_smoke_results.json`. The full `--live` suite can make nine calls
and writes `eval/graph_live_results.json`; `--output` selects another report path.

Before every live call, the runner independently checks the current token and
expanded chunks' ACLs, compares the actual serialized graph context with the
authorized evidence, and checks expected and unrelated fixture salary values.
It stops the suite on a failed outbound guard. Regression tests deliberately
bypass production ACL checks or tamper with the prompt and verify that the
evaluator prevents the real endpoint from being called.

Returned live answers are checked for expected salary/policy facts, eligibility
qualifiers where specified, source-and-chunk citations, invented citations,
foreign tenant mentions, and known unrelated fixture salaries. A conservative
role-before-value check also catches swapped salaries in comparisons. This
supports common prose/table formats and numeric variants; it may reject valid
alternative phrasing. These checks test fixture presence and associations,
not general semantic correctness, arbitrary encoded disclosure, or arithmetic.
No extra model judge call is made.

Reports include per-case checks, authorized answers for inspection, completion
counts, and timings. They omit raw contexts, tokens, keys, and exception messages;
upstream failures record only safe error types/status codes. Planned and executed
case counts remain separate if a guard stops the suite early. Latency excludes
login and model startup and is not a production benchmark.

The offline suite passed all 16 cases. That increment brought the regression
suite to 117 passing tests, including the live path exercised with a mocked endpoint. A live result
is evidence only for the selected cases; offline success does not establish
live answer quality.

Live smoke execution is pending explicit authorization to send the demo salary
and policy evidence to DeepSeek. Automatic approval review blocked the attempted
run before execution; no live GraphRAG results are claimed by this increment.

## GraphRAG: evidence inspection

Successful graph responses now include `graph_evidence`, exposing the selected
relationships and their exact supporting sentences. Each entry has this shape:

```json
{
  "subject": "role:acme:software_engineer",
  "relation": "annual_salary",
  "target": "$110,000 per year",
  "source": "data/acme/salaries.txt",
  "chunk_id": 0,
  "evidence": "Acme software engineers earn $110,000 per year."
}
```

The evidence describes model context. It does not certify every claim in the
generated answer. It preserves the original policy qualifiers and allows a
client to display the organization/role paths with source citations. Only
selected relationships are returned: a CEO/software-engineer comparison has
four edges from the salary document, without the engineering-manager salary
or unrelated policy facts. Ordinary RAG, ordinary RAG cache hits, and abstentions return
`graph_evidence: []`.

User revocation and current document ACLs are checked before evidence is
returned. The API also rebuilds the graph from the request's selected chunk
payloads after generation or cache lookup and compares it with the selected edges. A missing,
conflicting, or changed selected fact/citation returns HTTP 409 with a retry
message, without returning the answer or trace. Changes to unused facts do not
invalidate selected evidence. This check uses the request's in-memory chunks;
subsequent retrieval loads later persistent-document updates.

The graph evaluator now checks that the returned trace exactly matches the
evidence checked at the outgoing model boundary. It also requires empty traces
for abstentions. Evaluation reports store the trace comparison result. Audit
logs retain source IDs, counts, and timings. No new DeepSeek calls are required
for these checks; all completions remain mocked for this increment.

That increment brought the suite to 125 passing tests, including exact evidence/citation
selection, Carol's tenant isolation, empty traces on other routes, changed or
removed facts during generation, and detection of a deliberately corrupted
response trace by the evaluator. Existing vectors and ingestion were retained.

## GraphRAG: permission-aware answer caching

Graph answers now reuse the existing process-local semantic cache, with fresh
authenticated retrieval, ACL validation, expansion, and complete graph traversal
required on every request. Reuse skips only model generation. Cache hits return
`route: "graph_cache"`, `routing_reason: "cached_answer"`, and `cache_status:
"exact_hit"` or `"semantic_hit"`, along with the current sources and evidence
trace. Fresh generation uses `route: "graph"` and `cache_status: "miss"`.

Graph entries use a separate namespace from ordinary RAG. The namespace includes
the requested jobs/policies, selected relationships, facts, citations, and
`GRAPH_ANSWER_VERSION` in `graph_rag.py`. Bump that version when graph prompting,
generation model/settings, or extraction semantics change. Full selected chunk
payloads and point IDs must also match, and entries remain scoped to user,
tenant, access role, and token version. Edge ordering does not affect the graph
namespace. All namespaces share the existing 128-entry capacity and five-minute
TTL; there is no new persistent cache.

Graph reuse keeps the same conservative question-signature and cosine rules as
ordinary RAG reuse. It may miss valid paraphrases and is not proof that questions
are equivalent. New conflicting or missing required facts cause abstention
before cache lookup. Changed facts/citations, requested roles/policies, or graph
versions force a miss. User and document revocation are rechecked around lookup
and before the answer and trace are returned. Selected evidence changing during
a hit causes HTTP 409. Unknown/empty answers, upstream failures, and rejected
answers are not stored.

Set `GRAPH_ANSWER_CACHE_ENABLED=false` and restart to bypass graph answer reuse
while keeping ordinary RAG caching. `ANSWER_CACHE_ENABLED=false` disables both.
The fresh-generation graph evaluator disables the shared cache explicitly, so
its 16 cases and live smoke selection continue to exercise model generation or
abstention rather than warmed answers. Live execution is still pending approval.

Run the offline graph cache evaluation sequentially with other persistent-Qdrant
scripts and with the server stopped:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python eval/evaluate_graph_cache.py
```

All 14 scenarios passed with real embeddings and existing vectors. Three cache
hits avoided three mocked generation calls; seven cases generated mocked
answers, two abstained, and two were rejected on revocation. The report is
`eval/graph_cache_results.json`. It checks fresh authorized evidence on hits,
tenant/role isolation, changed questions, token/graph version changes, revocation,
and disabling graph caching. The full regression suite has 201 passing tests,
including conflicting facts, revocation during hits, namespace isolation, and
upstream failures. No new DeepSeek calls were made, and no live latency or cost
savings are claimed. Existing vectors and ingestion were retained.

## Verification and evaluation

Run the offline regression suite (uses cached embeddings, isolated temporary
Qdrant storage, and mocked DeepSeek calls):

```sh
HF_HUB_OFFLINE=1 .venv/bin/python -m unittest discover -s tests -v
```

Coverage includes Bob/Alice/Carol authenticated requests, 28 adversarial
tenant/role/query combinations, malformed/forged/expired JWTs, claim omissions,
role and tenant spoofing, user and document revocation, malformed payloads,
direct LLM-call protection, audit fields, citations, and concurrent local queries.

### Repeatable adversarial API checks

`eval/adversarial_cases.json` defines 22 authenticated API cases: three basic
salary checks, cross-tenant questions, prompt attempts to switch role/tenant,
query/header/body spoofing, invalid tokens, user revocation, and document ACL
revocation. Run against the existing vectors with mocked completions:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python eval/adversarial.py
```

All 22 current offline cases passed: two HR salary requests used authorized model
context, seven requests abstained, and 13 were rejected with HTTP 401/403/422.
The seven abstentions and 13 rejections made zero completion calls. Tests also
simulate bypassing production checks to verify that the evaluation guard stops
unsafe context before sending it. The original vectors were not changed.

The runner writes `eval/adversarial_offline_results.json`. It records per-case
status, identity, retrieved source citations, completion counts, checks, and
latency. Revocation changes are temporary and restored after each case; user and
ACL configuration files are untouched. Reports never store bearer tokens or keys.

With current routing enabled, the live version would make two real DeepSeek
calls with permitted HR context; seven cases must abstain and the other 13 must
be rejected without an LLM call:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python eval/adversarial.py --live
```

Live mode saves `eval/adversarial_live_results.json` and separately checks answers
for the expected salary/refusal and known forbidden salary amounts. Numeric
checks cover fixture variants such as `310000`, `310k`, and `310 thousand`; they
do not detect every possible paraphrase or encoding.

Before caching and routing were added, all 22 live-suite cases passed after user
approval: nine real DeepSeek calls
returned the expected salary/refusal with authorized context and no known
restricted salary amounts, and 13 rejected requests made zero completion calls.
Cross-tenant questions and prompt attempts to switch tenant or role returned the
standard unknown answer. Accepted requests had a median latency of 1.004 seconds
in this small sequential run, excluding login and model startup. This is evidence
for these cases, not a guarantee against every possible disclosure or a
production load benchmark. The original live smoke-test report remains intact.

Measure filtering and defense validation locally, without sending anything to
DeepSeek:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python eval/benchmark_acl.py --iterations 200
```

Results are saved in `eval/acl_benchmark.json`. The benchmark alternates filtered
and unfiltered timing controls, excludes embedding/model loading, HTTP, storage
opening, and LLM calls, and uses the existing four-point collection. These small
corpus measurements do not predict production load performance. Unfiltered
queries exist only in this offline timing script.

The raw RAG answer evaluator uses `eval/multitenant_questions.json`, passes verified
identity to retrieval, and guards context before generation. RAG latency excludes
the separate correctness judge. Running it makes real DeepSeek generation and
judge calls with permitted document context. It deliberately evaluates uncached
generation; routing and cache behavior are evaluated by the separate scripts:

```sh
.venv/bin/python eval/evaluate.py
```

Its output is `eval/multitenant_results.csv`, leaving `eval/questions.json` and
`eval/baseline_results.csv` intact. The original full handbook contains policies
missing from the smaller tenant handbooks; its historical ten-question benchmark
cannot be rerun unchanged against this corpus. No live multi-tenant evaluation
results are claimed by this update.
