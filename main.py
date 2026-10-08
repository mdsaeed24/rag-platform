from time import perf_counter
from dataclasses import asdict
import os
from secrets import compare_digest

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from auth import authenticate_user, create_access_token, verify_access_token
from search import embed_query, search_documents, load_graph_documents, GraphExpansionLimitError, local_dependencies_ready
from llm import generate_answer, generate_graph_answer, GenerationError
from authorization import AuthorizationError, authorize_results
from audit import log_event
from semantic_cache import SemanticAnswerCache
from query_router import QueryRouter, RoutingDecision, UNKNOWN_ANSWER
from graph_rag import plan_graph_query, traverse_graph, graph_cache_namespace
from operations import OperationalMetrics, RequestControlsMiddleware, positive_integer_setting, metrics_token_setting


answer_cache = SemanticAnswerCache(
    enabled=os.getenv("ANSWER_CACHE_ENABLED", "true").lower() == "true",
)
query_router = QueryRouter(
    enabled=os.getenv("QUERY_ROUTING_ENABLED", "true").lower() == "true",
    min_score=float(os.getenv("ROUTING_MIN_SCORE", "0.25")),
)
graph_rag_enabled = os.getenv("GRAPH_RAG_ENABLED", "true").lower() == "true"
graph_answer_cache_enabled = os.getenv("GRAPH_ANSWER_CACHE_ENABLED", "true").lower() == "true"
metrics_token = metrics_token_setting()
operational_metrics = OperationalMetrics(
    ask_limit=positive_integer_setting("MAX_INFLIGHT_ASK", 4),
    login_limit=positive_integer_setting("MAX_INFLIGHT_LOGIN", 4),
)


app = FastAPI(
    title="RAG Platform",
    description="Permission-aware multi-tenant RAG with caching, routing, and evidence-backed graph traversal",
    version="0.4.5",
)
app.add_middleware(RequestControlsMiddleware, metrics=operational_metrics,
                   max_body_bytes=positive_integer_setting("MAX_REQUEST_BODY_BYTES", 16384))


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=4000)


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=1024)


def authentication_error(reason, query=None, identity=None, results=()):
    log_event(decision="denied", reason=reason, query=query, identity=identity, results=results)
    return HTTPException(
        status_code=401,
        detail="Invalid credentials or access token",
        headers={"WWW-Authenticate": "Bearer"},
    )


@app.get("/")
def home():
    return {
        "message": "RAG platform is running!"
    }


@app.get("/health/live")
def liveness():
    return {"status": "alive"}


@app.get("/health/ready")
def readiness():
    try:
        ready = local_dependencies_ready()
    except Exception:
        ready = False
    return JSONResponse({"status": "ready" if ready else "not_ready"}, status_code=200 if ready else 503)


@app.get("/metrics", response_class=PlainTextResponse)
def metrics(authorization: str | None = Header(None)):
    if not metrics_token:
        raise HTTPException(status_code=404, detail="Not found")
    parts = authorization.split() if authorization else []
    if (len(parts) != 2 or parts[0].lower() != "bearer"
            or not compare_digest(parts[1].encode(), metrics_token.encode())):
        raise HTTPException(status_code=401, detail="Invalid metrics credentials",
                            headers={"WWW-Authenticate": "Bearer"})
    return PlainTextResponse(operational_metrics.render(), media_type="text/plain; version=0.0.4")


@app.post("/login")
def login(request: LoginRequest):
    user = authenticate_user(
        request.username,
        request.password,
    )

    if not user:
        raise authentication_error("invalid_login")

    token = create_access_token(
        user_id=user["user_id"],
        tenant_id=user["tenant_id"],
        role=user["role"],
    )
    log_event(decision="allowed", reason="login", identity=user)

    return {
        "access_token": token,
        "token_type": "bearer",
    }


@app.post("/ask")
def ask(
    request: QuestionRequest,
    authorization: str | None = Header(None),
):
    # Require Authorization header
    if not authorization:
        raise authentication_error("missing_authorization", request.question)

    # Expect: Authorization: Bearer <token>
    parts = authorization.split()

    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise authentication_error("malformed_authorization", request.question)

    token = parts[1]

    # Verify JWT
    payload = verify_access_token(token)

    if not payload:
        raise authentication_error("invalid_or_revoked_token", request.question)

    # Identity comes from the VERIFIED token
    tenant_id = payload["tenant_id"]
    role = payload["role"]
    user_id = payload["user_id"]

    # Permission-aware retrieval
    retrieval_start = perf_counter()
    query_embedding = embed_query(request.question)
    results = search_documents(
        query=request.question,
        tenant_id=tenant_id,
        role=role,
        limit=3,
        query_embedding=query_embedding,
    )
    retrieval_ms = (perf_counter() - retrieval_start) * 1000
    # A revocation may have happened while embedding/retrieval was running.
    if verify_access_token(token) is None:
        raise authentication_error("revoked_during_retrieval", request.question, payload, results)
    authorization_start = perf_counter()
    try:
        authorize_results(results, tenant_id, role)
    except AuthorizationError:
        log_event(
            decision="denied", reason="chunk_acl_violation", identity=payload,
            query=request.question, results=results, retrieval_ms=retrieval_ms,
        )
        raise HTTPException(status_code=403, detail="Retrieved context failed authorization")
    authorization_ms = (perf_counter() - authorization_start) * 1000
    routing_start = perf_counter()
    decision = query_router.decide(request.question, tenant_id, results)
    routing_ms = (perf_counter() - routing_start) * 1000
    graph_start = perf_counter()
    graph_plan = plan_graph_query(request.question) if graph_rag_enabled and decision.route == "rag" else None
    graph_evidence = None
    if graph_plan is not None:
        try:
            expanded_results = load_graph_documents(tenant_id, role)
            if verify_access_token(token) is None:
                raise authentication_error("revoked_during_graph_expansion", request.question, payload)
            graph_evidence = traverse_graph(graph_plan, expanded_results, tenant_id=tenant_id, role=role)
        except GraphExpansionLimitError:
            decision = RoutingDecision("abstain", "insufficient_context", decision.top_score)
        except AuthorizationError:
            log_event(decision="denied", reason="graph_acl_violation", identity=payload, query=request.question)
            raise HTTPException(status_code=403, detail="Retrieved context failed authorization")
        else:
            if graph_evidence.complete:
                results = graph_evidence.results
                decision = RoutingDecision("graph", "connected_evidence", decision.top_score)
            else:
                decision = RoutingDecision("abstain", "insufficient_context", decision.top_score)
    graph_ms = (perf_counter() - graph_start) * 1000
    uses_graph = decision.route == "graph"
    cache_allowed = decision.route == "rag" or (uses_graph and graph_answer_cache_enabled)
    cache_options = {"namespace": graph_cache_namespace(graph_plan, graph_evidence)} if uses_graph else {}
    cache_start = perf_counter()
    match = answer_cache.lookup(request.question, query_embedding, payload, results, **cache_options) if cache_allowed else None
    cache_ms = (perf_counter() - cache_start) * 1000
    cache_status = match.status if match else ("miss" if answer_cache.enabled and cache_allowed else "bypass")
    route = ("graph_cache" if uses_graph else "cache") if match else decision.route
    routing_reason = "cached_answer" if match else decision.reason
    if verify_access_token(token) is None:
        raise authentication_error("revoked_before_answer", request.question, payload, results)
    log_event(
        decision="allowed", reason="cached_answer_authorized" if match else "request_abstained" if route == "abstain" else "context_authorized", identity=payload,
        query=request.question, results=results, retrieval_ms=retrieval_ms,
        authorization_ms=authorization_ms, cache_lookup_ms=cache_ms,
        cache_status=cache_status,
        route=route, routing_reason=routing_reason, routing_ms=routing_ms,
        graph_ms=graph_ms, graph_edge_count=len(graph_evidence.edges) if graph_evidence else 0,
    )

    generation_start = perf_counter()
    try:
        if match:
            answer = match.answer
        elif route == "rag":
            answer = generate_answer(
                question=request.question,
                search_results=results,
                tenant_id=tenant_id,
                role=role,
            )
        elif route == "graph":
            answer = generate_graph_answer(
                request.question, results, tenant_id=tenant_id, role=role, plan=graph_plan,
            )
        else:
            answer = UNKNOWN_ANSWER
    except AuthorizationError:
        log_event(decision="denied", reason="graph_acl_violation" if uses_graph else "chunk_acl_violation", identity=payload, query=request.question, results=results)
        raise HTTPException(status_code=403, detail="Retrieved context failed authorization")
    except GenerationError as exc:
        log_event(
            decision="error", reason=exc.reason, identity=payload,
            query=request.question, results=results, route=route,
            cache_status=cache_status, generation_ms=(perf_counter() - generation_start) * 1000,
        )
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from None

    # Generation may take seconds. Recheck revocation and document policy
    # before either returning the answer or making it reusable by the cache.
    if verify_access_token(token) is None:
        raise authentication_error("revoked_before_response", request.question, payload, results)
    evidence_trace = []
    try:
        authorize_results(results, tenant_id, role)
        if uses_graph:
            final_evidence = traverse_graph(graph_plan, results, tenant_id=tenant_id, role=role)
            if not final_evidence.complete or final_evidence.edges != graph_evidence.edges:
                log_event(decision="denied", reason="graph_evidence_changed", identity=payload, query=request.question, results=results)
                raise HTTPException(status_code=409, detail="Graph evidence changed during the request; retry")
            evidence_trace = [asdict(edge) for edge in final_evidence.edges]
    except AuthorizationError:
        log_event(decision="denied", reason="acl_changed_before_response", identity=payload, query=request.question, results=results)
        raise HTTPException(status_code=403, detail="Retrieved context failed authorization")
    if not match and cache_allowed:
        answer_cache.store(request.question, query_embedding, payload, results, answer, **cache_options)

    sources = []

    for result in results if route != "abstain" else ():
        sources.append({
            "score": getattr(result, "score", None),
            "source": result.payload["source"],
            "chunk_id": result.payload["chunk_id"],
        })

    operational_metrics.record_answer(route)
    return {
        "question": request.question,
        "user_id": user_id,
        "tenant_id": tenant_id,
        "role": role,
        "answer": answer,
        "sources": sources,
        "route": route,
        "routing_reason": routing_reason,
        "retrieval_top_score": round(decision.top_score, 6) if decision.top_score is not None else None,
        "cache_status": cache_status,
        "cache_similarity": round(match.similarity, 6) if match else None,
        "graph_edge_count": len(graph_evidence.edges) if graph_evidence else 0,
        "graph_evidence": evidence_trace,
    }
