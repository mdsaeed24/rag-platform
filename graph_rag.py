"""Request-local, evidence-backed graph for the demo salary/policy corpus.

No graph is shared between identities. Every edge comes from a currently
authorized chunk, and every traversal preserves that chunk's citation.
Extraction deliberately supports only the documented English sentence shapes.
"""

import re
import hashlib
import json
from dataclasses import asdict, dataclass

from authorization import authorize_results

# Bump when graph prompting, generation model/settings, or extraction semantics
# change. Selected facts and provenance are additionally hashed per request.
GRAPH_ANSWER_VERSION = "graph-answer:v1"


@dataclass(frozen=True)
class GraphPlan:
    jobs: tuple[str, ...]
    policies: tuple[str, ...]


@dataclass(frozen=True)
class Edge:
    subject: str
    relation: str
    target: str
    source: str
    chunk_id: int
    evidence: str


@dataclass(frozen=True)
class GraphEvidence:
    edges: tuple[Edge, ...]
    results: tuple
    complete: bool


def graph_cache_namespace(plan, evidence):
    if not evidence.complete:
        raise ValueError("Incomplete evidence cannot select a graph cache namespace")
    edges = sorted((asdict(edge) for edge in evidence.edges), key=lambda edge: json.dumps(edge, sort_keys=True))
    fingerprint = hashlib.sha256(json.dumps({"plan": asdict(plan), "edges": edges}, sort_keys=True).encode()).hexdigest()
    return f"{GRAPH_ANSWER_VERSION}:{fingerprint}"


JOBS = {
    "ceo": r"ceo",
    "engineering_manager": r"engineering managers?",
    "software_engineer": r"software engineers?",
}
POLICIES = {
    "annual_leave": r"\b(?:annual leave|paid leave|vacation)\b",
    "remote_work": r"\b(?:remote|remotely)\b",
    "remote_availability": (r"\bremote(?:ly)?\b.*\b(?:hours|available|availability)\b"
                            r"|\b(?:hours|available|availability)\b.*\bremote(?:ly)?\b"),
    "expense_approval": r"\b(?:expenses?|reimbursements?)\b",
}


def plan_graph_query(question):
    """Select salary/policy joins or comparisons between supported job roles."""
    policies = tuple(name for name, pattern in POLICIES.items() if re.search(pattern, question, re.I))
    jobs = [job for job, pattern in JOBS.items() if re.search(r"\b(?:" + pattern + r")\b", question, re.I)]
    salary = re.search(r"\b(?:salar(?:y|ies)|compensation|wages?)\b", question, re.I)
    earnings = jobs and re.search(r"\bearn(?:s)?\b", question, re.I)
    if not (salary or earnings) or (not policies and len(jobs) < 2):
        return None
    return GraphPlan(tuple(jobs), policies)


def build_graph(results, *, tenant_id, role):
    # Reject the entire batch before examining any text or constructing edges.
    authorize_results(results, tenant_id, role)
    organization = f"organization:{tenant_id}"
    tenant = re.escape(tenant_id)
    edges = []
    for result in results:
        payload = result.payload
        source, chunk_id = payload["source"], payload["chunk_id"]
        for line in payload["text"].splitlines():
            sentence = line.strip()
            for job, pattern in JOBS.items():
                match = re.fullmatch(
                    rf"(?:The )?{tenant} (?:{pattern}) earns? (\$[\d,]+) per year\.",
                    sentence, re.I,
                )
                if match:
                    job_node = f"role:{tenant_id}:{job}"
                    edges.append(Edge(organization, "has_salary_entry", job_node, source, chunk_id, sentence))
                    edges.append(Edge(job_node, "annual_salary", match[1] + " per year", source, chunk_id, sentence))
            shapes = {
                "annual_leave": rf"Full-time {tenant} employees receive \d+ days of paid annual leave per calendar year\.",
                "remote_work": rf"{tenant} employees may work remotely up to (?:\w+|\d+) days per week\.",
                "remote_availability": r"Employees must be available between \d{1,2}:\d{2} [AP]M and \d{1,2}:\d{2} [AP]M on remote work days\.",
                "expense_approval": rf"{tenant} business expenses above \$[\d,]+ require manager approval\.",
            }
            for policy, pattern in shapes.items():
                if re.fullmatch(pattern, sentence, re.I):
                    edges.append(Edge(organization, policy, sentence, source, chunk_id, sentence))
    return tuple(edges)


def traverse_graph(plan, results, *, tenant_id, role):
    """Join organization -> salary entry -> salary and organization -> policy.

    Policies remain organization-level evidence. A salary entry does not prove
    an employee's full-time status or eligibility for a particular benefit.
    Conflicting values fail closed rather than selecting an arbitrary value.
    """
    edges = build_graph(results, tenant_id=tenant_id, role=role)
    root = f"organization:{tenant_id}"
    jobs = tuple(f"role:{tenant_id}:{job}" for job in plan.jobs)
    selected = [edge for edge in edges if
                (edge.subject == root and edge.relation == "has_salary_entry" and edge.target in jobs)
                or (edge.subject in jobs and edge.relation == "annual_salary")
                or (edge.subject == root and edge.relation in plan.policies)]
    # Validate each role's path independently. Different roles can have
    # different salaries; only competing values for the same role conflict.
    complete = bool(jobs) and all(
        any(edge.subject == root and edge.relation == "has_salary_entry" and edge.target == job for edge in selected)
        and len({edge.target for edge in selected if edge.subject == job and edge.relation == "annual_salary"}) == 1
        for job in jobs
    ) and all(
        len({edge.target for edge in selected if edge.relation == relation}) == 1
        for relation in plan.policies
    )
    if not complete:
        return GraphEvidence((), (), False)
    citations = {(edge.source, edge.chunk_id) for edge in selected}
    evidence_results = tuple(point for point in results if (point.payload["source"], point.payload["chunk_id"]) in citations)
    return GraphEvidence(tuple(selected), evidence_results, True)


def graph_context(plan, results, *, tenant_id, role):
    # Rebuild from authorized current payloads at the LLM boundary. Callers
    # cannot inject independently supplied graph facts into the prompt.
    evidence = traverse_graph(plan, results, tenant_id=tenant_id, role=role)
    if not evidence.complete:
        raise ValueError("Incomplete graph evidence")
    parts = ["Graph relationships describe document evidence. Preserve policy eligibility qualifiers; "
             "a salary entry does not establish full-time status or benefit eligibility."]
    for edge in evidence.edges:
        parts.append(f"[Source: {edge.source}, Chunk: {edge.chunk_id}]\n"
                     f"Relationship: {edge.subject} -> {edge.relation} -> {edge.target}\n"
                     f"Evidence: {edge.evidence}")
    return "\n\n".join(parts)
