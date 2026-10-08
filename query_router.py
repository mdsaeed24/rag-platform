"""Local generation-cost routing after mandatory retrieval and authorization.

These heuristics decide whether to abstain. They never grant access, remove
unauthorized chunks from a batch, or change the verified identity.
"""

import math
import re
from dataclasses import dataclass

from acl_config import DOCUMENT_ACLS

UNKNOWN_ANSWER = "I don't know based on the provided documents."
_SALARY_TOPIC = re.compile(r"\b(?:salar(?:y|ies)|compensation|wages?|earn(?:s|ing|ings)?)\b", re.IGNORECASE)
_SALARY_EXPLICIT = re.compile(r"\b(?:salar(?:y|ies)|compensation|wages?)\b", re.IGNORECASE)
_POLICY_TOPIC = re.compile(r"\b(?:leave|vacation|remote|expenses?|reimbursements?)\b", re.IGNORECASE)
_JOB = re.compile(r"\b(?:ceo|engineers?|managers?|employees?)\b", re.IGNORECASE)
_PAY = re.compile(r"\b(?:pay|paid)\b", re.IGNORECASE)


@dataclass(frozen=True)
class RoutingDecision:
    route: str
    reason: str
    top_score: float | None


class QueryRouter:
    def __init__(self, *, enabled=True, min_score=0.25):
        if not math.isfinite(min_score) or not -1 <= min_score <= 1:
            raise ValueError("ROUTING_MIN_SCORE must be finite and between -1 and 1")
        self.enabled = enabled
        self.min_score = min_score

    def decide(self, question, tenant_id, results):
        scores = [float(point.score) for point in results if isinstance(point.score, (int, float)) and math.isfinite(point.score)]
        top_score = max(scores) if scores else None
        if not results:
            return RoutingDecision("abstain", "no_context", None)
        if not self.enabled:
            return RoutingDecision("rag", "router_disabled", top_score)
        if top_score is None or top_score < self.min_score:
            return RoutingDecision("abstain", "insufficient_context", top_score)

        # A request naming another configured tenant cannot be answered fully
        # from this tenant's chunks. The response never confirms another
        # tenant's document existence or explains a user's missing permission.
        tenants = {acl["tenant_id"] for acl in DOCUMENT_ACLS.values()}
        if any(
            tenant != tenant_id and re.search(r"(?<!\w)" + re.escape(tenant) + r"(?!\w)", question, re.IGNORECASE)
            for tenant in tenants
        ):
            return RoutingDecision("abstain", "insufficient_context", top_score)

        salary_question = bool(
            _SALARY_EXPLICIT.search(question)
            or (not _POLICY_TOPIC.search(question) and (_SALARY_TOPIC.search(question) or (_JOB.search(question) and _PAY.search(question))))
        )
        if salary_question and not any(_SALARY_TOPIC.search(point.payload["text"]) for point in results):
            return RoutingDecision("abstain", "insufficient_context", top_score)
        return RoutingDecision("rag", "supported_context", top_score)
