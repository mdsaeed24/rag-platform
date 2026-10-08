"""Bounded, process-local answer cache; retrieval and authorization stay mandatory."""

import hashlib
import json
import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from time import monotonic


def normalize_question(question):
    return " ".join(question.casefold().split())


def question_signature(question):
    """Preserve content order, entities, numbers and negation across paraphrases.

    Only a small set of framing words and salary synonyms are normalized.
    Similarity alone is not enough to reuse an answer. This deliberately misses
    many valid paraphrases rather than conflating different questions.
    """
    framing = {"what", "how", "much", "does", "do", "is", "the", "a", "an", "s", "please"}
    synonyms = {"earn": "salary", "earns": "salary", "earning": "salary", "salary": "salary"}
    tokens = re.findall(r"\w+", question.casefold())
    if tokens[:1] == ["what"] or tokens[:2] == ["how", "much"]:
        question_form = "value"
    else:
        # In particular, a yes/no question and a value question must not share
        # an answer even if their embeddings happen to be almost identical.
        question_form = tokens[0] if tokens else "empty"
    return (question_form, *(synonyms.get(token, token) for token in tokens if token not in framing))


def context_fingerprint(results):
    # Scores and ranking order vary with paraphrases. Full payloads and point
    # identities must still match, including text, citations and all ACL fields.
    chunks = [{"point_id": str(point.id), "payload": point.payload} for point in results]
    chunks.sort(key=lambda chunk: json.dumps(chunk, sort_keys=True, ensure_ascii=True))
    return hashlib.sha256(json.dumps(chunks, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def normalize_vector(vector):
    values = tuple(float(value) for value in vector)
    if len(values) != 384 or not all(math.isfinite(value) for value in values):
        raise ValueError("Cache requires a finite 384-dimensional embedding")
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        raise ValueError("Cache requires a nonzero embedding")
    return tuple(value / norm for value in values)


@dataclass(frozen=True)
class CacheMatch:
    answer: str
    status: str
    similarity: float


@dataclass(frozen=True)
class CacheEntry:
    namespace: str
    scope: tuple
    context: str
    question: str
    signature: tuple
    vector: tuple
    answer: str
    expires_at: float


class SemanticAnswerCache:
    def __init__(self, *, enabled=True, max_entries=128, ttl_seconds=300, similarity_threshold=0.95, clock=monotonic):
        if max_entries < 1 or ttl_seconds <= 0 or not 0 < similarity_threshold <= 1:
            raise ValueError("Invalid cache capacity, TTL or similarity threshold")
        self.enabled = enabled
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self._clock = clock
        self._entries = OrderedDict()
        self._lock = Lock()

    @staticmethod
    def scope(identity):
        return tuple(identity[key] for key in ("user_id", "tenant_id", "role", "token_version"))

    def _prune(self):
        now = self._clock()
        for key in list(self._entries):
            if self._entries[key].expires_at <= now:
                del self._entries[key]

    def lookup(self, question, vector, identity, results, *, namespace="rag:v1"):
        if not self.enabled or not results:
            return None
        scope = self.scope(identity)
        context = context_fingerprint(results)
        question = normalize_question(question)
        key = (namespace, scope, context, question)
        with self._lock:
            self._prune()
            exact = self._entries.get(key)
            if exact:
                self._entries.move_to_end(key)
                return CacheMatch(exact.answer, "exact_hit", 1.0)
            vector = normalize_vector(vector)
            signature = question_signature(question)
            best_key, best, best_score = None, None, self.similarity_threshold
            for candidate_key, candidate in self._entries.items():
                if candidate.namespace != namespace or candidate.scope != scope or candidate.context != context or candidate.signature != signature:
                    continue
                score = max(-1.0, min(1.0, sum(a * b for a, b in zip(vector, candidate.vector))))
                if score >= best_score:
                    best_key, best, best_score = candidate_key, candidate, score
            if best:
                self._entries.move_to_end(best_key)
                return CacheMatch(best.answer, "semantic_hit", best_score)
        return None

    def store(self, question, vector, identity, results, answer, *, namespace="rag:v1"):
        if (
            not self.enabled or not results or not isinstance(answer, str)
            or not answer.strip() or len(answer) > 16000
            or "i don't know based on the provided documents" in answer.casefold()
        ):
            return
        question = normalize_question(question)
        entry = CacheEntry(
            namespace, self.scope(identity), context_fingerprint(results), question,
            question_signature(question), normalize_vector(vector), answer,
            self._clock() + self.ttl_seconds,
        )
        key = (entry.namespace, entry.scope, entry.context, entry.question)
        with self._lock:
            self._prune()
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self):
        with self._lock:
            self._entries.clear()
