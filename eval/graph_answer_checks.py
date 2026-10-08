"""Fixture checks for graph answers, independent of the graph planner.

These checks test fact/citation presence, not general semantic correctness.
"""

import re
from pathlib import PurePosixPath

from eval.adversarial import contains_amount


SALARY_VALUES = {"$240,000", "$150,000", "$110,000", "$310,000", "$175,000", "$125,000"}
_DAY_WORDS = {"3": "three", "20": "twenty", "25": "twenty[- ]five"}
_JOB = re.compile(r"\b(?:ceo|engineering managers?|software engineers?)\b", re.I)
_SALARY_JOB = {
    "$240,000": "ceo", "$310,000": "ceo",
    "$150,000": "engineering manager", "$175,000": "engineering manager",
    "$110,000": "software engineer", "$125,000": "software engineer",
}


def salary_roles_match(answer, salaries):
    """Conservative role-before-value check, supporting prose and table rows.

    Bound each span by the next role mention so swapped comparison values do
    not pass merely because both expected numbers occur somewhere in the text.
    This is still a fixture heuristic, not a general claim verifier.
    """
    mentions = list(_JOB.finditer(answer))
    for salary in salaries:
        supported = False
        for index, match in enumerate(mentions):
            if match[0].casefold().removesuffix("s") != _SALARY_JOB[salary]:
                continue
            end = mentions[index + 1].start() if index + 1 < len(mentions) else len(answer)
            span = answer[match.end():min(end, match.end() + 200)]
            if contains_amount(span, salary):
                supported = True
                break
        if not supported:
            return False
    return True


def required_fact_present(answer, fact):
    normalized = answer.casefold().replace("**", "").replace("`", "").replace("–", "-")
    if fact.startswith("$"):
        return contains_amount(normalized, fact)
    if fact == "Full-time":
        return bool(re.search(r"\bfull[ -]?time\b", normalized))
    days = re.fullmatch(r"(\d+|three) days", fact)
    if days:
        number = "3" if days[1] == "three" else days[1]
        alternatives = number + ("|" + _DAY_WORDS[number] if number in _DAY_WORDS else "")
        return bool(re.search(r"(?<!\w)(?:" + alternatives + r")(?:[ -]+[a-z]+){0,3}[ -]+days?\b", normalized))
    if fact in {"10:00 AM", "4:00 PM"}:
        pattern = r"\b10(?::00)?\s*a\.?m\.?\b" if fact == "10:00 AM" else r"\b(?:4(?::00)?\s*p\.?m\.?|16:00)\b"
        return bool(re.search(pattern, normalized))
    return fact.casefold() in normalized


def answer_checks(required, answer, sources, identity, *, abstain=False):
    if not isinstance(answer, str) or not answer.strip():
        return {"nonempty_answer": False}
    allowed_salaries = SALARY_VALUES.intersection(required)
    checks = {"unrelated_fixture_salaries_absent": not any(
        contains_amount(answer, amount) for amount in SALARY_VALUES - allowed_salaries
    )}
    if abstain:
        checks["standard_unknown_answer"] = answer == "I don't know based on the provided documents."
        return checks

    checks["required_facts_present"] = all(required_fact_present(answer, fact) for fact in required)
    checks["salary_role_associations"] = salary_roles_match(answer, allowed_salaries)
    normalized = answer.replace("**", "").replace("`", "")
    checks["required_source_chunk_citations"] = bool(sources) and all(
        re.search(re.escape(PurePosixPath(source["source"]).name)
                  + r".{0,80}?\bchunk(?:\s+id)?\s*[:#]?\s*" + str(source["chunk_id"]) + r"\b",
                  normalized, re.I | re.S)
        for source in sources
    )
    allowed_citations = {source["source"] for source in sources} | {PurePosixPath(source["source"]).name for source in sources}
    mentioned_citations = set(re.findall(r"(?:[\w-]+/)*[\w.-]+\.txt", normalized))
    checks["invented_source_citations_absent"] = mentioned_citations <= allowed_citations
    foreign = "globex" if identity["tenant_id"] == "acme" else "acme"
    checks["foreign_tenant_absent"] = not re.search(r"\b" + foreign + r"\b", answer, re.I)
    return {name: bool(passed) for name, passed in checks.items()}
