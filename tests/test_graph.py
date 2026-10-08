"""Graph evidence, ACL expansion, prompt construction, and API regressions."""

import copy
import os
import secrets
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
from unittest.mock import patch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("DEEPSEEK_API_KEY", "test-only-no-network")
os.environ.setdefault("JWT_SECRET_KEY", secrets.token_urlsafe(48))

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

import auth
import graph_rag
import llm
import main
import search
from acl_config import DOCUMENT_ACLS
from authorization import AuthorizationError
from graph_rag import build_graph, graph_context, graph_cache_namespace, plan_graph_query, traverse_graph
from query_router import QueryRouter, UNKNOWN_ANSWER
from semantic_cache import SemanticAnswerCache

ROOT = Path(__file__).resolve().parent.parent
QUESTION = "What is the software engineer salary and annual leave policy?"


def point(source, *, text=None, point_id=1):
    return SimpleNamespace(id=point_id, score=0.8, payload={
        **copy.deepcopy(DOCUMENT_ACLS[source]), "source": source, "chunk_id": 0,
        "text": text if text is not None else (ROOT / source).read_text(),
    })


def corpus(tenant="acme"):
    return [point(f"data/{tenant}/handbook.txt"), point(f"data/{tenant}/salaries.txt", point_id=2)]


class GraphEvidenceTests(unittest.TestCase):
    def test_query_selection_keeps_existing_single_topic_routes(self):
        for question in ("CEO salary?", "Annual leave and remote work?", "How many days of paid leave do engineering managers receive?", "CEO salary and office hours?"):
            self.assertIsNone(plan_graph_query(question))
        self.assertEqual(plan_graph_query(QUESTION).jobs, ("software_engineer",))

    def test_two_hop_salary_path_joins_policy_with_provenance(self):
        evidence = traverse_graph(plan_graph_query(QUESTION), corpus(), tenant_id="acme", role="hr")
        self.assertTrue(evidence.complete)
        self.assertEqual(len(evidence.edges), 3)
        self.assertEqual(len(evidence.results), 2)
        self.assertEqual({edge.relation for edge in evidence.edges}, {"has_salary_entry", "annual_salary", "annual_leave"})
        self.assertIn("$110,000 per year", {edge.target for edge in evidence.edges})
        self.assertTrue(all(edge.source.startswith("data/acme/") and edge.chunk_id == 0 for edge in evidence.edges))

    def test_policy_qualifiers_are_preserved_without_claiming_eligibility(self):
        context = graph_context(plan_graph_query(QUESTION), corpus(), tenant_id="acme", role="hr")
        self.assertIn("Full-time Acme employees receive 20 days", context)
        self.assertIn("does not establish full-time status", context)
        self.assertNotIn("$240,000", context)
        self.assertNotIn("$150,000", context)

    def test_globex_graph_uses_only_globex_evidence(self):
        context = graph_context(plan_graph_query(QUESTION), corpus("globex"), tenant_id="globex", role="hr")
        self.assertIn("$125,000", context)
        self.assertIn("25 days", context)
        self.assertNotIn("acme", context.casefold())

    def test_missing_salary_policy_or_unknown_job_refuses_partial_graph(self):
        for question, documents in ((QUESTION, corpus()[:1]), (QUESTION, corpus()[1:]),
                                    ("CFO salary and annual leave?", corpus())):
            with self.subTest(question=question):
                plan = plan_graph_query(question)
                self.assertIsNotNone(plan)
                evidence = traverse_graph(plan, documents, tenant_id="acme", role="hr")
                self.assertFalse(evidence.complete)
                self.assertEqual(evidence.edges, ())

    def test_remote_availability_requires_its_own_evidence(self):
        plan = plan_graph_query("CEO salary and remote availability hours?")
        self.assertTrue(traverse_graph(plan, corpus(), tenant_id="acme", role="hr").complete)
        self.assertFalse(traverse_graph(plan, corpus("globex"), tenant_id="globex", role="hr").complete)

    def test_conflicting_salary_values_abstain(self):
        documents = corpus() + [point("data/acme/salaries.txt", text="Acme software engineers earn $999,000 per year.", point_id=3)]
        self.assertFalse(traverse_graph(plan_graph_query(QUESTION), documents, tenant_id="acme", role="hr").complete)

    def test_salary_comparison_selects_both_paths_without_requiring_policy(self):
        plan = plan_graph_query("Compare CEO and software engineer salaries.")
        evidence = traverse_graph(plan, corpus(), tenant_id="acme", role="hr")
        self.assertTrue(evidence.complete)
        self.assertEqual(len(evidence.edges), 4)
        self.assertEqual(len(evidence.results), 1)
        self.assertEqual({edge.target for edge in evidence.edges if edge.relation == "annual_salary"}, {"$240,000 per year", "$110,000 per year"})
        context = graph_context(plan, corpus(), tenant_id="acme", role="hr")
        self.assertNotIn("$150,000", context)
        self.assertNotIn("20 days", context)

    def test_all_three_roles_can_join_shared_policies(self):
        plan = plan_graph_query("Compare CEO, engineering manager, and software engineer salaries, annual leave and remote work.")
        evidence = traverse_graph(plan, corpus(), tenant_id="acme", role="hr")
        self.assertTrue(evidence.complete)
        self.assertEqual(len(evidence.edges), 8)
        self.assertEqual(len(evidence.results), 2)

    def test_comparison_needs_every_requested_role(self):
        plan = plan_graph_query("CEO and software engineer salaries and annual leave?")
        documents = corpus()
        documents[1].payload["text"] = "The Acme CEO earns $240,000 per year."
        self.assertFalse(traverse_graph(plan, documents, tenant_id="acme", role="hr").complete)
        with self.assertRaises(ValueError):
            graph_context(plan, documents, tenant_id="acme", role="hr")

    def test_conflict_in_one_compared_role_refuses_entire_comparison(self):
        plan = plan_graph_query("CEO and software engineer salaries and annual leave?")
        documents = corpus() + [point("data/acme/salaries.txt", text="Acme software engineers earn $999,000 per year.", point_id=3)]
        self.assertFalse(traverse_graph(plan, documents, tenant_id="acme", role="hr").complete)

    def test_equal_salaries_are_valid_for_distinct_roles(self):
        documents = corpus()
        documents[1].payload["text"] = documents[1].payload["text"].replace("$110,000", "$240,000")
        plan = plan_graph_query("Compare CEO and software engineer salaries.")
        self.assertTrue(traverse_graph(plan, documents, tenant_id="acme", role="hr").complete)

    def test_repeated_role_mention_does_not_create_a_comparison(self):
        self.assertIsNone(plan_graph_query("Compare CEO salary with CEO salary."))
        plan = plan_graph_query("Compare CEO, CEO and software engineer salaries.")
        self.assertEqual(len(plan.jobs), 2)

    def test_graph_namespace_preserves_facts_and_version_but_ignores_edge_order(self):
        plan = plan_graph_query(QUESTION)
        evidence = traverse_graph(plan, corpus(), tenant_id="acme", role="hr")
        namespace = graph_cache_namespace(plan, evidence)
        self.assertEqual(namespace, graph_cache_namespace(plan, replace(evidence, edges=evidence.edges[::-1])))
        changed = replace(evidence, edges=(replace(evidence.edges[0], chunk_id=9), *evidence.edges[1:]))
        self.assertNotEqual(namespace, graph_cache_namespace(plan, changed))
        with patch.object(graph_rag, "GRAPH_ANSWER_VERSION", "graph-answer:v2"):
            self.assertNotEqual(namespace, graph_cache_namespace(plan, evidence))
        with self.assertRaises(ValueError):
            graph_cache_namespace(plan, replace(evidence, complete=False))

    def test_mixed_tenant_and_role_denied_before_edge_construction(self):
        for documents, role in ((corpus() + corpus("globex"), "hr"), (corpus(), "employee")):
            with self.subTest(role=role):
                with self.assertRaises(AuthorizationError):
                    build_graph(documents, tenant_id="acme", role=role)

    def test_direct_graph_generator_checks_current_acl_before_completion(self):
        with patch.object(llm.client.chat.completions, "create") as completion:
            with self.assertRaises(AuthorizationError):
                llm.generate_graph_answer(QUESTION, corpus(), tenant_id="acme", role="employee", plan=plan_graph_query(QUESTION))
            with self.assertRaises(ValueError):
                llm.generate_graph_answer(QUESTION, corpus()[:1], tenant_id="acme", role="hr", plan=plan_graph_query(QUESTION))
        completion.assert_not_called()


class GraphAPITests(unittest.TestCase):
    def setUp(self):
        self.users = copy.deepcopy(auth.USERS)
        self.cache = SemanticAnswerCache()
        self.enterContext(patch.object(main, "answer_cache", self.cache))
        self.enterContext(patch.object(main, "query_router", QueryRouter()))
        self.enterContext(patch.object(main, "graph_rag_enabled", True))
        self.enterContext(patch.object(main, "graph_answer_cache_enabled", False))
        self.enterContext(patch.object(main, "embed_query", return_value=[1.0] + [0.0] * 383))
        # The salary seed alone is sufficient to begin graph expansion; the
        # handbook is found through filtered expansion, outside the seed set.
        self.retrieve = self.enterContext(patch.object(main, "search_documents", return_value=corpus()[1:]))
        self.expand = self.enterContext(patch.object(main, "load_graph_documents", return_value=corpus()))
        self.raw_generate = self.enterContext(patch.object(main, "generate_answer", return_value="Raw answer"))
        self.audit = self.enterContext(patch.object(main, "log_event"))
        self.completion = self.enterContext(patch.object(llm.client.chat.completions, "create", return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Mocked graph answer"))])))
        self.client = TestClient(main.app)
        self.addCleanup(self.client.close)

    def tearDown(self):
        auth.USERS.clear()
        auth.USERS.update(self.users)

    def ask(self, question=QUESTION, username="bob", token=None):
        user = auth.USERS[username]
        token = token or auth.create_access_token(username, user["tenant_id"], user["role"])
        return self.client.post("/ask", json={"question": question}, headers={"Authorization": "Bearer " + token})

    def test_graph_expands_sources_and_passes_only_selected_evidence(self):
        response = self.ask()
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["route"], "graph")
        self.assertEqual(data["graph_edge_count"], 3)
        self.assertEqual(len(data["sources"]), 2)
        self.expand.assert_called_once_with("acme", "hr")
        prompt = self.completion.call_args.kwargs["messages"][1]["content"]
        self.assertIn("$110,000", prompt)
        self.assertIn("20 days", prompt)
        self.assertNotIn("$240,000", prompt)
        self.assertNotIn("globex", prompt.casefold())
        self.raw_generate.assert_not_called()

    def test_graph_response_includes_exact_cited_evidence(self):
        data = self.ask().json()
        edges = data["graph_evidence"]
        self.assertEqual(len(edges), data["graph_edge_count"])
        self.assertEqual({edge["relation"] for edge in edges}, {"has_salary_entry", "annual_salary", "annual_leave"})
        citations = {(item["source"], item["chunk_id"]) for item in data["sources"]}
        prompt = self.completion.call_args.kwargs["messages"][1]["content"]
        for edge in edges:
            self.assertEqual(set(edge), {"subject", "relation", "target", "source", "chunk_id", "evidence"})
            self.assertIn((edge["source"], edge["chunk_id"]), citations)
            self.assertIn(edge["evidence"], prompt)
            self.assertTrue(edge["subject"].startswith(("organization:acme", "role:acme:")))
        salary = next(edge for edge in edges if edge["relation"] == "annual_salary")
        self.assertEqual(salary["target"], "$110,000 per year")
        self.assertEqual(salary["evidence"], "Acme software engineers earn $110,000 per year.")
        self.assertNotIn("$240,000", str(edges))
        self.assertNotIn("$150,000", str(edges))
        self.assertNotIn("globex", str(edges))

    def test_comparison_evidence_contains_only_requested_salary_paths(self):
        data = self.ask("Compare CEO and software engineer salaries.").json()
        self.assertEqual(len(data["graph_evidence"]), 4)
        self.assertEqual({edge["source"] for edge in data["graph_evidence"]}, {"data/acme/salaries.txt"})
        self.assertNotIn("$150,000", str(data["graph_evidence"]))
        self.assertNotIn("annual_leave", str(data["graph_evidence"]))

    def test_carol_evidence_contains_only_globex_facts(self):
        self.retrieve.return_value = corpus("globex")
        self.expand.return_value = corpus("globex")
        data = self.ask(username="carol").json()
        self.assertEqual(data["route"], "graph")
        self.assertIn("$125,000", str(data["graph_evidence"]))
        self.assertNotIn("acme", str(data["graph_evidence"]))

    def test_rag_and_cache_responses_have_no_graph_evidence(self):
        first, second = self.ask("CEO salary?").json(), self.ask("CEO salary?").json()
        self.assertEqual((first["route"], second["route"]), ("rag", "cache"))
        for data in (first, second):
            self.assertEqual(data["graph_evidence"], [])
            self.assertEqual(data["graph_edge_count"], 0)

    def test_disabled_graph_cache_bypasses_repeated_requests(self):
        with patch.object(self.cache, "lookup", wraps=self.cache.lookup) as lookup, patch.object(self.cache, "store", wraps=self.cache.store) as store:
            for _ in range(2):
                self.assertEqual(self.ask().json()["cache_status"], "bypass")
        self.assertEqual(self.completion.call_count, 2)
        lookup.assert_not_called()
        store.assert_not_called()

    def test_enabled_graph_cache_skips_only_generation_and_keeps_current_evidence(self):
        with patch.object(main, "graph_answer_cache_enabled", True):
            first, exact = self.ask().json(), self.ask().json()
            semantic = self.ask("How much is the software engineer salary and annual leave policy?").json()
        self.assertEqual([first["route"], exact["route"], semantic["route"]], ["graph", "graph_cache", "graph_cache"])
        self.assertEqual([first["cache_status"], exact["cache_status"], semantic["cache_status"]], ["miss", "exact_hit", "semantic_hit"])
        self.assertEqual(first["graph_evidence"], exact["graph_evidence"])
        self.assertEqual(exact["graph_evidence"], semantic["graph_evidence"])
        self.assertEqual(self.retrieve.call_count, 3)
        self.assertEqual(self.expand.call_count, 3)
        self.completion.assert_called_once()

    def test_rag_entry_cannot_substitute_for_graph_answer(self):
        identity = auth.verify_access_token(auth.create_access_token("bob", "acme", "hr"))
        self.cache.store(QUESTION, [1.0] + [0.0] * 383, identity, corpus(), "Wrong RAG answer")
        with patch.object(main, "graph_answer_cache_enabled", True):
            response = self.ask().json()
        self.assertEqual(response["cache_status"], "miss")
        self.assertNotEqual(response["answer"], "Wrong RAG answer")
        self.completion.assert_called_once()

    def test_graph_fact_or_version_change_forces_cache_miss(self):
        with patch.object(main, "graph_answer_cache_enabled", True):
            self.ask()
            salary = self.expand.return_value[1]
            salary.payload["text"] = salary.payload["text"].replace("$110,000", "$111,000")
            changed = self.ask().json()
            self.assertEqual(changed["cache_status"], "miss")
            self.assertIn("$111,000", str(changed["graph_evidence"]))
            with patch.object(graph_rag, "GRAPH_ANSWER_VERSION", "graph-answer:v2"):
                self.assertEqual(self.ask().json()["cache_status"], "miss")
                self.assertEqual(self.ask().json()["cache_status"], "exact_hit")
        self.assertEqual(self.completion.call_count, 3)

    def test_new_conflicting_graph_fact_cannot_use_warmed_answer(self):
        with patch.object(main, "graph_answer_cache_enabled", True):
            self.ask()
            self.expand.return_value.append(point("data/acme/salaries.txt", text="Acme software engineers earn $999,000 per year.", point_id=3))
            with patch.object(self.cache, "lookup", wraps=self.cache.lookup) as lookup:
                data = self.ask().json()
        self.assertEqual(data["route"], "abstain")
        self.assertEqual(data["graph_evidence"], [])
        lookup.assert_not_called()
        self.completion.assert_called_once()

    def test_graph_cache_cannot_cross_tenant_or_employee_boundary(self):
        with patch.object(main, "graph_answer_cache_enabled", True):
            self.ask()
            self.retrieve.return_value = corpus()[:1]
            self.expand.return_value = corpus()[:1]
            self.assertEqual(self.ask(username="alice").json()["route"], "abstain")
            self.retrieve.return_value = corpus("globex")
            self.expand.return_value = corpus("globex")
            carol = self.ask(username="carol").json()
            self.assertEqual(carol["cache_status"], "miss")
            self.assertNotIn("acme", str(carol["graph_evidence"]))
        self.assertEqual(self.completion.call_count, 2)

    def test_graph_cache_respects_user_revocation_and_token_version(self):
        with patch.object(main, "graph_answer_cache_enabled", True):
            token = auth.create_access_token("bob", "acme", "hr")
            self.ask()
            auth.USERS["bob"]["active"] = False
            with patch.object(self.cache, "lookup", wraps=self.cache.lookup) as lookup:
                self.assertEqual(self.ask(token=token).status_code, 401)
            lookup.assert_not_called()
            auth.USERS["bob"]["active"] = True
            auth.USERS["bob"]["token_version"] += 1
            self.assertEqual(self.ask().json()["cache_status"], "miss")
        self.assertEqual(self.completion.call_count, 2)

    def test_graph_cache_respects_document_revocation_before_lookup(self):
        with patch.object(main, "graph_answer_cache_enabled", True):
            self.ask()
            revised = copy.deepcopy(DOCUMENT_ACLS)
            del revised["data/acme/salaries.txt"]
            with patch.dict(DOCUMENT_ACLS, revised, clear=True), patch.object(self.cache, "lookup", wraps=self.cache.lookup) as lookup:
                self.assertEqual(self.ask().status_code, 403)
            lookup.assert_not_called()
        self.completion.assert_called_once()

    def test_revocation_during_graph_cache_lookup_blocks_response(self):
        with patch.object(main, "graph_answer_cache_enabled", True):
            self.ask()
            lookup = self.cache.lookup
            def revoke(*args, **kwargs):
                match = lookup(*args, **kwargs)
                auth.USERS["bob"]["active"] = False
                return match
            with patch.object(self.cache, "lookup", side_effect=revoke):
                self.assertEqual(self.ask().status_code, 401)
        self.completion.assert_called_once()

    def test_document_revocation_during_graph_cache_hit_blocks_response(self):
        with patch.object(main, "graph_answer_cache_enabled", True):
            self.ask()
            lookup = self.cache.lookup
            with patch.dict(DOCUMENT_ACLS, copy.deepcopy(DOCUMENT_ACLS), clear=True):
                def revoke(*args, **kwargs):
                    match = lookup(*args, **kwargs)
                    del DOCUMENT_ACLS["data/acme/salaries.txt"]
                    return match
                with patch.object(self.cache, "lookup", side_effect=revoke):
                    self.assertEqual(self.ask().status_code, 403)
        self.completion.assert_called_once()

    def test_evidence_changed_during_graph_cache_hit_blocks_response(self):
        with patch.object(main, "graph_answer_cache_enabled", True):
            self.ask()
            lookup = self.cache.lookup
            def change(*args, **kwargs):
                match = lookup(*args, **kwargs)
                salary = self.expand.return_value[1]
                salary.payload["text"] = salary.payload["text"].replace("$110,000", "$111,000")
                return match
            with patch.object(self.cache, "lookup", side_effect=change):
                self.assertEqual(self.ask().status_code, 409)
        self.completion.assert_called_once()

    def test_graph_unknown_answers_and_upstream_failures_are_not_cached(self):
        with patch.object(main, "graph_answer_cache_enabled", True):
            self.completion.return_value.choices[0].message.content = UNKNOWN_ANSWER
            self.assertEqual(self.ask().json()["cache_status"], "miss")
            self.assertEqual(self.ask().json()["cache_status"], "miss")
            self.completion.side_effect = RuntimeError("Mocked upstream failure")
            with self.assertRaises(RuntimeError):
                self.ask()
            self.completion.side_effect = None
            self.completion.return_value.choices[0].message.content = "Mocked graph answer"
            self.assertEqual(self.ask().json()["cache_status"], "miss")

    def test_global_cache_switch_disables_graph_reuse(self):
        self.cache.enabled = False
        with patch.object(main, "graph_answer_cache_enabled", True):
            self.assertEqual(self.ask().json()["cache_status"], "bypass")
            self.assertEqual(self.ask().json()["cache_status"], "bypass")
        self.assertEqual(self.completion.call_count, 2)

    def test_comparison_uses_only_salary_citation_and_bypasses_cache(self):
        data = self.ask("Compare CEO and software engineer salaries.").json()
        self.assertEqual(data["route"], "graph")
        self.assertEqual(data["graph_edge_count"], 4)
        self.assertEqual(data["cache_status"], "bypass")
        self.assertEqual([item["source"] for item in data["sources"]], ["data/acme/salaries.txt"])
        prompt = self.completion.call_args.kwargs["messages"][1]["content"]
        self.assertIn("$240,000", prompt)
        self.assertIn("$110,000", prompt)
        self.assertNotIn("$150,000", prompt)
        self.raw_generate.assert_not_called()

    def test_incomplete_comparison_does_not_send_partial_context(self):
        self.expand.return_value[1].payload["text"] = "The Acme CEO earns $240,000 per year."
        data = self.ask("CEO and software engineer salaries and annual leave?").json()
        self.assertEqual(data["route"], "abstain")
        self.assertEqual(data["sources"], [])
        self.assertEqual(data["graph_evidence"], [])
        self.completion.assert_not_called()

    def test_comparison_denies_employee_salary_and_foreign_tenant_requests(self):
        self.retrieve.return_value = corpus()[:1]
        self.assertEqual(self.ask("Compare CEO and software engineer salaries.", username="alice").json()["route"], "abstain")
        self.retrieve.return_value = corpus()[1:]
        self.assertEqual(self.ask("Compare Acme and Globex CEO and software engineer salaries.").json()["route"], "abstain")
        self.expand.assert_not_called()
        self.completion.assert_not_called()

    def test_employee_cannot_derive_salary_through_graph(self):
        self.retrieve.return_value = corpus()[:1]
        self.expand.return_value = corpus()[:1]
        data = self.ask(username="alice").json()
        self.assertEqual(data["route"], "abstain")
        self.assertEqual(data["answer"], UNKNOWN_ANSWER)
        self.assertEqual(data["sources"], [])
        self.assertEqual(data["graph_evidence"], [])
        self.completion.assert_not_called()

    def test_employee_earnings_paraphrase_cannot_derive_missing_salary(self):
        self.retrieve.return_value = corpus()[:1]
        self.expand.return_value = corpus()[:1]
        self.assertEqual(self.ask("What does the software engineer earn and what is the remote policy?", username="alice").json()["route"], "abstain")
        self.expand.assert_called_once_with("acme", "employee")
        self.completion.assert_not_called()

    def test_foreign_tenant_question_does_not_expand_graph(self):
        self.assertEqual(self.ask("Globex software engineer salary and annual leave?").json()["route"], "abstain")
        self.expand.assert_not_called()
        self.completion.assert_not_called()

    def test_unauthorized_expansion_rejects_whole_batch(self):
        self.expand.return_value += corpus("globex")
        self.assertEqual(self.ask().status_code, 403)
        self.completion.assert_not_called()

    def test_incomplete_or_over_budget_graph_abstains_without_completion(self):
        self.expand.return_value = corpus()[1:]
        self.assertEqual(self.ask().json()["route"], "abstain")
        self.expand.side_effect = search.GraphExpansionLimitError()
        self.assertEqual(self.ask().json()["route"], "abstain")
        self.completion.assert_not_called()

    def test_revocation_during_expansion_blocks_completion(self):
        def expand(*args):
            auth.USERS["bob"]["active"] = False
            return corpus()
        self.expand.side_effect = expand
        self.assertEqual(self.ask().status_code, 401)
        self.completion.assert_not_called()

    def test_revocation_during_generation_blocks_response(self):
        def complete(**kwargs):
            auth.USERS["bob"]["active"] = False
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Hidden answer"))])
        self.completion.side_effect = complete
        self.assertEqual(self.ask().status_code, 401)

    def test_changed_graph_fact_prevents_stale_answer_and_evidence_response(self):
        def complete(**kwargs):
            point = self.expand.return_value[1]
            point.payload["text"] = point.payload["text"].replace("$110,000", "$111,000")
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="STALE_ANSWER"))])
        self.completion.side_effect = complete
        response = self.ask()
        self.assertEqual(response.status_code, 409)
        self.assertNotIn("STALE_ANSWER", response.text)
        self.assertNotIn("graph_evidence", response.json())
        self.assertEqual(self.audit.call_args.kwargs["reason"], "graph_evidence_changed")

    def test_removed_graph_fact_prevents_stale_evidence_response(self):
        def complete(**kwargs):
            self.expand.return_value[1].payload["text"] = "The Acme CEO earns $240,000 per year."
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="STALE_ANSWER"))])
        self.completion.side_effect = complete
        self.assertEqual(self.ask().status_code, 409)

    def test_change_to_unused_fact_does_not_invalidate_selected_evidence(self):
        def complete(**kwargs):
            point = self.expand.return_value[1]
            point.payload["text"] = point.payload["text"].replace("$240,000", "$241,000")
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Mocked graph answer"))])
        self.completion.side_effect = complete
        response = self.ask()
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("$241,000", str(response.json()["graph_evidence"]))

    def test_document_revocation_during_generation_blocks_response(self):
        revised = copy.deepcopy(DOCUMENT_ACLS)
        with patch.dict(DOCUMENT_ACLS, revised, clear=True):
            def complete(**kwargs):
                del DOCUMENT_ACLS["data/acme/salaries.txt"]
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Hidden answer"))])
            self.completion.side_effect = complete
            self.assertEqual(self.ask().status_code, 403)

    def test_policy_revocation_at_graph_prompt_boundary_prevents_completion(self):
        generate = main.generate_graph_answer
        revised = copy.deepcopy(DOCUMENT_ACLS)
        with patch.dict(DOCUMENT_ACLS, revised, clear=True):
            def revoke(*args, **kwargs):
                del DOCUMENT_ACLS["data/acme/salaries.txt"]
                return generate(*args, **kwargs)
            with patch.object(main, "generate_graph_answer", side_effect=revoke):
                self.assertEqual(self.ask().status_code, 403)
        self.completion.assert_not_called()

    def test_disabled_graph_preserves_existing_rag_path(self):
        with patch.object(main, "graph_rag_enabled", False):
            self.assertEqual(self.ask().json()["route"], "rag")
        self.expand.assert_not_called()
        self.completion.assert_not_called()
        self.raw_generate.assert_called_once()


class FilteredGraphStorageTests(unittest.TestCase):
    def test_real_qdrant_filters_expansion_and_enforces_chunk_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            client = QdrantClient(path=directory)
            try:
                client.create_collection(search.COLLECTION_NAME, vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
                documents = corpus() + corpus("globex")
                points = [models.PointStruct(id=index, vector=[1.0] + [0.0] * 383, payload=document.payload) for index, document in enumerate(documents)]
                points.append(models.PointStruct(id=99, vector=[1.0] + [0.0] * 383, payload={"text": "Unscoped secret"}))
                client.upsert(search.COLLECTION_NAME, points)
            finally:
                client.close()
            with patch.object(search, "QDRANT_PATH", Path(directory)):
                for tenant, role, count in (("acme", "employee", 1), ("acme", "hr", 2), ("globex", "hr", 2)):
                    results = search.load_graph_documents(tenant, role)
                    self.assertEqual(len(results), count)
                    self.assertTrue(all(point.payload["tenant_id"] == tenant and role in point.payload["allowed_roles"] for point in results))
                with self.assertRaises(search.GraphExpansionLimitError):
                    search.load_graph_documents("acme", "hr", max_chunks=1)
                # Even the exact boundary works, and the failed attempt releases storage.
                self.assertEqual(len(search.load_graph_documents("acme", "hr", max_chunks=2)), 2)


if __name__ == "__main__":
    unittest.main()
