"""Run the fixed offline validation plan sequentially and write a bounded summary."""

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.benchmark_pipeline import benchmark, offline_environment

EXPECTED_COUNTS = {"adversarial": 22, "routing": 17, "rag_cache": 15,
                   "graph": 16, "graph_cache": 14, "deployment_smoke": 11}

UNIT_RUNNER = '''import json, pathlib, sys, unittest
from unittest.mock import patch
with patch("socket.socket.connect", side_effect=AssertionError("Offline network guard")), patch("socket.socket.connect_ex", side_effect=AssertionError("Offline network guard")):
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover("tests"))
    report = {"total": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped), "successful": result.wasSuccessful()}
    pathlib.Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
'''


def summarize_suite(name, report):
    rows = report.get("checks", report.get("cases", []))
    expected = EXPECTED_COUNTS[name]
    passed = (type(report.get("passed")) is int and report["passed"] == expected
              and report.get("total") == expected and len(rows) == expected
              and all(isinstance(row.get("id"), str) and row["id"] for row in rows)
              and len({row.get("id") for row in rows}) == expected
              and all(row.get("passed") is True for row in rows)
              and all(bool(row["checks"]) and all(value is True for value in row["checks"].values())
                      for row in rows if "checks" in row)
              and not report.get("live_answer_correctness_evaluated", False)
              and not report.get("answer_behavior_evaluated", False)
              and not report.get("real_provider_calls", 0)
              and not report.get("unsafe_outbound_context_detected", 0)
              and not report.get("unauthorized_outbound_context_detected", 0))
    return {"name": name, "passed": report.get("passed", 0), "total": report.get("total", 0),
            "checks_passed": passed,
            "mock_completion_calls": report.get("completion_calls", sum(row.get("mock_completion_calls", 0) for row in rows)),
            "mock_calls_avoided_by_cache": report.get("completion_calls_avoided_by_cache", 0)}


def corpus_fingerprint():
    from acl_config import DOCUMENT_ACLS
    from authorization import authorize_results
    from search import load_graph_documents
    from semantic_cache import context_fingerprint

    points = {}
    scopes = {(acl["tenant_id"], role) for acl in DOCUMENT_ACLS.values() for role in acl["allowed_roles"]}
    for tenant, role in sorted(scopes):
        batch = load_graph_documents(tenant, role)
        authorize_results(batch, tenant, role)
        points.update((str(point.id), point) for point in batch)
    return context_fingerprint(list(points.values()))


def input_fingerprint():
    paths = sorted({*ROOT.glob("*.py"), *ROOT.glob("tests/test_*.py"), *ROOT.glob("eval/*.py"),
                    *ROOT.glob("ui/*.py"), *ROOT.glob("ui/tests/test_*.py"), ROOT / "ui/requirements.txt",
                    ROOT / ".streamlit/config.toml", ROOT / ".streamlit/secrets.toml.example",
                    *ROOT.glob("deploy/*.py"), ROOT / "requirements.txt", ROOT / "requirements.lock",
                    ROOT / "eval/adversarial_cases.json", ROOT / "eval/routing_cases.json"})
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def render_markdown(report):
    unchanged = report.get("stored_payloads_unchanged")
    corpus_status = "yes" if unchanged is True else "no" if unchanged is False else "not checked"
    lines = ["# Final offline validation", "",
             f"Generated: {report['timestamp']}. Status: **{report['status']}**.", "",
             "The four-stage roadmap has local implementations of baseline RAG, tenant/role authorization, semantic caching and routing, and bounded GraphRAG. This report verifies the listed offline cases; it does not certify production readiness or live GraphRAG answer correctness.", "",
             "## Verification", "", "| Check | Result |", "| --- | --- |"]
    unit = report.get("unit_tests", {})
    if unit:
        lines.append(f"| Regression tests | {unit['total'] - unit['failures'] - unit['errors'] - unit['skipped']}/{unit['total']}; {unit['skipped']} skipped |")
    for suite in report.get("suites", []):
        lines.append(f"| {suite['name']} | {suite['passed']}/{suite['total']} |")
    lines.extend([f"| Real provider calls in this run | {report['real_provider_calls']} |",
                  f"| Stored payload fingerprint unchanged | {corpus_status} |", ""])
    pipeline = report.get("pipeline_benchmark")
    if pipeline:
        lines.extend(["## Local pipeline timing", "",
                      f"{pipeline['measured_requests']} measured requests across {pipeline['total']} scenarios; {pipeline['iterations_per_scenario']} samples and {pipeline['warmup_per_scenario']} excluded warmups per scenario. Corpus: {pipeline['point_count']} points. Cases run in alternating forward/reverse round-robin order with independent caches.", "",
                      "| Scenario | Median ms | p95 ms | Mocked generations | Retrievals | Graph expansions |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for case in pipeline["scenarios"]:
            lines.append(f"| {case['id']} | {case['median_ms']:.3f} | {case['p95_ms']:.3f} | {case['mock_completion_calls']} | {case['retrieval_calls']} | {case['graph_expansion_calls']} |")
        lines.extend(["", pipeline["latency_scope"], "",
                      "Cache-hit scenarios still perform retrieval and, for GraphRAG, graph expansion. Zero mocked generations on these hits demonstrate call avoidance for these requests. They do not establish a live latency reduction, cost saving, traffic hit rate, or answer accuracy. p95 uses the nearest-rank definition.", ""])
    acl = report.get("acl_benchmark")
    if acl:
        lines.extend(["## ACL microbenchmark", "",
                      f"{acl['iterations_per_identity']} measured query pairs per identity, after {acl['warmup_pairs']} excluded pairs. Filtered/unfiltered execution order alternates. The unfiltered query exists only as an offline timing control.", "",
                      "| Identity | Filtered median ms | Unfiltered median ms | Paired overhead median ms | Validation median ms |",
                      "| --- | ---: | ---: | ---: | ---: |"])
        for case in acl["cases"]:
            lines.append(f"| {case['tenant_id']}/{case['role']} | {case['filtered_qdrant']['median_ms']:.4f} | {case['unfiltered_timing_control']['median_ms']:.4f} | {case['paired_filter_overhead']['median_ms']:.4f} | {case['defense_validation']['median_ms']:.4f} |")
        lines.extend(["", acl["scope"], ""])
    lines.extend(["## Reproduce", "", "In the full development workspace, stop any RAG process sharing this local Qdrant directory, then run:", "",
                  "```sh", ".venv/bin/python -m eval.final_validation --iterations 30 --acl-iterations 200", "```", "",
                  "The runner uses dummy provider credentials, forces cached-model offline mode, blocks socket connections in the coordinator and test subprocess, and runs storage-owning phases sequentially. Detailed fresh reports and private logs are written to `dist/final-validation/`; the aggregate machine-readable result is `eval/final_results.json`. Historical baseline/live results are preserved. No ingestion runs. Evaluation tools, tests, and raw reports are excluded from the deployment archive; only this aggregate Markdown summary is packaged. Streamlit interface tests run separately in the UI environment; see ui/README.md.", "",
                  f"Validation input SHA-256: `{report['validation_inputs_sha256']}`.", "",
                  "## Remaining work", "",
                  "- Live GraphRAG fact/citation evaluation remains pending explicit approval for the specified demo evidence and DeepSeek destination. This run sends no evidence to a provider.",
                  "- This offline runner does not install dependencies or deploy containers. Separate native installation validation and its platform limits are recorded in the [deployment runbook](deploy/README.md). Container deployment remains untested.",
                  "- Production scaling needs shared persistent identity/policy storage, server-based vector storage, shared caching, and multi-instance controls. Current revocation, caching, metrics, and admission limits are process local.",
                  "- Broader GraphRAG needs a larger corpus, broader extraction, and retrieval/answer comparisons. The current graph handles limited English salary/policy sentence patterns; no graph retrieval-quality gain has been established.",
                  "- Public operation needs deployment-specific TLS, rate limits, slow-client controls, monitoring, backup/recovery drills, and replacement of public demo credentials. No public service is published.", ""])
    if report.get("failed_phase"):
        lines.extend([f"Failed phase: `{report['failed_phase']}`. Error details remain in local logs; this report does not claim completion.", ""])
    return "\n".join(lines)


def validate(iterations=30, acl_iterations=200):
    if iterations < 1 or acl_iterations < 1:
        raise ValueError("Iteration counts must be positive")
    directory = ROOT / "dist/final-validation"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "status": "failed",
              "runtime": {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine()},
              "validation_inputs_sha256": input_fingerprint(), "real_provider_calls": 0,
              "live_graph_answer_correctness_evaluated": False, "production_ready": False, "suites": []}
    phase = "unit_tests"
    with patch.dict(os.environ, offline_environment()), patch("socket.socket.connect", side_effect=AssertionError("Offline network guard")), patch("socket.socket.connect_ex", side_effect=AssertionError("Offline network guard")):
        try:
            unit_path = directory / "unit_tests.json"
            unit_path.unlink(missing_ok=True)
            with (directory / "unit_tests.log").open("w") as log:
                result = subprocess.run([sys.executable, "-c", UNIT_RUNNER, str(unit_path)],
                                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=120)
            unit = json.loads(unit_path.read_text())
            report["unit_tests"] = unit
            if result.returncode or not unit["successful"] or unit["total"] < 1 or unit["skipped"]:
                raise AssertionError("Regression suite failed or skipped checks")
            phase = "preflight"
            from deploy.run import preflight
            from deploy.smoke import run_smoke
            from eval import adversarial, evaluate_cache, evaluate_graph, evaluate_graph_cache, benchmark_acl
            with (directory / "preflight.log").open("w") as log, redirect_stdout(log):
                report["preflight"] = preflight()
            before = corpus_fingerprint()
            jobs = (
                ("adversarial", lambda: adversarial.run_suite(json.loads((ROOT / "eval/adversarial_cases.json").read_text()))),
                ("routing", lambda: adversarial.run_suite(json.loads((ROOT / "eval/routing_cases.json").read_text()))),
                ("rag_cache", evaluate_cache.evaluate), ("graph", evaluate_graph.evaluate),
                ("graph_cache", evaluate_graph_cache.evaluate), ("deployment_smoke", run_smoke),
            )
            for phase, job in jobs:
                with (directory / f"{phase}.log").open("w") as log, redirect_stdout(log):
                    result = job()
                (directory / f"{phase}.json").write_text(json.dumps(result, indent=2) + "\n")
                summary = summarize_suite(phase, result)
                report["suites"].append(summary)
                if not summary["checks_passed"]:
                    raise AssertionError("Offline scenario suite failed")
                print(f"{phase}: {summary['passed']}/{summary['total']}", flush=True)
            phase = "pipeline_benchmark"
            report[phase] = benchmark(iterations)
            (directory / f"{phase}.json").write_text(json.dumps(report[phase], indent=2) + "\n")
            phase = "acl_benchmark"
            report[phase] = benchmark_acl.benchmark(acl_iterations)
            (directory / f"{phase}.json").write_text(json.dumps(report[phase], indent=2) + "\n")
            phase = "corpus_integrity"
            report["stored_payloads_unchanged"] = before == corpus_fingerprint()
            if not report["stored_payloads_unchanged"]:
                raise AssertionError("Stored payloads changed")
            phase = "validation_input_integrity"
            if input_fingerprint() != report["validation_inputs_sha256"]:
                raise AssertionError("Validation inputs changed during execution")
            report["status"] = "offline_checks_passed"
        except Exception as exc:
            report["failed_phase"] = phase
            report["error_type"] = type(exc).__name__
    (ROOT / "eval/final_results.json").write_text(json.dumps(report, indent=2) + "\n")
    (ROOT / "FINAL_RESULTS.md").write_text(render_markdown(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--acl-iterations", type=int, default=200)
    args = parser.parse_args()
    report = validate(args.iterations, args.acl_iterations)
    print(json.dumps({key: report[key] for key in ("status", "real_provider_calls")}))
    raise SystemExit(0 if report["status"] == "offline_checks_passed" else 1)
