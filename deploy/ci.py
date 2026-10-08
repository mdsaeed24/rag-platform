"""CI entry point: isolated offline checks without workspace vectors or real keys."""

import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

UNIT_RUNNER = '''import json, pathlib, sys, unittest
from unittest.mock import patch
with patch("socket.socket.connect", side_effect=AssertionError("Offline CI")), patch("socket.socket.connect_ex", side_effect=AssertionError("Offline CI")):
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(sys.argv[1]))
    report = {"total": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped), "successful": result.wasSuccessful()}
    pathlib.Path(sys.argv[2]).write_text(json.dumps(report, indent=2) + "\\n")
    raise SystemExit(0 if result.wasSuccessful() and result.testsRun and not result.skipped else 1)
'''

SMOKE_RUNNER = '''import importlib, json, pathlib, sys
from unittest.mock import patch
with patch("socket.socket.connect", side_effect=AssertionError("Offline CI")), patch("socket.socket.connect_ex", side_effect=AssertionError("Offline CI")):
    report = importlib.import_module(sys.argv[1]).run_smoke()
pathlib.Path(sys.argv[2]).write_text(json.dumps(report, indent=2) + "\\n")
passed = report["total"] == int(sys.argv[3]) and report["passed"] == report["total"] and not report["real_provider_calls"] and len(report["cases"]) == report["total"] and all(row["passed"] is True for row in report["cases"])
raise SystemExit(0 if passed else 1)
'''


def run(suite):
    output = ROOT / "dist/ci" / suite
    output.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "DEEPSEEK_API_KEY": "offline-ci-no-provider-access",
                   "JWT_SECRET_KEY": secrets.token_urlsafe(48), "HF_HUB_OFFLINE": "1",
                   "HF_HUB_DISABLE_TELEMETRY": "1", "METRICS_TOKEN": "", "RAG_API_URL": "http://127.0.0.1:8001"}
    phases = [("unit_tests", ["-c", UNIT_RUNNER, "tests" if suite == "backend" else "ui/tests", str(output / "unit_tests.json")])]
    if suite == "backend":
        phases.extend((name, ["-c", SMOKE_RUNNER, module, str(output / (name + ".json")), str(count)])
                      for name, module, count in (("deployment_smoke", "deploy.smoke", 11),
                                                  ("ui_integration_smoke", "ui.integration_smoke", 9)))
    results = []
    for name, command in phases:
        with (output / (name + ".log")).open("w") as log:
            completed = subprocess.run([sys.executable, *command], cwd=ROOT, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, timeout=180)
        passed = completed.returncode == 0
        results.append({"phase": name, "passed": passed})
        print(f"{suite}/{name}: {'passed' if passed else 'failed'}", flush=True)
        if not passed:
            break
    report = {"suite": suite, "passed": len(results) == len(phases) and all(row["passed"] for row in results),
              "real_provider_calls": 0, "socket_connections_blocked": True, "phases": results}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("backend", "ui"))
    args = parser.parse_args()
    raise SystemExit(0 if run(args.suite) else 1)
