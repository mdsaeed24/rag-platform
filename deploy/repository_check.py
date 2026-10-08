"""Review Git's indexed blobs without printing credential values or file content."""

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
INPUT_JSON = {"questions.json", "multitenant_questions.json", "adversarial_cases.json", "routing_cases.json"}
PRIVATE_PARTS = {".git", ".venv", ".venv-ui", "__pycache__", "qdrant_data", "logs", "dist", ".aws", ".codex", ".agents"}
PATTERNS = (
    rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    rb"\bsk-[A-Za-z0-9_-]{20,}\b",
    rb"\bgh[pousr]_[A-Za-z0-9_]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{30,}\b",
    rb"\bAKIA[A-Z0-9]{16}\b",
    rb"\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]{20,}\b",
    rb"JWT_SECRET_KEY\s*(?:=|:)\s*[\"'][A-Za-z0-9_-]{32,}[\"']",
)


def blocked_path(name):
    path = PurePosixPath(name)
    return (path.is_absolute() or ".." in path.parts or any(part in PRIVATE_PARTS for part in path.parts)
            or (path.name == ".env" or path.name.startswith(".env.")) and path.name != ".env.example"
            or path.name == "secrets.toml" or path.suffix in {".pyc", ".pyo", ".pem", ".key", ".p12", ".pfx"}
            or (path.parent == PurePosixPath("eval") and
                (path.suffix == ".csv" or path.suffix == ".json" and path.name not in INPUT_JSON)))


def inspect_index(root=ROOT, *, private_values=()):
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True).stdout
    entries = git("ls-files", "--stage", "-z").split(b"\0")
    findings, count, size = [], 0, 0
    for entry in filter(None, entries):
        metadata, name_bytes = entry.split(b"\t", 1)
        mode, object_id, stage = metadata.decode().split()
        name = name_bytes.decode("utf-8")
        count += 1
        if stage != "0" or mode not in {"100644", "100755"}:
            findings.append({"path": name, "reason": "non_regular_or_unmerged_file"})
            continue
        if blocked_path(name):
            findings.append({"path": name, "reason": "private_or_generated_path"})
            continue
        content = git("cat-file", "blob", object_id)
        size += len(content)
        if len(content) > 2_000_000 or b"\0" in content:
            findings.append({"path": name, "reason": "unexpected_binary_or_large_file"})
        if any(re.search(pattern, content) for pattern in PATTERNS):
            findings.append({"path": name, "reason": "credential_pattern"})
        if any(value and value in content for value in private_values):
            findings.append({"path": name, "reason": "local_secret_value"})
    if not count:
        findings.append({"path": "", "reason": "empty_index"})
    return {"status": "failed" if findings else "passed", "files": count,
            "source_bytes": size, "findings": findings,
            "scope": "Current Git index only; selected credential patterns and optional exact local secret values. Not a comprehensive secret scanner or history scan."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-local-env", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/github-review.json")
    args = parser.parse_args()
    try:
        values = ()
        if args.check_local_env:
            from dotenv import dotenv_values
            settings = dotenv_values(ROOT / ".env")
            values = tuple(value.encode() for key in ("DEEPSEEK_API_KEY", "JWT_SECRET_KEY", "METRICS_TOKEN")
                           if (value := settings.get(key)))
        report = inspect_index(private_values=values)
    except Exception:
        print(json.dumps({"status": "failed", "reason": "index_review_failed"}))
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "scope"}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
