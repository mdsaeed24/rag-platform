"""Preflight and run the native deployment with one Uvicorn worker."""

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import hashlib
import os
from pathlib import Path
import platform
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class PreflightError(Exception):
    """A fixed, non-sensitive deployment failure code."""


def check_dependencies(root=ROOT):
    mismatches = []
    lines = (root / "requirements.lock").read_text().splitlines()
    targets = [line.removeprefix("# Target: ") for line in lines if line.startswith("# Target: ")]
    if len(targets) != 1 or json.loads(targets[0]) != {
        "python": "3.14", "system": platform.system(), "machine": platform.machine(),
    }:
        raise PreflightError("dependency_lock_target_mismatch")
    expected_hash = "# Requirements-SHA256: " + hashlib.sha256((root / "requirements.txt").read_bytes()).hexdigest()
    if expected_hash not in lines:
        raise PreflightError("dependency_lock_out_of_date")
    count = 0
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)", line)
        if match is None:
            raise PreflightError("invalid_dependency_lock")
        name, expected = match.groups()
        count += 1
        try:
            installed = version(name)
        except PackageNotFoundError:
            installed = None
        if installed != expected:
            mismatches.append(name)
    if mismatches:
        raise PreflightError("dependencies_mismatch")
    if not count:
        raise PreflightError("invalid_dependency_lock")


def prepare_environment(root=ROOT):
    from dotenv import load_dotenv

    load_dotenv(root / ".env", override=False)
    # Require a provisioned model cache; deployment startup never downloads it.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.chdir(root)
    os.umask(0o077)
    if not os.getenv("DEEPSEEK_API_KEY") or os.environ["DEEPSEEK_API_KEY"] == "your-deepseek-api-key":
        raise PreflightError("provider_key_missing_or_placeholder")


def preflight(root=ROOT):
    if sys.version_info[:2] != (3, 14):
        raise PreflightError("python_3_14_required")
    from deploy.package import verify_release

    try:
        verify_release(root)
    except Exception:
        raise PreflightError("release_integrity_failed") from None
    check_dependencies(root)
    prepare_environment(root)
    try:
        import main
        from acl_config import DOCUMENT_ACLS
        from authorization import validate_document_acl

        for source, acl in DOCUMENT_ACLS.items():
            validate_document_acl(source, acl)
        if not main.local_dependencies_ready():
            raise PreflightError("local_dependencies_not_ready")
        log_dir = root / "logs"
        log_dir.mkdir(mode=0o700, exist_ok=True)
        with tempfile.TemporaryFile(dir=log_dir):
            pass
    except PreflightError:
        raise
    except Exception:
        raise PreflightError("application_configuration_or_storage_failed") from None
    return {"status": "ready", "workers": 1, "model_downloads": False, "provider_calls": 0}


def serve(*, host="127.0.0.1", port=8001):
    import portalocker
    import uvicorn

    os.umask(0o077)
    try:
        with portalocker.Lock(str(ROOT / ".rag-server.lock"), timeout=0):
            preflight()
            uvicorn.run("main:app", host=host, port=port, workers=1, reload=False,
                        proxy_headers=False, access_log=False, server_header=False,
                        limit_concurrency=32, timeout_keep_alive=5, timeout_graceful_shutdown=45)
    except portalocker.exceptions.LockException:
        raise PreflightError("deployment_instance_already_running") from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "serve"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    try:
        if args.command == "check":
            print(json.dumps(preflight(), sort_keys=True))
        else:
            serve(host=args.host, port=args.port)
    except PreflightError as exc:
        print(json.dumps({"status": "failed", "reason": str(exc)}), file=sys.stderr)
        return 1
    except Exception:
        print(json.dumps({"status": "failed", "reason": "deployment_startup_failed"}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
