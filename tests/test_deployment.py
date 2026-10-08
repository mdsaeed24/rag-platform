"""Native launcher and release packaging checks; no network or live generation."""

import hashlib
from importlib.metadata import PackageNotFoundError
import json
import os
from pathlib import Path
import platform
import secrets
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import zipfile

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("DEEPSEEK_API_KEY", "test-only-no-network")
os.environ.setdefault("JWT_SECRET_KEY", secrets.token_urlsafe(48))

import portalocker

from deploy.lock_dependencies import installed_closure
from deploy.package import RELEASE_FILES, build_release, verify_release
from deploy import run


class DependencyLockTests(unittest.TestCase):
    def test_transitive_extras_and_cycles_are_resolved_without_unused_extras(self):
        packages = {
            "root": SimpleNamespace(version="1.0", requires=["leaf>=2", "crypto; extra == 'secure'", "unused; extra == 'other'"]),
            "leaf": SimpleNamespace(version="2.1", requires=["root[secure]>=1"]),
            "crypto": SimpleNamespace(version="3.0", requires=[]),
        }
        result = installed_closure(["root==1.0"], lookup=packages.__getitem__)
        self.assertEqual(result, {"root": "1.0", "leaf": "2.1", "crypto": "3.0"})

    def test_unsatisfied_installed_versions_and_direct_urls_are_rejected(self):
        package = SimpleNamespace(version="1.0", requires=[])
        for requirement in ("root>=2", "root @ https://example.invalid/package.whl"):
            with self.subTest(requirement=requirement), self.assertRaises(ValueError):
                installed_closure([requirement], lookup=lambda name: package)

    def test_current_lock_matches_installed_dependency_closure(self):
        expected = installed_closure((run.ROOT / "requirements.txt").read_text().splitlines())
        actual = dict(line.split("==") for line in (run.ROOT / "requirements.lock").read_text().splitlines()
                      if line and not line.startswith("#"))
        self.assertEqual(actual, expected)
        run.check_dependencies()

    def make_lock(self, root, pin="sample==1.0"):
        declarations = "sample==1.0\n"
        (root / "requirements.txt").write_text(declarations)
        header = ("# Target: " + json.dumps({"python": "3.14", "system": platform.system(), "machine": platform.machine()})
                  + "\n# Requirements-SHA256: " + hashlib.sha256(declarations.encode()).hexdigest() + "\n")
        (root / "requirements.lock").write_text(header + pin + "\n")

    def test_version_mismatch_and_missing_package_fail_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_lock(root)
            for effect in (None, PackageNotFoundError()):
                with self.subTest(effect=type(effect).__name__), patch.object(run, "version", return_value="2.0", side_effect=effect):
                    with self.assertRaisesRegex(run.PreflightError, "dependencies_mismatch"):
                        run.check_dependencies(root)

    def test_changed_declarations_require_regenerating_the_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_lock(root)
            (root / "requirements.txt").write_text("sample==2.0\n")
            with self.assertRaisesRegex(run.PreflightError, "dependency_lock_out_of_date"):
                run.check_dependencies(root)

    def test_other_platform_and_empty_or_malformed_locks_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for pin in ("", "sample>=1", "sample @ https://example.invalid/file"):
                self.make_lock(root, pin)
                with self.subTest(pin=pin), self.assertRaisesRegex(run.PreflightError, "invalid_dependency_lock"):
                    run.check_dependencies(root)
            self.make_lock(root)
            with patch.object(run.platform, "machine", return_value="unsupported-platform"):
                with self.assertRaisesRegex(run.PreflightError, "dependency_lock_target_mismatch"):
                    run.check_dependencies(root)


class ReleasePackagingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for name in RELEASE_FILES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Allowlisted source: " + name)

    def test_archive_is_deterministic_and_excludes_sensitive_runtime_files(self):
        for name in (".env", "qdrant_data/meta.json", "data/acme/salaries.txt", "logs/audit.jsonl",
                     ".venv/config", ".venv-ui/config", ".streamlit/secrets.toml",
                     "eval/live_results.json", "dist/old.zip", ".rag-server.lock"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("PRIVATE_RUNTIME_SENTINEL")
        first, second = self.root / "first.zip", self.root / "second.zip"
        self.assertEqual(build_release(self.root, first), build_release(self.root, second))
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with zipfile.ZipFile(first) as archive:
            self.assertEqual(set(archive.namelist()), set(RELEASE_FILES) | {"release-manifest.json"})
            self.assertNotIn(b"PRIVATE_RUNTIME_SENTINEL", b"".join(archive.read(name) for name in archive.namelist()))
            archive.extractall(self.root / "extracted")
        verify_release(self.root / "extracted")

    def test_missing_input_does_not_replace_an_existing_archive(self):
        target = self.root / "release.zip"
        target.write_bytes(b"previous release")
        (self.root / "main.py").unlink()
        with self.assertRaises(ValueError):
            build_release(self.root, target)
        self.assertEqual(target.read_bytes(), b"previous release")

    def test_symlinked_input_cannot_include_runtime_secrets(self):
        (self.root / ".env").write_text("PRIVATE_RUNTIME_SENTINEL")
        (self.root / "main.py").unlink()
        (self.root / "main.py").symlink_to(self.root / ".env")
        with self.assertRaises(ValueError):
            build_release(self.root, self.root / "release.zip")

    def test_manifest_detects_source_changes(self):
        build_release(self.root, self.root / "release.zip")
        with zipfile.ZipFile(self.root / "release.zip") as archive:
            archive.extractall(self.root / "extracted")
        (self.root / "extracted" / "main.py").write_text("changed source")
        with self.assertRaises(ValueError):
            verify_release(self.root / "extracted")

    def test_manifest_rejects_paths_outside_the_allowlist(self):
        (self.root / "release-manifest.json").write_text(json.dumps({"format": 1, "files": {"../.env": "digest"}}))
        with self.assertRaises(ValueError):
            verify_release(self.root)

    def test_output_must_be_an_archive(self):
        with self.assertRaises(ValueError):
            build_release(self.root, self.root / "main.py")


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        old_umask = os.umask(0o077)
        os.umask(old_umask)
        self.addCleanup(os.umask, old_umask)

    def test_serve_forces_one_worker_and_disables_reload_and_proxy_trust(self):
        with patch.object(run, "ROOT", self.root), patch.object(run, "preflight") as check, patch("uvicorn.run") as server, patch.dict(os.environ, {"WEB_CONCURRENCY": "99"}):
            run.serve(port=8002)
        check.assert_called_once()
        options = server.call_args.kwargs
        self.assertEqual(options["workers"], 1)
        self.assertFalse(options["reload"])
        self.assertFalse(options["proxy_headers"])
        self.assertFalse(options["access_log"])
        self.assertEqual(options["host"], "127.0.0.1")
        self.assertEqual(options["port"], 8002)

    def test_second_instance_is_rejected_and_lock_releases_after_exit(self):
        with patch.object(run, "ROOT", self.root), patch.object(run, "preflight") as check, patch("uvicorn.run") as server:
            with portalocker.Lock(str(self.root / ".rag-server.lock"), timeout=0):
                with self.assertRaisesRegex(run.PreflightError, "deployment_instance_already_running"):
                    run.serve()
                check.assert_not_called()
                server.assert_not_called()
            run.serve()
            server.assert_called_once()

    def test_preflight_configuration_failures_are_sanitized(self):
        import main
        with patch.object(run, "check_dependencies"), patch.object(run, "prepare_environment"), patch("deploy.package.verify_release"), patch.object(main, "local_dependencies_ready", side_effect=RuntimeError("PRIVATE_CONFIGURATION_DETAILS")):
            with self.assertRaises(run.PreflightError) as failure:
                run.preflight(self.root)
        self.assertEqual(str(failure.exception), "application_configuration_or_storage_failed")

    def test_missing_local_dependencies_fail_preflight(self):
        import main
        with patch.object(run, "check_dependencies"), patch.object(run, "prepare_environment"), patch("deploy.package.verify_release"), patch.object(main, "local_dependencies_ready", return_value=False):
            with self.assertRaisesRegex(run.PreflightError, "local_dependencies_not_ready"):
                run.preflight(self.root)

    def test_ready_preflight_probes_audit_storage_without_calling_provider(self):
        import main
        import llm
        with patch.object(run, "check_dependencies"), patch.object(run, "prepare_environment"), patch("deploy.package.verify_release"), patch.object(main, "local_dependencies_ready", return_value=True), patch.object(llm.client.chat.completions, "create") as provider:
            report = run.preflight(self.root)
        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["provider_calls"], 0)
        self.assertTrue((self.root / "logs").is_dir())
        self.assertEqual(list((self.root / "logs").iterdir()), [])
        provider.assert_not_called()


if __name__ == "__main__":
    unittest.main()
