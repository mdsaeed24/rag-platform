"""Secret review must inspect indexed content, including staged-only leaks."""

from pathlib import Path
import subprocess
import tempfile
import unittest

from deploy.repository_check import blocked_path, inspect_index


class RepositoryCheckTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.git("init", "-q")

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.root), *args], check=True, capture_output=True)

    def stage(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        self.git("add", "--", name)

    def test_source_and_example_configuration_are_allowed(self):
        self.stage("app.py", "print('Demo')\n")
        self.stage(".env.example", "DEEPSEEK_API_KEY=your-deepseek-api-key\n")
        self.stage(".streamlit/secrets.toml.example", 'RAG_API_URL="http://127.0.0.1:8001"\n')
        self.assertEqual(inspect_index(self.root)["status"], "passed")

    def test_private_and_generated_paths_are_rejected(self):
        for name in (".env", ".env.production", ".streamlit/secrets.toml", "logs/audit.jsonl",
                     "qdrant_data/meta.json", ".venv-ui/config", "eval/graph_results.json", "eval/baseline_results.csv"):
            self.assertTrue(blocked_path(name), name)
        self.assertFalse(blocked_path("eval/adversarial_cases.json"))
        self.stage(".env", "key=value\n")
        self.assertEqual(inspect_index(self.root)["findings"][0]["reason"], "private_or_generated_path")

    def test_staged_secret_is_found_even_if_working_file_is_clean(self):
        key = "sk-" + "x" * 40
        self.stage("app.py", repr(key))
        (self.root / "app.py").write_text("clean working copy")
        report = inspect_index(self.root)
        self.assertEqual(report["status"], "failed")
        self.assertNotIn(key, str(report))
        self.assertEqual(report["findings"][0]["reason"], "credential_pattern")

    def test_exact_local_secret_values_are_found_without_outputting_them(self):
        value = "arbitrary-secret-" + "z" * 32
        self.stage("notes.md", "accidental value: " + value)
        report = inspect_index(self.root, private_values=(value.encode(),))
        self.assertEqual(report["findings"][0]["reason"], "local_secret_value")
        self.assertNotIn(value, str(report))

    def test_symlinks_and_empty_index_fail_review(self):
        self.assertEqual(inspect_index(self.root)["status"], "failed")
        (self.root / "shortcut").symlink_to("/etc/passwd")
        self.git("add", "shortcut")
        self.assertEqual(inspect_index(self.root)["findings"][0]["reason"], "non_regular_or_unmerged_file")


if __name__ == "__main__":
    unittest.main()
