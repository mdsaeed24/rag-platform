"""Build a deterministic source archive from an explicit file allowlist."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RELEASE_FILES = (
    ".env.example", ".gitignore", "README.md", "GITHUB.md", "FINAL_RESULTS.md", "requirements.txt", "requirements.lock",
    "acl_config.py", "audit.py", "auth.py", "authorization.py", "graph_rag.py",
    "ingest.py", "llm.py", "main.py", "operations.py", "query_router.py", "search.py", "semantic_cache.py",
    "deploy/__init__.py", "deploy/README.md", "deploy/lock_dependencies.py",
    "deploy/package.py", "deploy/run.py", "deploy/smoke.py",
    "streamlit_app.py", "ui/__init__.py", "ui/api.py", "ui/session.py",
    "ui/requirements.txt", "ui/README.md", "ui/integration_smoke.py",
    ".streamlit/config.toml", ".streamlit/secrets.toml.example",
)


def release_contents(root):
    root = Path(root).resolve()
    contents = {}
    for name in RELEASE_FILES:
        path = root / name
        if (not path.is_file() or not path.resolve().is_relative_to(root)
                or any(part.is_symlink() for part in (path, *path.parents) if part != root and part.is_relative_to(root))):
            raise ValueError("Release input missing or contains a symlink")
        contents[name] = path.read_bytes()
    return contents


def build_release(root, output):
    output = Path(output)
    if output.suffix != ".zip":
        raise ValueError("Release output must be a .zip file")
    contents = release_contents(root)
    manifest = {"format": 1, "files": {name: hashlib.sha256(data).hexdigest()
                                        for name, data in sorted(contents.items())}}
    contents["release-manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".zip", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(contents.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, data)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(output.read_bytes()).hexdigest()


def verify_release(root):
    root = Path(root)
    manifest_path = root / "release-manifest.json"
    if not manifest_path.exists():
        return  # Working source tree, rather than an extracted release.
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != 1 or set(manifest.get("files", {})) != set(RELEASE_FILES):
        raise ValueError("Invalid release manifest")
    contents = release_contents(root)
    if any(hashlib.sha256(data).hexdigest() != manifest["files"][name] for name, data in contents.items()):
        raise ValueError("Release files differ from manifest")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "rag-platform.zip")
    args = parser.parse_args()
    digest = build_release(ROOT, args.output)
    print(f"Created {args.output}\nSHA256: {digest}")
