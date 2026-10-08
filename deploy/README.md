# Native deployment package

This package targets the environment validated in this workspace: **Python 3.14
on macOS ARM64** (currently Python 3.14.6). It runs one Uvicorn worker with local
Qdrant. It does not deploy a public endpoint, container, reverse proxy, or shared
database. For the separate Linux demo container and Streamlit Cloud setup, see [CLOUD.md](../CLOUD.md). Docker is not installed locally; container validation runs in GitHub Actions.

Validation completed on this host: a clean dependency installation, 201 regression
tests, 11/11 isolated smoke cases from the extracted release, and preflight against
the existing initialized collection. A temporary
localhost HTTP server also passed liveness, readiness, authenticated employee
salary abstention, and duplicate-launcher rejection, with outbound connections
blocked. The probe server shut down afterward. No real DeepSeek calls were made.

## Validate the existing workspace

From the project root:

```sh
.venv/bin/python -m deploy.run check
.venv/bin/python -m deploy.smoke
```

The preflight checks the Python minor version, lock target, all pinned installed
versions, release manifest when present, application/ACL configuration, cached
MiniLM model, existing Qdrant collection metadata, and writable audit storage.
It never calls DeepSeek, queries document text, or rebuilds vectors. It can
create the private `logs/` directory and a temporary write probe. Failure output
uses fixed reason codes and excludes underlying exception details and secrets.

Startup forces Hugging Face offline mode and disables Hugging Face telemetry.
Provision the model cache before launch; a missing model fails preflight rather
than downloading at startup. The provider key must be supplied, but its validity
and provider availability are not tested by preflight. Existing environment
variables take precedence over values in `.env`.

The smoke test is a separate CLI process. It supplies dummy provider credentials,
blocks socket connections, creates a temporary four-point fictional corpus,
uses real cached embeddings and retrieval, and mocks completion responses. It
checks ordinary RAG, cache reuse, GraphRAG, graph-cache reuse, employee and
cross-tenant abstention, authentication, health, and disabled metrics. It never
opens the workspace's Qdrant collection or writes audit logs. Its report is
`dist/deployment-smoke.json`; mocked answers do not establish live answer quality.

## Build the release

```sh
.venv/bin/python -m deploy.package
```

This creates `dist/rag-platform.zip` and prints its SHA-256 digest. The archive
contains only explicitly allowlisted source modules, deployment tools, dependency
files, documentation (including the aggregate `FINAL_RESULTS.md`), and
`.env.example`. It excludes `.env`, model caches, virtual environments, source
documents, Qdrant storage, logs, tests, raw evaluation reports, and previous build
output. Inputs containing symlinks are rejected.

The archive also includes the optional Streamlit interface, its separate
`ui/requirements.txt`, interface guide, and example configuration. The backend
environment does not need Streamlit. Install and run the UI in `.venv-ui` as
described in `ui/README.md`. Actual `.streamlit/secrets.toml` and `.venv-ui/` are
excluded. Backend preflight verifies all packaged files, including UI source.

`release-manifest.json` records each packaged file's SHA-256. With unchanged
inputs, repeated builds on the same environment produce identical archive bytes.
Preflight checks the manifest in an extracted release. This detects accidental
changes; it is not a signature or proof of publisher authenticity.

## Install on a matching host

Use a new application directory and a dedicated OS user. Extract the archive
there, then create a virtual environment and install the version lock:

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
```

`requirements.lock` pins the 72-package installed dependency closure, including
the requested `python-jose[cryptography]` and `httpx[http2]` dependencies. It is
a version lock, not a wheel-hash lock. All 72 packages were downloaded and installed
successfully into a clean environment on the validated host. The launcher rejects
a different OS/architecture; resolve and test a new lock before targeting Linux,
Windows, another architecture, or another Python minor version.

### Fresh-install validation

On 2026-10-07, a new Python 3.14.6 virtual environment on this macOS ARM64 host
installed the exact lock using wheels only and isolated pip configuration:

```sh
python3.14 -m venv /tmp/rag-clean-venv
/tmp/rag-clean-venv/bin/python -m pip --isolated install \
  --disable-pip-version-check --cache-dir /tmp/rag-clean-pip-cache \
  --only-binary=:all: --report /tmp/rag-clean-pip-report.json \
  -r requirements.lock
/tmp/rag-clean-venv/bin/python -m pip --isolated --no-cache-dir check
```

All 72 locked distributions were freshly installed inside that environment;
system site packages were disabled and no packages loaded from the working
`.venv`. `pip check` and the launcher's exact-version/target checks passed. The
fresh interpreter also passed all 193 regression tests, the five evaluation
suites (84/84 cases), the deployment smoke (11/11), and the 300-request offline
pipeline benchmark. Stored document payload fingerprints remained unchanged.

The release archive was then extracted into a separate temporary directory.
With that directory as the working directory, the fresh interpreter passed
manifest verification, dependency checks, and all 11 isolated smoke cases:

```sh
/tmp/rag-clean-venv/bin/python -c \
  'from deploy.package import verify_release; from deploy.run import check_dependencies; verify_release("."); check_dependencies()'
/tmp/rag-clean-venv/bin/python -m deploy.smoke
```

Package downloads required network access. Application validation used dummy
credentials, blocked outbound socket connections, and made no real provider
calls. The existing Hugging Face model cache was reused. This validates a clean
dependency install on the same host; it does not validate model download, runtime
asset restore onto another host, or container deployment. The development
workspace retains a sanitized report with package versions and downloaded wheel
SHA-256 digests in `eval/fresh_install_results.json`; pip index URLs and secrets
are excluded. These recorded digests do not make the version lock hash-enforced.

Provision runtime assets separately:

1. Place `.env` in the extracted project root with the required provider key and
   a random JWT secret of at least 32 characters. Use `.env.example` as the
   configuration reference. Set file permissions to owner-only (`chmod 600 .env`).
2. Provision the cached Hugging Face repository for
   `sentence-transformers/all-MiniLM-L6-v2`. On the same host/user, the existing
   cache is reused. On another matching host, copy its complete cached repository
   directory, including `blobs`, `refs`, and `snapshots`, into the destination
   Hugging Face hub cache (`HF_HUB_CACHE` if configured).
3. Stop the source RAG process before taking a consistent backup of
   `qdrant_data/`. Restore that directory into the extracted project root.
   Retain the ACL configuration matching those stored payloads. Do not share
   one local Qdrant directory between simultaneously running application copies.
4. Restore `data/` separately if future ingestion is needed. Serving existing
   vectors does not read source documents. Ingestion replaces the collection;
   the deployment launcher never invokes it.

Then run:

```sh
.venv/bin/python -m deploy.run check
.venv/bin/python -m deploy.smoke
.venv/bin/python -m deploy.run serve
```

The default address is `127.0.0.1:8001`. Verify `GET /health/live` and
`GET /health/ready`. `/metrics` remains disabled until an independent
`METRICS_TOKEN` is configured. To select another address explicitly:

```sh
.venv/bin/python -m deploy.run serve --host 127.0.0.1 --port 8002
```

The launcher fixes `workers=1` and `reload=False`, including when
`WEB_CONCURRENCY` is set. A process-lifetime `.rag-server.lock` rejects a second
launcher in the same project directory. The empty lock file can remain after
shutdown; the OS lock is released on exit. Direct Uvicorn commands bypass this
cooperative guard, so use the launcher consistently.

The launcher also disables forwarded-header trust, HTTP access logs, and the
server-identification header; caps Uvicorn concurrency at 32; uses a five-second
keep-alive timeout and a 45-second graceful shutdown allowance; and sets a
private file-creation umask. Ask/login admission controls remain configurable
as documented in the main README. HTTP access logs are disabled to avoid logging
query strings; the application audit trail and metrics remain active.

Use Ctrl-C or SIGTERM for shutdown. Before switching releases, stop the old
process, back up state, provision the new directory, run preflight, and launch
one instance. A previous source release can be restored with the corresponding
runtime configuration and storage backup. No database migration, service manager,
automatic restart policy, or automatic rollback is installed by this package.

## Updating the lock

After intentionally updating and testing dependencies on the target platform:

```sh
.venv/bin/python -m deploy.lock_dependencies
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m deploy.smoke
.venv/bin/python -m deploy.package
```

The lock generator records the installed dependency closure and the hash of
`requirements.txt`. It validates installed versions against declared constraints;
it does not resolve or install new dependencies. Keep the original dependency
declarations and lock together. Rebuild the archive after any packaged source
or documentation change.
