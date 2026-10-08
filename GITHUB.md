# GitHub repository

Repository: https://github.com/mdsaeed24/rag-platform

The public source includes the Python backend, Streamlit interface, tests,
evaluation scripts/input cases, fictional Acme/Globex demo documents, and the
aggregate `FINAL_RESULTS.md`. Demo credentials and expected salary values are
intentional fixtures, not real employee records. Provider keys, JWT secrets,
virtual environments, model caches, vectors, audit logs, build output, and raw
evaluation results are excluded. Generated results remain available in the
original development workspace and can be regenerated locally.

## First-time setup from a fresh clone

The native backend lock currently targets Python 3.14/macOS ARM64. For another
backend platform, resolve and validate dependencies separately; the interface
has its own dependency file.

```sh
git clone https://github.com/mdsaeed24/rag-platform.git
cd rag-platform
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
cp .env.example .env
chmod 600 .env
```

Edit `.env` with your DeepSeek key and a random JWT secret of at least 32
characters. A secret can be generated locally with
`python3.14 -c 'import secrets; print(secrets.token_urlsafe(48))'`.

Provision the public embedding model and initialize demo vectors **in the new
clone only**:

```sh
.venv/bin/python -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("all-MiniLM-L6-v2", trust_remote_code=False)'
.venv/bin/python ingest.py
.venv/bin/python -m deploy.run check
```

`ingest.py` replaces an existing collection. For an existing deployment, use the
storage backup/restore procedure in `deploy/README.md` instead. Model provisioning
downloads public model files; ingestion embeds the fictional demo documents
locally and does not call DeepSeek.

Run the backend and interface in separate terminals as described in
`ui/README.md`. Submitting an answer request uses your configured provider;
running the offline checks below uses dummy credentials and mocked completions.

## Automated checks

`.github/workflows/ci.yml` runs on pushes to `main`, pull requests, and manual
dispatch. Its token has `contents: read`; checkout does not persist credentials.
Official actions are pinned to full commit IDs. No repository secrets or live
model-provider credentials are required.

The source-review job examines the indexed files for private/generated paths,
symlinks, unexpected binary/large files, and selected credential patterns. The
backend job uses Python 3.14.6 on the macOS ARM64 runner matching the native lock.
It installs exact package versions and provisions the public MiniLM model before
blocking socket connections during tests. It runs the backend regression suite,
11 deployment smoke cases, nine UI-to-FastAPI integration cases, and builds the
source archive. The separate Linux interface job runs 17 Streamlit/client tests.
Both application jobs use temporary fixtures and never need a persistent demo
vector collection. A CI pass does not establish live answer quality, production
capacity, or backend Linux compatibility.

Equivalent commands in a full checkout with the dependencies/model provisioned:

```sh
.venv/bin/python -m deploy.ci backend
.venv-ui/bin/python -m deploy.ci ui
.venv/bin/python -m deploy.repository_check --check-local-env
```

The index review checks the staged blobs, so cleaning a working file after
staging a secret does not hide the staged value. The optional local check also
compares indexed content to the three actual credentials in `.env`, without
printing them. Reports contain only counts, paths, and finding categories and
are saved under ignored `dist/`. This is a limited safeguard, not a comprehensive
secret scanner or a scan of past commits. Review `git diff --cached --stat` and
rerun the check before pushing future changes. CI tooling and tests require the
full checkout and are not included in the native runtime archive.

## Hosting

Publishing this repository does not host the API or interface. The next hosting
step needs a backend platform, validated dependencies, model/storage provisioning,
and runtime secrets. A hosted Streamlit interface must use that hosted API's
HTTPS address rather than the laptop's localhost. See `ui/README.md`.
