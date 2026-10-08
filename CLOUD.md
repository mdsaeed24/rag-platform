# Free demo backend and Streamlit Community Cloud

Source: <https://github.com/mdsaeed24/rag-platform>. The frontend and backend
are separate services. No live cloud service has been created by these files.

## Backend preparation

The root Dockerfile builds a Linux/Python 3.14 CPU image, downloads the public
MiniLM model during build, and creates a fresh index from the four fictional
Acme/Globex documents already published in this repository. It never copies the
developer's `.env`, vector store, model cache, or audit logs. The Docker context
uses an explicit allowlist. Changing the demo documents requires rebuilding.

The container resolves Linux dependencies from the direct version pins and
generates an image-specific dependency closure with the existing lock tool.
The macOS lock in Git remains unchanged. Transitive versions, model downloads,
and base-image tags are not hash locked, so rebuilds can differ. The CI validation
target runs backend regression tests and isolated deployment/UI integration
smokes with mocked completions and blocked sockets. The runtime target runs as
UID 1000 and keeps the existing preflight, one-worker guard, offline model mode,
bounded concurrency, and disabled HTTP access logging.

GitHub's `linux-container` check builds both targets and starts the actual runtime
with networking disabled, dummy credentials, and a 512 MB memory limit. It checks
readiness, login, and employee salary abstention without calling DeepSeek. Check
that this job passes for the commit you deploy; this is a narrow resource smoke,
not a capacity or sustained-load guarantee.

The native release ZIP excludes the source fixtures and Docker deployment files.
Build the demo container from a fresh repository checkout, not that ZIP.

## Create the free backend

1. Create a [Render account](https://dashboard.render.com/register), connecting
   your GitHub account if desired. Select a free workspace/instance; no paid disk
   is needed for this fixed demo.
2. Create a Blueprint from `mdsaeed24/rag-platform`, branch `main`. Render reads
   `render.yaml`, which explicitly requests `plan: free`, Docker, and the
   `/health/ready` health check. Check that the review screen still shows Free.
3. Enter `DEEPSEEK_API_KEY` in Render's secret environment setting yourself.
   Render generates `JWT_SECRET_KEY`. Do not paste either into GitHub or the
   Streamlit frontend. Builds require no provider credentials; runtime preflight
   requires them. Verify the provider's spending controls before sharing the demo.
4. Deploy and wait for `/health/ready` on the assigned HTTPS `onrender.com` URL to
   return HTTP 200. The API is exposed on the host's `PORT` with one worker and
   one simultaneous ask/login slot each. Auto-deployment is disabled so updates
   can be checked before deployment.
5. Keep the assigned base URL for Streamlit. Health checks and login do not call
   the answer provider. User questions that generate answers do incur provider
   charges even when web hosting is free.

Public demo credentials remain visible in the interface and source; anyone can
use them. This deployment is for the fictional portfolio corpus. Never put real
employee documents into it. Process concurrency limits bound simultaneous work,
but do not impose a per-user quota or a total provider spending cap.

[Render's free services](https://render.com/docs/free) sleep after 15 idle minutes
and can take about a minute to wake. Runtime file changes are lost on sleep,
restart, and redeployment. The image restores its baked demo index; answer caches
and local audit logs reset. This does not provide durable production storage.
If the memory smoke or hosted startup fails, do not silently choose a paid plan;
the deployment needs a smaller runtime or a separately agreed hosting option.

## Connect Streamlit

In [Streamlit Community Cloud](https://share.streamlit.io), create an app with:

| Setting | Value |
| --- | --- |
| Repository | `mdsaeed24/rag-platform` |
| Branch | `main` |
| Main file path | `ui/cloud_app.py` |
| Python | 3.14, if offered |

In Advanced settings → Secrets, enter only the backend address:

```toml
RAG_API_URL = "https://YOUR-ACTUAL-BACKEND.onrender.com"
```

Streamlit [searches the entry point directory for dependencies first](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies),
so this path installs `ui/requirements.txt` instead of the backend dependencies.
The cloud entry point requires a public HTTPS backend address and does not fall
back to localhost. Deploy, then use **Check connection**, sign in, and inspect the
displayed tenant, role, citations, and graph evidence with the fictional accounts.

The backend URL is the only frontend configuration. Provider/JWT secrets stay
on the backend. Streamlit sessions keep tokens and recent displayed answers in
memory; sign out when done. See [the UI guide](ui/README.md) for session behavior.
