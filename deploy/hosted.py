"""Single-worker container entry point; secrets are runtime environment values."""

import json
import os
import sys

from deploy.run import PreflightError, serve


def hosted_port():
    value = os.getenv("PORT", "10000")
    if not value.isascii() or not value.isdecimal() or not 1 <= int(value) <= 65535:
        raise PreflightError("invalid_hosted_port")
    return int(value)


def main():
    try:
        serve(host="0.0.0.0", port=hosted_port())
    except PreflightError as exc:
        print(json.dumps({"status": "failed", "reason": str(exc)}), file=sys.stderr)
        return 1
    except Exception:
        print(json.dumps({"status": "failed", "reason": "deployment_startup_failed"}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
