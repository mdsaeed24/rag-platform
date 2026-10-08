"""Run inside a dummy-key, network-disabled CI container; no live completions."""

import json
from pathlib import Path
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def request(path, data=None, token=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    body = json.dumps(data).encode() if data is not None else None
    with urlopen(Request("http://127.0.0.1:10000" + path, data=body, headers=headers), timeout=5) as response:
        assert response.status == 200
        return json.load(response)


def main():
    deadline = time.monotonic() + 90
    while True:
        try:
            request("/health/ready")
            break
        except (URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise RuntimeError("Container readiness timed out") from None
            time.sleep(1)
    request("/health/live")
    login = request("/login", {"username": "alice", "password": "alice123"})
    answer = request("/ask", {"question": "What does the CEO earn?"}, login["access_token"])
    assert answer["route"] == "abstain" and answer["sources"] == []
    assert answer["tenant_id"] == "acme" and answer["role"] == "employee"
    peak_file = Path("/sys/fs/cgroup/memory.peak")
    peak = int(peak_file.read_text()) if peak_file.exists() else None
    print(json.dumps({"ready": True, "login": True, "employee_abstention": True,
                      "memory_peak_bytes": peak, "real_provider_calls": 0}))


if __name__ == "__main__":
    main()
