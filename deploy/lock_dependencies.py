"""Lock the installed dependency closure, including requested extras."""

from collections import deque
from importlib.metadata import distribution
from pathlib import Path
import platform
import hashlib
import json

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]


def installed_closure(requirements, *, lookup=distribution):
    pending = deque(Requirement(line) for line in requirements
                    if line.strip() and not line.lstrip().startswith("#"))
    selected = {}
    expanded = {}
    while pending:
        requirement = pending.popleft()
        if requirement.url:
            raise ValueError("Direct URL requirements are unsupported")
        name = canonicalize_name(requirement.name)
        package = lookup(name)
        if not requirement.specifier.contains(package.version, prereleases=True):
            raise ValueError(f"Installed version does not satisfy {name}")
        selected[name] = package.version
        extras = {"", *requirement.extras}
        new_extras = extras - expanded.get(name, set())
        if not new_extras:
            continue
        expanded.setdefault(name, set()).update(new_extras)
        for text in package.requires or []:
            dependency = Requirement(text)
            if dependency.marker is None or any(dependency.marker.evaluate({"extra": extra}) for extra in new_extras):
                pending.append(dependency)
    return selected


def write_lock():
    selected = installed_closure((ROOT / "requirements.txt").read_text().splitlines())
    header = ("# Installed dependency closure; regenerate with python -m deploy.lock_dependencies\n"
              f"# Validated target: Python {platform.python_version()} / {platform.system()} / {platform.machine()}\n"
              f"# Target: {json.dumps({'python': '3.14', 'system': platform.system(), 'machine': platform.machine()}, sort_keys=True)}\n"
              f"# Requirements-SHA256: {hashlib.sha256((ROOT / 'requirements.txt').read_bytes()).hexdigest()}\n"
              "# Version pins only; this is not a wheel-hash lock or a cross-platform resolution.\n")
    (ROOT / "requirements.lock").write_text(header + "".join(f"{name}=={version}\n" for name, version in sorted(selected.items())))
    return len(selected)


if __name__ == "__main__":
    print(f"Pinned {write_lock()} runtime packages in requirements.lock")
