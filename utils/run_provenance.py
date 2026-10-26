"""Small, non-secret run provenance helpers shared by experiment writers."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DISTRIBUTIONS = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "scikit_learn": "scikit-learn",
    "pandas": "pandas",
    "neal": "dwave-neal",
}


def collect_run_provenance(device: str | None = None) -> dict:
    """Return reproducibility metadata without exposing environment variables."""
    commit = "unavailable"
    dirty = "unavailable"
    worktree_fingerprint = "unavailable"
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "HEAD", "--binary"], cwd=ROOT, check=True,
            capture_output=True,
        ).stdout
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.splitlines()
        untracked_digest = hashlib.sha256()
        for relative_path in sorted(untracked):
            candidate = (ROOT / relative_path).resolve()
            if candidate.is_file() and candidate.is_relative_to(ROOT):
                untracked_digest.update(relative_path.encode("utf-8"))
                untracked_digest.update(candidate.read_bytes())
        dirty = bool(status.strip())
        worktree_fingerprint = hashlib.sha256(
            status.encode("utf-8") + diff + untracked_digest.digest()
        ).hexdigest()
    except (OSError, subprocess.SubprocessError):
        pass

    packages = {}
    for name, distribution in PACKAGE_DISTRIBUTIONS.items():
        try:
            packages[name] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"

    return {
        "git_commit_sha": commit,
        "git_worktree_dirty": dirty,
        "git_worktree_fingerprint": worktree_fingerprint,
        "python_version": platform.python_version(),
        "os_platform": platform.platform(),
        "device": device or "unavailable",
        "package_versions": packages,
    }


def stable_run_id(protocol: str, configuration: dict, provenance: dict) -> str:
    """Build a stable, opaque identifier from protocol/configuration/provenance."""
    payload = {
        "protocol": protocol,
        "configuration": configuration,
        "git_commit_sha": provenance.get("git_commit_sha", "unavailable"),
        "git_worktree_fingerprint": provenance.get("git_worktree_fingerprint", "unavailable"),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    safe_protocol = "".join(ch.lower() if ch.isalnum() else "_" for ch in protocol).strip("_")
    return f"{safe_protocol}_{digest}"

