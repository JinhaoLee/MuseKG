"""Fail if a proposed public release contains common private artifacts or secrets."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MAX_PUBLIC_FILE_BYTES = 5 * 1024 * 1024

BLOCKED_PATH_PATTERNS = (
    re.compile(r"^\.env(?:\.|$)"),
    re.compile(r"^data/(?!README\.md$)"),
    re.compile(r"^demo/static/images/"),
    re.compile(r"^demo/static/image_mapping\.json$"),
    re.compile(r"^(?:output|logs|triton|huggingface|\.cache)/"),
    re.compile(r"\.(?:gexf|zip|out|err|log|pdf|jpe?g|png|mp4)$", re.IGNORECASE),
)

CONTENT_PATTERNS = {
    "Google API key": re.compile(rb"AIza[0-9A-Za-z_-]{30,}"),
    "OpenAI-style API key": re.compile(rb"sk-[0-9A-Za-z_-]{20,}"),
    "private key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "private cluster path": re.compile(rb"/data/" rb"gpfs/projects/"),
    "private home path": re.compile(rb"/home/[A-Za-z0-9._-]+/"),
}


def candidate_files() -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [REPOSITORY_ROOT / item for item in result.stdout.splitlines() if item]


def main() -> int:
    problems: list[str] = []
    paths = candidate_files()
    for path in paths:
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        if any(pattern.search(relative) for pattern in BLOCKED_PATH_PATTERNS):
            problems.append(f"blocked public path: {relative}")
            continue
        if not path.is_file():
            continue
        if path.stat().st_size > MAX_PUBLIC_FILE_BYTES:
            problems.append(f"file exceeds 5 MiB: {relative}")
            continue
        content = path.read_bytes()
        for description, pattern in CONTENT_PATTERNS.items():
            if pattern.search(content):
                problems.append(f"{description} found in {relative}")

    if problems:
        print("Publication safety check failed:")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print(f"Publication safety check passed for {len(paths)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
