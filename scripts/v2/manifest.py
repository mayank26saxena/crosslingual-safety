"""Write MANIFEST.sha256: sha256 of every untracked v2 input and output (prompts, replies, activations, grading folders, keys, runs).

Usage: python3 scripts/v2/manifest.py
These files hold harmful text and are not in git; the manifest lets anyone holding the archive verify they have exactly the files the results
were computed from. Tracked files are covered by git itself.
"""
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCOPE = ["data/v2", "data/external", "results/v2", "notes/hindi_calibration", "notes/grading", "notes/audit", "notes/screening"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    tracked = set(subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split())
    lines = []
    for s in SCOPE:
        for p in sorted((ROOT / s).rglob("*")):
            rel = p.relative_to(ROOT).as_posix()
            if p.is_file() and rel not in tracked and p.name != ".DS_Store" and "__pycache__" not in rel:
                lines.append(f"{sha(p)}  {rel}")
    (ROOT / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    print(len(lines), "files")


if __name__ == "__main__":
    main()
