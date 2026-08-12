"""Cross-platform environment bootstrap — the portable equivalent of
`make setup`.

Run with any Python >= 3.11, from the repository root:

    python scripts/setup.py            (macOS / Linux)
    py scripts/setup.py                (Windows)

Creates .venv, upgrades pip, installs the package editable with dev extras,
and prints the interpreter path to use afterwards. Idempotent: an existing
.venv is reused, not recreated.
"""

from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = REPO_ROOT / ".venv"


def venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def main() -> int:
    if sys.version_info < (3, 11):
        print(f"Python >= 3.11 required, found {sys.version.split()[0]}")
        return 1
    if not VENV_DIR.exists():
        print(f"creating venv at {VENV_DIR}")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    py = venv_python(VENV_DIR)
    if not py.exists():
        print(f"venv python not found at {py} — delete .venv and rerun")
        return 1
    for cmd in (
        [str(py), "-m", "pip", "install", "--upgrade", "pip", "--quiet"],
        [str(py), "-m", "pip", "install", "-e", ".[dev]"],
    ):
        r = subprocess.run(cmd, cwd=REPO_ROOT)
        if r.returncode:
            return r.returncode
    print("\nsetup complete. Use this interpreter:")
    print(f"  {py}")
    print("run the tests with:")
    print(f"  {py} -m pytest")
    if sys.platform == "win32":
        print(
            "\nWindows console note: repository files are UTF-8. If output "
            "shows mojibake (e.g. an em-dash rendering as three garbled "
            "characters), the console is decoding with a legacy codepage — "
            "the files are NOT corrupted. Fix the console, not the files:\n"
            "  PowerShell: [Console]::OutputEncoding = "
            "[System.Text.Encoding]::UTF8\n"
            "  cmd.exe:    chcp 65001\n"
            "or set the environment variable PYTHONUTF8=1."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
