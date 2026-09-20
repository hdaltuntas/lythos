#!/usr/bin/env python3
"""Run Lythos without installing it.

    python main.py                      start the interface in a browser
    python main.py run model.json       analyse a model and write a report
    python main.py mesh model.json      mesh a model and report its statistics
    python main.py examples             write the built-in example models

This file puts its own directory on the import path, so it works from a fresh
clone with nothing installed beyond NumPy, SciPy and Matplotlib.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

#: what to do when no command is named
DEFAULT_COMMAND = "gui"
COMMANDS = ("run", "mesh", "gui", "examples")

REQUIRED = {
    "numpy": "python-numpy",
    "scipy": "python-scipy",
    "matplotlib": "python-matplotlib",
}


def check_dependencies() -> list[str]:
    """Names of the packages that are needed but not importable."""
    import importlib.util

    return [name for name in REQUIRED if importlib.util.find_spec(name) is None]


def explain_missing(missing: list[str]) -> None:
    print("Lythos needs these Python packages, which are not installed:\n", file=sys.stderr)
    for name in missing:
        print(f"    {name}", file=sys.stderr)
    print("\nInstall them in a virtual environment:\n", file=sys.stderr)
    activate = "source .venv/bin/activate.fish" if _fish() else "source .venv/bin/activate"
    print(f"    python -m venv .venv\n    {activate}\n"
          f"    pip install {' '.join(missing)}\n", file=sys.stderr)
    print("Or, on a distribution that manages Python packages itself "
          "(Arch, Debian, Fedora):\n", file=sys.stderr)
    print(f"    sudo pacman -S {' '.join(REQUIRED[n] for n in missing)}"
          "        # Arch", file=sys.stderr)
    print(f"    sudo apt install {' '.join('python3-' + n for n in missing)}"
          "   # Debian, Ubuntu", file=sys.stderr)


def _fish() -> bool:
    return os.path.basename(os.environ.get("SHELL", "")) == "fish"


def main() -> int:
    if sys.version_info < (3, 10):
        print(f"Lythos needs Python 3.10 or later; this is {sys.version.split()[0]}.",
              file=sys.stderr)
        return 1

    missing = check_dependencies()
    if missing:
        explain_missing(missing)
        return 1

    from lythos.cli import main as run_cli

    # "python main.py" and "python main.py --port 9000" both mean: open the
    # interface.  Only an explicit command name changes that.
    argv = list(sys.argv[1:])
    if not argv or (argv[0] not in COMMANDS and argv[0] not in ("-h", "--help")):
        argv.insert(0, DEFAULT_COMMAND)
    return run_cli(argv)


if __name__ == "__main__":
    sys.exit(main())
