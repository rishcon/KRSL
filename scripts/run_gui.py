"""Launch or validate the local KRSL tkinter application."""

from __future__ import annotations

import argparse
from pathlib import Path

from krsl_ai.ui.tk_app import check_runtime, main


def run() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    if args.check:
        print(check_runtime(project_root))
        return
    main(project_root)


if __name__ == "__main__":
    run()
