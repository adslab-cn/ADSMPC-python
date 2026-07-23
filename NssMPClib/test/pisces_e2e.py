"""Recommended Pisces end-to-end entry point.

This file intentionally delegates to rag.py, which contains the current
two-party NssMPClib Pisces RAG implementation. Keep this thin wrapper as the
stable command/VSCode target so protocol checks can be moved around without
changing the day-to-day run command.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("rag.py")), run_name="__main__")
