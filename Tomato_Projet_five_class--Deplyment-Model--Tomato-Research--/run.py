"""
Launches the Streamlit dashboard (dashboard_app.py).

Usage:
    python run.py

Runs "python -m streamlit run dashboard_app.py" instead of relying on the
installed "streamlit" command/exe -- on this machine the streamlit.exe
wrapper script exits immediately without starting the server or printing
any error, while "python -m streamlit" works correctly.
"""

import subprocess
import sys
from pathlib import Path

APP_FILE = Path(__file__).resolve().parent / "dashboard_app.py"


def main() -> int:
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(APP_FILE)],
        cwd=APP_FILE.parent,
    )


if __name__ == "__main__":
    sys.exit(main())
