"""Community Cloud entry point beside its lightweight requirements.txt."""

from pathlib import Path
import sys

# Match imports when Streamlit puts the entry point's directory on sys.path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.app import main

if __name__ == "__main__":
    main(cloud=True)
