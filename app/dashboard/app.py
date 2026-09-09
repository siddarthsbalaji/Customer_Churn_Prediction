"""
Streamlit dashboard entrypoint with namespace collision guard.
"""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Guard against Streamlit namespace collision where app.py occupies 'app' in sys.modules
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

from app.dashboard.main import main

if __name__ == "__main__":
    main()
