"""
Streamlit dashboard entrypoint with namespace collision guard.
"""
import sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]
for mod in list(sys.modules.keys()):
    if mod.startswith("src.") or mod.startswith("app.dashboard.components"):
        del sys.modules[mod]
from app.dashboard.main import main
if __name__ == "__main__":
    main()
