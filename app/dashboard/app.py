"""
Streamlit dashboard entrypoint with namespace collision guard.
"""
import sys
from pathlib import Path

# Clean script directory from sys.path and ensure PROJECT_ROOT is at sys.path[0]
# to prevent namespace collisions (e.g. app/ package vs app.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
script_dir = str(Path(__file__).resolve().parent)
while script_dir in sys.path:
    sys.path.remove(script_dir)

root_str = str(PROJECT_ROOT)
while root_str in sys.path:
    sys.path.remove(root_str)
sys.path.insert(0, root_str)

if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

for mod in list(sys.modules.keys()):
    if mod.startswith("src.") or mod.startswith("app.dashboard.components"):
        del sys.modules[mod]

from app.dashboard.main import main
if __name__ == "__main__":
    main()
