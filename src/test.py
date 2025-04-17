from pathlib import Path

current_dir = Path(__file__).parent
project_root = current_dir.parent

DICTIONARIES_PATH = project_root / "data" / "dictionaries"

print(DICTIONARIES_PATH)
