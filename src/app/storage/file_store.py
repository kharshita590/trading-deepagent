import json
from pathlib import Path

DATA_FILE = Path("state.json")

def save_state(data: dict):
    with DATA_FILE.open("w") as f:
        json.dump(data, f, indent=2)

def load_state():
    if not DATA_FILE.exists():
        return {}
    with DATA_FILE.open() as f:
        return json.load(f)
