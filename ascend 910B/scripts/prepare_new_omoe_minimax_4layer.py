from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

MODEL_CONFIG = Path("/home/ma-user/work/models/minimax-m2.7/config.json")
HOTSET = Path(
    "/home/ma-user/work/new_omoe/O-MoE/expert_hotset_profiles/minimax-m2.7.json"
)
BACKUP_DIR = Path(
    "/home/ma-user/work/experiments/new_omoe_minimax_4layer_single_910b4/backups"
)
CONFIG_BACKUP = BACKUP_DIR / "config.json.original"
HOTSET_BACKUP = BACKUP_DIR / "minimax-m2.7.json.original"


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def apply() -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_BACKUP.exists():
        shutil.copy2(MODEL_CONFIG, CONFIG_BACKUP)
    if not HOTSET_BACKUP.exists():
        shutil.copy2(HOTSET, HOTSET_BACKUP)

    config = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
    config["num_hidden_layers"] = 4
    write_json(MODEL_CONFIG, config)

    hotset = json.loads(HOTSET.read_text(encoding="utf-8"))
    layer_ids = sorted(int(layer_id) for layer_id in hotset["layers"])
    if layer_ids != [0, 1, 2, 3]:
        raise ValueError(f"Expected hotset layers [0, 1, 2, 3], got {layer_ids}")
    hotset["model"] = "/home/ma-user/work/models/minimax-m2.7"
    write_json(HOTSET, hotset)
    print("applied num_hidden_layers=4")
    print(f"hotset_layers={layer_ids}")


def restore() -> None:
    if not CONFIG_BACKUP.exists() or not HOTSET_BACKUP.exists():
        raise FileNotFoundError("MiniMax config/hotset backup is missing")
    shutil.copy2(CONFIG_BACKUP, MODEL_CONFIG)
    shutil.copy2(HOTSET_BACKUP, HOTSET)
    restored = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
    print(f"restored num_hidden_layers={restored['num_hidden_layers']}")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "apply"
    if action == "apply":
        apply()
    elif action == "restore":
        restore()
    else:
        raise SystemExit(f"usage: {sys.argv[0]} [apply|restore]")
