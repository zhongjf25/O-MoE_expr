from __future__ import annotations

import json
import shutil
from pathlib import Path

HOTSET = Path(
    "/home/ma-user/work/new_omoe/O-MoE/expert_hotset_profiles/minimax-m2.7.json"
)
BACKUP = Path(
    "/home/ma-user/work/experiments/new_omoe_minimax_full_8x910b4/"
    "backups/minimax-m2.7.json.original"
)
MODEL = "/home/ma-user/work/models/minimax-m2.7"


def main() -> None:
    data = json.loads(HOTSET.read_text(encoding="utf-8"))
    layer_ids = sorted(int(layer_id) for layer_id in data["layers"])
    if int(data["num_hidden_layers"]) != 62 or layer_ids != list(range(62)):
        raise ValueError(
            "Expected a full 62-layer MiniMax hotset, got "
            f"metadata={data.get('num_hidden_layers')} layers={layer_ids}"
        )
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    if not BACKUP.exists():
        shutil.copy2(HOTSET, BACKUP)
    data["model"] = MODEL
    HOTSET.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"hotset={HOTSET}")
    print(f"model={MODEL}")
    print(f"layers={len(layer_ids)} range={layer_ids[0]}..{layer_ids[-1]}")


if __name__ == "__main__":
    main()
