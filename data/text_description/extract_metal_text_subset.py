"""Extract a small dataset-description subset from the full JSON file.

This script keeps only the selected dataset keys and writes them into a new
JSON file for experiments that focus on metal-related datasets.
"""

import json
from pathlib import Path


SOURCE_PATH = Path("data/text_description/datasets_des_info_gpt4o_v2.json")
TARGET_PATH = Path("data/text_description/datasets_des_info_metal_part.json")
SELECTED_KEYS = [
    "casting_billet",
    "steel_pipe",
    "KolektorSDD",
    "KolektorSDD2",
]


def main():
    with SOURCE_PATH.open("r", encoding="utf-8") as f:
        all_data = json.load(f)

    missing_keys = [key for key in SELECTED_KEYS if key not in all_data]
    if missing_keys:
        raise KeyError(f"Missing keys in source JSON: {missing_keys}")

    subset = {key: all_data[key] for key in SELECTED_KEYS}

    with TARGET_PATH.open("w", encoding="utf-8") as f:
        json.dump(subset, f, indent=4, ensure_ascii=False)
        f.write("\n")

    print(f"Saved {len(subset)} entries to {TARGET_PATH}")
    print("Keys:", ", ".join(subset.keys()))


if __name__ == "__main__":
    main()
