from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from medical_accessibility.shapley_analysis import run_shapley_tasks  # noqa: E402


DEFAULT_SCENARIO_MAP = {
    "14road+14pop+24bed": "A001",
    "14road+24pop+14bed": "A010",
    "24road+14pop+14bed": "A100",
    "24road+14pop+24bed": "A101",
    "14road+24pop+24bed": "A011",
    "24road+24pop+14bed": "A110",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the public Shapley decomposition workflow.")
    parser.add_argument("--config-json", type=Path, default=PROJECT_ROOT / "data" / "sample" / "shapley" / "tasks.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "shapley")
    return parser.parse_args()


def main():
    args = parse_args()
    tasks = json.loads(args.config_json.read_text(encoding="utf-8"))
    run_shapley_tasks(tasks, DEFAULT_SCENARIO_MAP, args.output_dir)


if __name__ == "__main__":
    main()

