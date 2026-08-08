"""CHEAP: rebuild final reports and figures from saved Day-1–5 artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from day5_artifacts import build_final_artifacts  # noqa: E402


def main() -> int:
    summary = build_final_artifacts(root=PROJECT_ROOT)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
