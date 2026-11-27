#!/usr/bin/env bash
# Runs on the Raspberry Pi, with no internet. Creates .venv with the mission
# and its libraries from pi/wheels/ (made by pi/pack.sh on the PC) and checks
# that the model loads and runs. The UAV API is not touched.
#
#   pi/setup.sh
set -euo pipefail

cd "$(dirname "$0")/.."
WHEELS=pi/wheels
fail() { echo "setup: $*" >&2; exit 1; }

[[ -f "$WHEELS/PYTHON_VERSION" ]] || fail "no $WHEELS: run pi/pack.sh on the PC before copying"
[[ "$(uname -m)" == aarch64 ]] || fail "needs a 64-bit OS (aarch64), got $(uname -m)"
want=$(cat "$WHEELS/PYTHON_VERSION")
have=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
[[ "$have" == "$want" ]] || fail "wheels are for Python $want, this Pi has $have: run pi/pack.sh $have on the PC"

# --clear: a .venv copied from the PC is for another machine.
python3 -m venv --clear .venv || fail "python3 -m venv failed (sudo apt install python3-venv)"
PIP=".venv/bin/pip"
"$PIP" install -q --no-index --find-links "$WHEELS" "$WHEELS"/*.whl
# The repository itself, editable; its dependencies are the wheels above.
"$PIP" install -q --no-index --find-links "$WHEELS" --no-build-isolation --no-deps -e .

# The check's photos go to a throwaway log, not to flight_logs/.
AERORIO_LOG_DIR=$(mktemp -d) .venv/bin/python - <<'EOF'
import time
from pathlib import Path

import numpy as np

from drone_vision_core import ObbDetector
from mission_delivery.config import MissionConfig

model = Path(MissionConfig().model_path)
if not model.is_file():
    raise SystemExit(f"setup: model not found: {model}")
detector = ObbDetector(model)
frame = np.zeros((720, 1280, 3), dtype=np.uint8)
detector.detect(frame)
start = time.perf_counter()
detector.detect(frame)
print(f"model {model.name}: {1000 * (time.perf_counter() - start):.0f} ms per photo")
EOF

command -v rpicam-still > /dev/null || echo "warning: rpicam-still not found; /take_photo needs it"
echo "ready: .venv/bin/python -m mission_delivery.mission --target NORTH EAST"
