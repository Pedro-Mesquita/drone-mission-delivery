#!/usr/bin/env bash
# Starts the simulated reference track: Gazebo, ArduPilot SITL and the UAV API,
# with the UAV API's rpicam-still replaced by the Gazebo camera (sim/bin).
# Then run the mission against it, e.g.:
#   python -m mission_delivery.mission --target 28 0 --base-url http://localhost:8020
#
# Environment (defaults fit the team machine):
#   ARDUPILOT_DIR       ~/ardupilot
#   ARDUPILOT_GAZEBO    ~/ardupilot_gazebo
#   MAVPROXY_VENV       ~/venv-ardupilot      (MAVProxy; its python3 must import gz and cv2)
#   UAV_API_PYTHON      python with uav_api installed
#   UAV_API_CONFIG      UAV API config file (port 8020, udpin 127.0.0.1:17171)
#   GUI=1               open the Gazebo window instead of running headless
#   LOG_DIR             where the three logs go (default: sim/logs)
set -euo pipefail

SIM_DIR="$(cd "$(dirname "$0")" && pwd)"
ARDUPILOT_DIR="${ARDUPILOT_DIR:-$HOME/ardupilot}"
ARDUPILOT_GAZEBO="${ARDUPILOT_GAZEBO:-$HOME/ardupilot_gazebo}"
MAVPROXY_VENV="${MAVPROXY_VENV:-$HOME/venv-ardupilot}"
UAV_API_PYTHON="${UAV_API_PYTHON:?set UAV_API_PYTHON to a python with uav_api installed}"
UAV_API_CONFIG="${UAV_API_CONFIG:?set UAV_API_CONFIG to the UAV API config file}"
LOG_DIR="${LOG_DIR:-$SIM_DIR/logs}"
mkdir -p "$LOG_DIR"

export GZ_SIM_SYSTEM_PLUGIN_PATH="$ARDUPILOT_GAZEBO/build${GZ_SIM_SYSTEM_PLUGIN_PATH:+:$GZ_SIM_SYSTEM_PLUGIN_PATH}"
export GZ_SIM_RESOURCE_PATH="$SIM_DIR/models:$SIM_DIR/worlds:$ARDUPILOT_GAZEBO/models:$ARDUPILOT_GAZEBO/worlds${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"

pids=()
cleanup() {
  # gz sim only shuts down cleanly on SIGINT.
  kill -INT "${pids[@]}" 2>/dev/null || true
  # sim_vehicle.py runs the SITL binary in its own xterm, which outlives the script.
  pkill -f "$ARDUPILOT_DIR/build/sitl/bin/arducopter" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [[ "${GUI:-0}" == "1" ]]; then
  gz sim -r "$SIM_DIR/worlds/cartolinas_track.sdf" > "$LOG_DIR/gz.log" 2>&1 &
else
  gz sim -s -r --headless-rendering "$SIM_DIR/worlds/cartolinas_track.sdf" > "$LOG_DIR/gz.log" 2>&1 &
fi
pids+=($!)
until gz topic -l 2>/dev/null | grep -q "^/mission/cam$"; do sleep 1; done
echo "gazebo: camera on /mission/cam"

(cd "$LOG_DIR" && PATH="$MAVPROXY_VENV/bin:$PATH" \
  "$ARDUPILOT_DIR/Tools/autotest/sim_vehicle.py" -v ArduCopter -f gazebo-iris --model JSON -N \
  --no-extra-ports --out 127.0.0.1:17171 --mavproxy-args="--daemon --non-interactive" \
  > "$LOG_DIR/sitl.log" 2>&1) &
pids+=($!)
until grep -q "ArduPilot Ready" "$LOG_DIR/sitl.log" 2>/dev/null; do sleep 1; done
echo "sitl: ready"

PATH="$SIM_DIR/bin:$PATH" SIM_PYTHON="$MAVPROXY_VENV/bin/python3" "$UAV_API_PYTHON" -c \
  "import sys; sys.argv=['uav-api','start','--config','$UAV_API_CONFIG']; \
from uav_api.run_api import main; sys.exit(main())" > "$LOG_DIR/uav_api.log" 2>&1 &
pids+=($!)
echo "uav api: starting (logs in $LOG_DIR); Ctrl+C stops everything"
wait
