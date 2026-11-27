#!/usr/bin/env bash
# Runs on the PC, with internet and access to the puc-aerorio repositories.
# Downloads into pi/wheels/ every package the mission needs on the Raspberry
# Pi (64-bit ARM), so pi/setup.sh installs them there with no internet.
#
#   pi/pack.sh [PYTHON_VERSION]      default 3.11 (Raspberry Pi OS Bookworm)
#
# The team libraries are built from the git references in pyproject.toml.
# The model (MissionConfig.model_path) must already be in models/.
set -euo pipefail

cd "$(dirname "$0")/.."
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
python3 -m venv "$WORK/venv"
"$WORK/venv/bin/pip" install -q --upgrade pip
"$WORK/venv/bin/python" - "${1:-3.11}" <<'EOF'
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

python_version = sys.argv[1]
wheels = Path("pi/wheels").resolve()
pip = [sys.executable, "-m", "pip"]

config = Path("src/mission_delivery/config.py").read_text()
model = Path(re.search(r'model_path: str = "(.+?)"', config).group(1))
if not model.is_file():
    sys.exit(f"missing {model}: export it from the .pt (README, Raspberry Pi)")

shutil.rmtree(wheels, ignore_errors=True)
wheels.mkdir(parents=True)

# A git+ssh reference cannot be resolved on the Pi: it is built into a wheel
# here and replaced by that file.
requirements = []
for dep in tomllib.loads(Path("pyproject.toml").read_text())["project"]["dependencies"]:
    if " @ " not in dep:
        requirements.append(dep)
        continue
    name, url = dep.split(" @ ", 1)
    subprocess.run([*pip, "wheel", "-q", "--no-deps", "--wheel-dir", wheels, url], check=True)
    distribution = name.split("[")[0].replace("-", "_")
    (wheel,) = wheels.glob(f"{distribution}-*.whl")
    requirements.append(f"{name} @ {wheel.as_uri()}")
# Build backend for the editable install of this repository on the Pi.
requirements += ["hatchling", "editables"]

tag = python_version.replace(".", "")
platforms = ["manylinux_2_28_aarch64", "manylinux_2_17_aarch64", "manylinux2014_aarch64",
             "linux_aarch64"]
subprocess.run(
    [*pip, "download", "-q", "--dest", wheels, "--only-binary=:all:",
     "--implementation", "cp", "--python-version", python_version, "--abi", f"cp{tag}",
     *(arg for platform in platforms for arg in ("--platform", platform)),
     *requirements],
    check=True,
)
(wheels / "PYTHON_VERSION").write_text(python_version + "\n")
count = len(list(wheels.glob("*.whl")))
size = sum(path.stat().st_size for path in wheels.iterdir()) / 1e6
print(f"{count} wheels for Python {python_version} aarch64 in pi/wheels ({size:.0f} MB)")
print("next: copy the repository to the Pi and run pi/setup.sh there")
EOF
