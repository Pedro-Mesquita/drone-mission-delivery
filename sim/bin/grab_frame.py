"""Stands in for rpicam-still in the simulation: saves the Gazebo camera frame.

The UAV API's /peripherical/take_photo runs
    rpicam-still --width W --height H --shutter T -o OUTPUT [...]
so placing sim/bin first in the UAV API's PATH makes the mission code take
the same photo path in Gazebo as on the drone. Needs a Python with
gz.transport13, gz.msgs10 and OpenCV.
"""

import argparse
import sys
import threading

import cv2
import numpy as np
from gz.msgs10.image_pb2 import Image
from gz.transport13 import Node

TOPIC = "/mission/cam"
FRESH_FRAMES = 2  # skip the frame that may have been rendered before this call
TIMEOUT_S = 10.0


def grab() -> np.ndarray:
    node = Node()
    done = threading.Condition()
    frames: list[np.ndarray] = []

    def on_image(message: Image) -> None:
        image = np.frombuffer(message.data, np.uint8).reshape(message.height, message.width, -1)
        with done:
            frames.append(image.copy())
            done.notify_all()

    if not node.subscribe(Image, TOPIC, on_image):
        raise RuntimeError(f"cannot subscribe to {TOPIC}")
    with done:
        if not done.wait_for(lambda: len(frames) >= FRESH_FRAMES, TIMEOUT_S):
            raise TimeoutError(f"no frames on {TOPIC} (is gz sim running?)")
        return frames[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("-o", "--output", required=True)
    args, _ = parser.parse_known_args()
    rgb = grab()
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if bgr.shape[:2] != (args.height, args.width):
        bgr = cv2.resize(bgr, (args.width, args.height), interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(args.output, bgr):
        raise RuntimeError(f"could not write {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
