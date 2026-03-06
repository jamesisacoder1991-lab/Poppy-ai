from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

import cv2
import mss
import numpy as np
from pynput import keyboard

ACTIONS = ["w", "a", "s", "d", "space"]
FPS = 8
MAX_SECONDS = 60 * 20


class ActionState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.current_action = "w"

    def set_action(self, action: str) -> None:
        with self.lock:
            self.current_action = action

    def get_action(self) -> str:
        with self.lock:
            return self.current_action


def main() -> None:
    root = Path("demos")
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    actions_file = root / "actions.csv"

    state = ActionState()

    def on_press(key: keyboard.Key | keyboard.KeyCode) -> None:
        try:
            value = key.char if isinstance(key, keyboard.KeyCode) else key.name
        except AttributeError:
            return

        if value in ACTIONS:
            state.set_action(value)
        if value == "esc":
            raise keyboard.Listener.StopException

    print("Recording demo. Use W/A/S/D/SPACE while playing. Press ESC to stop.")
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    frame_period = 1.0 / FPS
    start = time.time()
    monitor = mss.mss().monitors[1]

    with actions_file.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["frame_file", "action", "timestamp"])
        writer.writeheader()

        idx = 0
        while listener.is_alive() and (time.time() - start) < MAX_SECONDS:
            now = time.time()
            raw = np.array(mss.mss().grab(monitor))[:, :, :3]
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            frame_name = f"frame_{idx:06d}.png"
            cv2.imwrite(str(frames_dir / frame_name), gray)
            writer.writerow({"frame_file": frame_name, "action": state.get_action(), "timestamp": now})
            idx += 1
            time.sleep(frame_period)

    listener.stop()
    print(f"Saved {idx} demo frames to {frames_dir}")
    print(f"Saved actions to {actions_file}")


if __name__ == "__main__":
    main()
