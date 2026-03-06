from __future__ import annotations

import time

import pydirectinput


class InputController:
    def __init__(self, actions: list[str], key_hold_s: float = 0.1) -> None:
        self.actions = actions
        self.key_hold_s = key_hold_s

    def perform(self, action_index: int) -> str:
        key = self.actions[action_index]
        pydirectinput.keyDown(key)
        time.sleep(self.key_hold_s)
        pydirectinput.keyUp(key)
        return key
