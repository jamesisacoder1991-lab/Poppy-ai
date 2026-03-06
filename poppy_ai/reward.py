from __future__ import annotations

import numpy as np


class RewardFunction:
    """Simple reward with movement, novelty and survival terms."""

    def __init__(self) -> None:
        self._seen_hashes: set[int] = set()

    def reset(self) -> None:
        self._seen_hashes.clear()

    def score(self, prev_frame: np.ndarray, frame: np.ndarray, alive_bonus: float = 0.01) -> float:
        diff = np.mean(np.abs(frame.astype(np.float32) - prev_frame.astype(np.float32))) / 255.0
        frame_hash = hash(frame[::8, ::8].tobytes())
        novelty = 0.05 if frame_hash not in self._seen_hashes else 0.0
        self._seen_hashes.add(frame_hash)
        movement_reward = float(diff)
        return movement_reward + novelty + alive_bonus
