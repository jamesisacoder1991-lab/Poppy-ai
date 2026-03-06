from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class TrainingConfig:
    window_title: str
    capture_fps: int
    frame_width: int
    frame_height: int
    population_size: int
    episode_seconds: int
    generations: int
    elite_keep: int
    mutation_sigma: float
    actions: list[str]
    save_dir: str
    checkpoint_every: int
    device: str
    demo_frames_dir: str
    demo_actions_file: str
    bootstrap_epochs: int
    bootstrap_batch_size: int
    bootstrap_learning_rate: float
    auto_download_demo: bool
    demo_video_url: str


def load_config(path: str | Path) -> TrainingConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return TrainingConfig(**data)
