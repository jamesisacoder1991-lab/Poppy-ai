from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DemoRecord:
    frame_file: str
    action: str


class DemonstrationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Loads (frame, action) pairs from a local demonstration run."""

    def __init__(self, frames_dir: str | Path, actions_file: str | Path, actions: list[str], width: int, height: int) -> None:
        self.frames_dir = Path(frames_dir)
        self.actions_file = Path(actions_file)
        self.actions = actions
        self.width = width
        self.height = height
        self._records = self._load_records()

    def _load_records(self) -> list[DemoRecord]:
        if not self.actions_file.exists() or not self.frames_dir.exists():
            return []

        records: list[DemoRecord] = []
        with self.actions_file.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                frame_file = row.get("frame_file", "")
                action = row.get("action", "")
                if frame_file and action:
                    records.append(DemoRecord(frame_file=frame_file, action=action))
        return records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self._records[idx]
        frame_path = self.frames_dir / record.frame_file
        gray = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Could not read frame: {frame_path}")

        gray = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        x = torch.tensor(gray.flatten(), dtype=torch.float32) / 255.0
        y = torch.tensor(self.actions.index(record.action), dtype=torch.long)
        return x, y
