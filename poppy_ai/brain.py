from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class BrainNet(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Brain:
    model: BrainNet

    def act(self, frame: np.ndarray, device: str) -> int:
        flat = torch.tensor(frame.flatten(), dtype=torch.float32, device=device).unsqueeze(0) / 255.0
        with torch.no_grad():
            logits = self.model(flat)
        return int(torch.argmax(logits, dim=1).item())

    def clone(self) -> "Brain":
        cloned = BrainNet(self.model.net[0].in_features, self.model.net[-1].out_features)
        cloned.load_state_dict(self.model.state_dict())
        return Brain(model=cloned)

    def mutate(self, sigma: float) -> None:
        for p in self.model.parameters():
            noise = torch.randn_like(p) * sigma
            p.data.add_(noise)
