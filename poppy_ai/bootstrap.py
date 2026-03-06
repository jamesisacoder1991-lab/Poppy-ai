from __future__ import annotations

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader

from poppy_ai.brain import Brain
from poppy_ai.config import TrainingConfig
from poppy_ai.dataset import DemonstrationDataset


class BootstrapTrainer:
    """Trains an initial policy from human demonstrations before evolution."""

    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg

    def train_from_demos(self, brain: Brain) -> Brain | None:
        dataset = DemonstrationDataset(
            frames_dir=self.cfg.demo_frames_dir,
            actions_file=self.cfg.demo_actions_file,
            actions=self.cfg.actions,
            width=self.cfg.frame_width,
            height=self.cfg.frame_height,
        )

        if len(dataset) == 0:
            return None

        loader = DataLoader(dataset, batch_size=self.cfg.bootstrap_batch_size, shuffle=True)
        model = copy.deepcopy(brain.model).to(self.cfg.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.bootstrap_learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(1, self.cfg.bootstrap_epochs + 1):
            epoch_loss = 0.0
            for x, y in loader:
                x = x.to(self.cfg.device)
                y = y.to(self.cfg.device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())

            avg_loss = epoch_loss / max(len(loader), 1)
            print(f"Bootstrap epoch {epoch}/{self.cfg.bootstrap_epochs} | loss={avg_loss:.4f}")

        return Brain(model=model.cpu())
