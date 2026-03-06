from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from poppy_ai.bootstrap import BootstrapTrainer
from poppy_ai.brain import Brain, BrainNet
from poppy_ai.capture import ScreenCapture
from poppy_ai.config import TrainingConfig
from poppy_ai.evolution import EvolutionEngine, ScoredBrain
from poppy_ai.input_controller import InputController
from poppy_ai.online_demo import OnlineDemoBuilder
from poppy_ai.reward import RewardFunction


class Trainer:
    def __init__(self, cfg: TrainingConfig) -> None:
        self.cfg = cfg
        self.capture = ScreenCapture(cfg.frame_width, cfg.frame_height)
        self.controller = InputController(cfg.actions)
        self.reward = RewardFunction()
        self.evo = EvolutionEngine(cfg.elite_keep, cfg.mutation_sigma)
        self.bootstrap = BootstrapTrainer(cfg)
        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        input_size = cfg.frame_width * cfg.frame_height
        self.population = [
            Brain(BrainNet(input_size=input_size, output_size=len(cfg.actions)))
            for _ in range(cfg.population_size)
        ]

    def _has_demo_data(self) -> bool:
        frames_dir = Path(self.cfg.demo_frames_dir)
        actions_file = Path(self.cfg.demo_actions_file)
        return frames_dir.exists() and actions_file.exists() and any(frames_dir.glob("*.png"))

    def _download_online_demo_if_needed(self) -> None:
        if self._has_demo_data() or not self.cfg.auto_download_demo:
            return

        print("No local demo found. Downloading a successful online run and building bootstrap dataset...")
        builder = OnlineDemoBuilder(
            source_url=self.cfg.demo_video_url,
            frames_dir=self.cfg.demo_frames_dir,
            actions_file=self.cfg.demo_actions_file,
            target_fps=self.cfg.capture_fps,
            frame_width=self.cfg.frame_width,
            frame_height=self.cfg.frame_height,
            actions=self.cfg.actions,
        )
        total = builder.build()
        print(f"Online demo prepared with {total} frames.")

    def _seed_population_from_demonstration(self) -> None:
        self._download_online_demo_if_needed()
        seeded = self.bootstrap.train_from_demos(self.population[0])
        if seeded is None:
            print("No demonstration dataset found; starting from random population.")
            return

        self.population[0] = seeded
        for idx in range(1, len(self.population)):
            clone = seeded.clone()
            clone.mutate(self.cfg.mutation_sigma)
            self.population[idx] = clone
        print("Seeded generation 1 from demonstrations (best brain cloned + mutated).")

    def _run_episode(self, brain: Brain) -> float:
        frame_time = 1.0 / self.cfg.capture_fps
        max_steps = self.cfg.episode_seconds * self.cfg.capture_fps
        self.reward.reset()

        prev = self.capture.grab()
        total_reward = 0.0
        for _ in range(max_steps):
            action = brain.act(prev.gray, self.cfg.device)
            self.controller.perform(action)
            time.sleep(frame_time)
            now = self.capture.grab()
            total_reward += self.reward.score(prev.gray, now.gray)
            prev = now
        return float(total_reward)

    def _save_checkpoint(self, generation: int, scored: list[ScoredBrain]) -> None:
        best = max(scored, key=lambda item: item.score)
        ckpt_path = self.save_dir / f"best_gen_{generation:05d}.pt"
        torch.save(best.brain.model.state_dict(), ckpt_path)

        metadata = {
            "generation": generation,
            "best_score": best.score,
            "population": len(scored),
            "checkpoint": ckpt_path.name,
            "timestamp": time.time(),
        }
        (self.save_dir / "latest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def train(self) -> None:
        self._seed_population_from_demonstration()
        print("Starting Poppy AI training...")
        for generation in range(1, self.cfg.generations + 1):
            scored: list[ScoredBrain] = []
            for idx, brain in enumerate(self.population, start=1):
                score = self._run_episode(brain)
                scored.append(ScoredBrain(brain=brain, score=score))
                print(f"Gen {generation} | Brain {idx}/{len(self.population)} | Score: {score:.3f}")

            self.population = self.evo.next_generation(scored, self.cfg.population_size)
            best_score = max(scored, key=lambda x: x.score).score
            print(f"Generation {generation} complete. Best score={best_score:.3f}")

            if generation % self.cfg.checkpoint_every == 0:
                self._save_checkpoint(generation, scored)
                print(f"Checkpoint saved in {self.save_dir}")
