from __future__ import annotations

from dataclasses import dataclass

from poppy_ai.brain import Brain


@dataclass
class ScoredBrain:
    brain: Brain
    score: float


class EvolutionEngine:
    def __init__(self, elite_keep: int, mutation_sigma: float) -> None:
        self.elite_keep = elite_keep
        self.mutation_sigma = mutation_sigma

    def next_generation(self, scored: list[ScoredBrain], population_size: int) -> list[Brain]:
        ranked = sorted(scored, key=lambda x: x.score, reverse=True)
        elites = [entry.brain.clone() for entry in ranked[: self.elite_keep]]

        new_population: list[Brain] = elites.copy()
        best = elites[0]
        while len(new_population) < population_size:
            child = best.clone()
            child.mutate(self.mutation_sigma)
            new_population.append(child)
        return new_population
