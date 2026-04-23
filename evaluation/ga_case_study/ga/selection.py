from copy import deepcopy
from random import sample
from typing import List, Tuple

from quapsim import QuaPSim, Circuit


class ISelection:
    n: int

    def __init__(self, n: int):
        self.n = n

    def select(self, circuits: List[Circuit], fitness_scores: List[Tuple]) -> List[Circuit]:
        """Returns a sorted list of the selected circuits."""
        raise NotImplementedError()


class RouletteSelection(ISelection):
    def select(self, circuits: List[Circuit], fitness_scores: List[Tuple]) -> List[Circuit]:
        raise NotImplementedError()


class TournamentSelection(ISelection):
    tournament_size: int

    def __init__(self, n: int, tournament_size: int):
        self.n = n
        self.tournament_size = tournament_size

    def select(self, circuits: List[Circuit], fitness_scores: List[Tuple]) -> List[Circuit]:
        assert len(circuits) == len(fitness_scores)

        selection: List[Tuple[Circuit, Tuple]] = []

        for _ in range(self.n):
            candidate_indices: List[int] = sample(
                population=range(len(circuits)), k=self.tournament_size)

            scores = [fitness_scores[i][0] for i in candidate_indices]

            winner_score = min(scores)
            winner_idx = scores.index(winner_score)

            winner = circuits[winner_idx]
            winner_fitness = fitness_scores[winner_idx]

            selection.append((deepcopy(winner), winner_fitness))

        selection.sort(key=lambda item: item[1][0])

        selection = [
            circuit for (circuit, fitness) in selection
        ]

        return selection


class NSGA2Selection(ISelection):
    def select(self, circuits: List[Circuit], fitness_scores: List[Tuple]) -> List[Circuit]:
        """Returns a sorted list of the selected circuits."""
        raise NotImplementedError()
