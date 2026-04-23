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

        selection = []

        for _ in range(self.n):
            candidate_indices: List[int] = sample(
                population=range(len(circuits)), k=self.tournament_size)

            scores = [fitness_scores[i][0] for i in candidate_indices]

            winner_score = min(scores)
            winner_idx = fitness_scores.index(winner_score)

            winner = circuits[winner_idx]
            selection.append(deepcopy(winner))

        return selection


class NSGA2Selection(ISelection):
    def select(self, circuits: List[Circuit], fitness_scores: List[Tuple]) -> List[Circuit]:
        """Returns a sorted list of the selected circuits."""
        raise NotImplementedError()
