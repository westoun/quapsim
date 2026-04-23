from copy import deepcopy
from random import sample, choices
from typing import List, Tuple

from quapsim import QuaPSim, Circuit


def get_best_circuit(circuits: List[Circuit], fitness_scores: List[Tuple]) -> Tuple[Circuit, Tuple]:
    distance_scores = [
        scores[0] for scores in fitness_scores
    ]
    min_distance = min(distance_scores)
    min_distance_idx = distance_scores.index(min_distance)

    return circuits[min_distance_idx], fitness_scores[min_distance_idx]


class ISelection:
    n: int

    def __init__(self, n: int):
        self.n = n

    def select(self, circuits: List[Circuit], fitness_scores: List[Tuple]) -> Tuple[List[Circuit], List[Tuple]]:
        """Returns two lists: the selected circuits sorted by fitness and 
        the fitness scores themselves. Further, it is ensured that the best
        circuit always gets selected."""
        raise NotImplementedError()


class RouletteSelection(ISelection):
    def select(self, circuits: List[Circuit], fitness_scores: List[Tuple]) -> Tuple[List[Circuit], List[Tuple]]:

        # Must be placed before selection to avoid problems
        # with variable reuse.
        elite, elite_fitness = get_best_circuit(circuits, fitness_scores)

        # Compute relative weight for each circuit based on its fitness.
        distance_scores = [
            scores[0] for scores in fitness_scores
        ]
        max_distance = max(distance_scores)

        similarity_scores = [
            max_distance - distance for distance in distance_scores
        ]

        # Combine circuits and their fitnesses here, to simplify
        # sorting of selected solutions.
        circuit_fitness_pairings = list(zip(circuits, fitness_scores))

        selection = choices(
            circuit_fitness_pairings, weights=similarity_scores, k=self.n
        )

        selection.sort(key=lambda item: item[1][0])

        circuits, fitness_scores = zip(*selection)
        circuits, fitness_scores = list(circuits), list(fitness_scores)

        # If new best circuit is worse than the previous elite,
        # add the previous elite explicitly.
        if fitness_scores[0][0] > elite_fitness[0]:
            circuits.insert(0, elite)
            fitness_scores.insert(0, elite_fitness)

            circuits.pop()
            fitness_scores.pop()

        circuits = [deepcopy(circuit) for circuit in circuits]

        return circuits, fitness_scores


class TournamentSelection(ISelection):
    tournament_size: int

    def __init__(self, n: int, tournament_size: int):
        self.n = n
        self.tournament_size = tournament_size

    def select(self, circuits: List[Circuit], fitness_scores: List[Tuple]) -> Tuple[List[Circuit], List[Tuple]]:
        assert len(circuits) == len(fitness_scores)

        # Must be placed before selection to avoid problems
        # with variable reuse.
        elite, elite_fitness = get_best_circuit(circuits, fitness_scores)

        distance_scores = [
            scores[0] for scores in fitness_scores
        ]

        selection: List[Tuple[Circuit, Tuple]] = []

        for _ in range(self.n):
            candidate_indices: List[int] = sample(
                population=range(len(circuits)), k=self.tournament_size)

            candidate_scores = [distance_scores[i] for i in candidate_indices]

            winner_score = min(candidate_scores)
            winner_idx = distance_scores.index(winner_score)

            winner = circuits[winner_idx]
            winner_fitness = fitness_scores[winner_idx]

            selection.append((winner, winner_fitness))

        selection.sort(key=lambda item: item[1][0])

        circuits, fitness_scores = zip(*selection)
        circuits, fitness_scores = list(circuits), list(fitness_scores)

        # If new best circuit is worse than the previous elite,
        # add the previous elite explicitly.
        if fitness_scores[0][0] > elite_fitness[0]:
            circuits.insert(0, elite)
            fitness_scores.insert(0, elite_fitness)

            circuits.pop()
            fitness_scores.pop()

        circuits = [deepcopy(circuit) for circuit in circuits]

        return circuits, fitness_scores


class NSGA2Selection(ISelection):
    def select(self, circuits: List[Circuit], fitness_scores: List[Tuple]) -> Tuple[List[Circuit], List[Tuple]]:
        """Returns a sorted list of the selected circuits."""
        raise NotImplementedError()
