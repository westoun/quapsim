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
    pass


class TournamentSelection(ISelection):
    tournament_size: int

    def __init__(self, n: int, tournament_size: int):
        self.n = n
        self.tournament_size = tournament_size


class NSGA2Selection(ISelection):
    pass
