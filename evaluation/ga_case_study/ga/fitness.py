import numpy as np
from typing import List, Tuple

from quapsim import Circuit
from quapsim.gates import Identity


def compute_absolute_distance(unitary1: np.ndarray, unitary2: np.ndarray) -> float:
    rows1, cols1 = unitary1.shape
    rows2, cols2 = unitary2.shape

    assert rows1 == rows2 and cols1 and cols2, "Unitary matrices must have same shape."

    distance = 0.0
    for row in range(rows1):
        for col in range(cols1):

            distance += abs(unitary1[row][col] - unitary2[row][col])

    return distance


def compute_length(circuit: Circuit) -> int:
    length = 0

    for gate in circuit.gates:
        if type(gate) != Identity:
            length += 1

    return length


class Fitness:
    target_unitary: np.ndarray

    def __init__(self, target_unitary: np.ndarray):
        self.target_unitary = target_unitary

    def score(self, circuits: List[Circuit]) -> List[Tuple]:
        fitness_scores = []

        for circuit in circuits:

            absolute_distance = compute_absolute_distance(
                self.target_unitary, circuit.unitary
            )

            length = compute_length(circuit)

            fitness_scores.append((absolute_distance, length))

        return fitness_scores
