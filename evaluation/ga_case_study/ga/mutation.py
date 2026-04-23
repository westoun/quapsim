from typing import List, Type

from quapsim.gates import IGate
from .utils.random_ import random_gate


class ReplaceGateMutation():
    qubit_num: int

    def __init__(self, qubit_num: int):
        self.qubit_num = qubit_num

    def mutate(self, gate: IGate) -> IGate:
        return random_gate(self.qubit_num)
