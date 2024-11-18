#!/usr/bin/env python3

from dataclasses import dataclass
import numpy as np
from typing import List

from .interface import IGate


class Swap(IGate):
    """Swap gate.
    Swaps the states of the two specified qubits."""

    qubit1: int
    qubit2: int

    def __init__(self, qubit1: int, qubit2: int) -> None:
        self.qubit1 = qubit1
        self.qubit2 = qubit2

    @property
    def qubits(self) -> List[int]:
        return [self.qubit1, self.qubit2]

    def __repr__(self) -> str:
        gate_name = str(type(self)).split(".")[-1].replace("'>", "")
        return f"{gate_name}(qubit1={self.qubit1}, qubit2={self.qubit2})"
