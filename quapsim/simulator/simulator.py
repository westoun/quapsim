#!/usr/bin/env python3

import numpy as np
from typing import List, Dict

from quapsim.circuit import Circuit
from quapsim.gates import IGate, Swap, Gate, CGate, CCGate
from .params import SimulatorParams, DEFAULT_PARAMS


class QuaPSim:
    def __init__(self, params: SimulatorParams = DEFAULT_PARAMS):
        self.params = params

    def evaluate(self, circuits: List[Circuit]) -> None:
        """Evaluates a list of quantum circuits and stores the
        state at the end of each circuit in circuit.state."""

        qubit_counts = [circuit.qubit_num for circuit in circuits]
        assert len(set(qubit_counts)) == 1, (
            "The current version of QuaPSim only "
            "supports a population of circuits in which all circuits have the "
            "same amount of qubits. As a work around, we recommend you add an "
            "extra qubits as padding until all circuits have the same qubit count."
        )

        raise NotImplementedError()
