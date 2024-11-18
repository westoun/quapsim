#!/usr/bin/env python3

import numpy as np
from typing import List, Dict

from quapsim.circuit import Circuit
from quapsim.gates import IGate, Swap, Gate, CGate, CCGate, create_unitary
from .params import SimulatorParams, DEFAULT_PARAMS
from quapsim.cache import ICache


class QuaPSim:
    def __init__(self, params: SimulatorParams = DEFAULT_PARAMS, cache: ICache = None):
        self.params = params
        self.cache = cache

    def evaluate(self, circuits: List[Circuit]) -> None:
        """Evaluates a list of quantum circuits and stores the
        state at the end of each circuit in circuit.state."""

        if len(circuits) == 0:
            return

        qubit_counts = [circuit.qubit_num for circuit in circuits]
        assert len(set(qubit_counts)) == 1, (
            "The current version of QuaPSim only "
            "supports a population of circuits in which all circuits have the "
            "same amount of qubits. As a work around, we recommend you add an "
            "extra qubits as padding until all circuits have the same qubit count."
        )

        if self.cache is None:
            self._simulate_without_cache(circuits)
        else:
            self._build_cache(circuits)
            self._simulate_using_cache(circuits)

    def _build_cache(self, circuits: List[Circuit]) -> None:
        # create gate count dict
        gate_counts = {}
        for circuit in circuits:
            for gate in circuit.gates:
                if gate in gate_counts:
                    gate_counts[gate] += 1
                else:
                    gate_counts[gate] = 1

        # permute circuits according to counts

        # build inverted index

        # until stopping condition is met, select n+1 gram with
        # highest potential

    def _simulate_using_cache(self, circuits: List[Circuit]) -> None:
        pass

    def _simulate_without_cache(self, circuits: List[Circuit]) -> None:
        for circuit in circuits:
            if circuit.state is not None:
                continue

            state = np.zeros(2**circuit.qubit_num, dtype=np.complex128)
            state[0] = 1

            state = np.matmul(circuit.unitary, state)

            circuit.set_state(state)
