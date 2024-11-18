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

        if self.cache is None or self.params.cache_size == 0:
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
        inverted_index = {}
        for i, circuit in enumerate(circuits):
            for j, gate in enumerate(circuit.gates):
                if gate in inverted_index:
                    if i in inverted_index[gate]:
                        inverted_index[gate][i].append(j)
                    else:
                        inverted_index[gate][i] = [j]
                else:
                    inverted_index[gate] = {i: [j]}

        # until stopping condition is met, select n+1 gram with
        # highest potential

    def _simulate_using_cache(self, circuits: List[Circuit]) -> None:
        for circuit in circuits:
            if circuit.state is not None:
                continue

            state = np.zeros(2**circuit.qubit_num, dtype=np.complex128)
            state[0] = 1

            current_gate_sequence = []
            for gate in circuit.gates:
                current_gate_sequence.append(gate)

                cache_value = self.cache.get(current_gate_sequence)

                # Case: gate sequence is in cache
                if cache_value is not None:
                    continue

                # Case: gate sequence is not in cache
                else:

                    # Case: no sequence starting with the current gate is
                    # in cache.
                    if len(current_gate_sequence) == 1:
                        unitary = create_unitary(
                            current_gate_sequence[0], qubit_num=circuit.qubit_num
                        )
                        state = np.matmul(unitary, state)
                        current_gate_sequence = []

                    # Case: gate sequence with all gates until the current
                    # gate is in cache. Update state with that sequence and
                    # let new sequence start with current gate.
                    else:
                        unitary = self.cache.get(current_gate_sequence[:-1])
                        state = np.matmul(unitary, state)

                        current_gate_sequence = current_gate_sequence[-1:]

            # Case: A single gate remains in cache since the
            # previous n-1 sequence has just been incorporated
            # in the state.
            if len(current_gate_sequence) == 1:
                unitary = create_unitary(
                    current_gate_sequence[0], qubit_num=circuit.qubit_num
                )
                state = np.matmul(unitary, state)

            # Case: The last gates at the end of the sequence are in
            # cache.
            elif len(current_gate_sequence) > 1:
                unitary = self.cache.get(current_gate_sequence[:-1])
                state = np.matmul(unitary, state)

            circuit.set_state(state)

    def _simulate_without_cache(self, circuits: List[Circuit]) -> None:
        for circuit in circuits:
            if circuit.state is not None:
                continue

            state = np.zeros(2**circuit.qubit_num, dtype=np.complex128)
            state[0] = 1

            state = np.matmul(circuit.unitary, state)

            circuit.set_state(state)
