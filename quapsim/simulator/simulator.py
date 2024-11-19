#!/usr/bin/env python3

import logging
import numpy as np
from typing import List, Dict, Callable, Iterable

from quapsim.circuit import Circuit
from quapsim.gates import IGate, create_unitary
from .params import SimulatorParams, DEFAULT_PARAMS
from quapsim.cache import ICache

from .utils import (
    GateFrequencyDict,
    InvertedGateFrequencyDict,
    InvertedGateIndex,
    calculate_gate_sequence_frequency,
    NgramFrequencyDict,
    InvertedNgramFrequencyDict,
    log_duration,
)


class QuaPSim:
    def __init__(self, params: SimulatorParams = DEFAULT_PARAMS, cache: ICache = None):
        self.params = params
        self.cache = cache

    @log_duration
    def evaluate(self, circuits: List[Circuit]) -> None:
        """Evaluates a list of quantum circuits and stores the
        state at the end of each circuit in circuit.state."""

        logging.info(f"Starting to evaluate {len(circuits)} circuits with params {self.params} and cache of type {type(self.cache)}.")

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

    @log_duration
    def _build_cache(self, circuits: List[Circuit]) -> None:
        qubit_num = circuits[0].qubit_num

        gate_frequency_dict = GateFrequencyDict().index(circuits)
        inverted_gate_frequency_dict = gate_frequency_dict.invert()

        # TODO: permute circuits according to counts

        inverted_gate_index = InvertedGateIndex().index(circuits)

        ngram_frequency_dict = self._build_ngram_frequency_dict(
            inverted_gate_frequency_dict, inverted_gate_index
        )

        # TODO: Get all ngrams relevant based on cache size
        # while ensuring that within the same cache size,
        # shorter gate sequences are added first. (needed for
        # retrieval logic)

        inverse_ngram_frequency_dict = ngram_frequency_dict.invert()
        self._fill_cache(inverse_ngram_frequency_dict, qubit_num)

    @log_duration
    def _fill_cache(
        self, inverse_ngram_frequency_dict: InvertedNgramFrequencyDict, qubit_num: int
    ) -> None:
        ngram_frequencies: List[int] = inverse_ngram_frequency_dict.frequencies
        ngram_frequencies.sort(reverse=True)

        cached_unitaries = 0
        for frequency in ngram_frequencies:
            # If ngram only occurrs once, there is no gain in caching it.
            if frequency == 1:
                break

            gate_sequences: List[List[IGate]] = inverse_ngram_frequency_dict[frequency]

            if cached_unitaries + len(gate_sequences) < self.params.cache_size:

                for gate_sequence in gate_sequences:
                    unitary = create_unitary(gate_sequence, qubit_num)
                    self.cache.add(gate_sequence, unitary)

                cached_unitaries += len(gate_sequences)

            else:
                cache_size_left = self.params.cache_size - cached_unitaries

                # Sort gate sequences in current front by ngram size from
                # smallest to largest. While it would be preferrable to
                # cache longer sequences first, as these sequences yield a
                # bigger safe in matrix operations, the current simulation
                # logic is built on the assumption that if an n-gram has been
                # cached, its n-1-gram has also been cached.
                gate_sequences.sort(key=lambda sequence: len(sequence))

                for gate_sequence in gate_sequences[:cache_size_left]:
                    unitary = create_unitary(gate_sequence, qubit_num)
                    self.cache.add(gate_sequence, unitary)

                break

    @log_duration
    def _build_ngram_frequency_dict(
        self,
        inverted_gate_frequency_dict: InvertedGateFrequencyDict,
        inverted_gate_index: InvertedGateIndex,
    ) -> NgramFrequencyDict:
        gate_frequencies: List[int] = inverted_gate_frequency_dict.frequencies
        gate_frequencies.sort(reverse=False)

        ngram_frequency_dict = NgramFrequencyDict()
        while len(gate_frequencies) > 0:

            front_threshold = gate_frequencies.pop()

            # If a gate only occurrs once, there is no
            # gain in caching it.
            if front_threshold <= 1:
                break

            if (
                len(
                    [
                        ngram
                        for ngram in ngram_frequency_dict.ngrams
                        if ngram_frequency_dict.get_frequency(ngram) >= front_threshold
                    ]
                )
                >= self.params.cache_size
            ):
                break

            start_candidate_sequences: List[List[IGate]] = []
            if front_threshold in inverted_gate_frequency_dict:
                for gate in inverted_gate_frequency_dict[front_threshold]:
                    start_candidate_sequences.append([gate])

            for ngram in ngram_frequency_dict.ngrams:
                if ngram_frequency_dict.get_frequency(ngram) >= front_threshold:
                    start_candidate_sequences.append(
                        ngram_frequency_dict.get_gates(ngram)
                    )

            expansion_candidates: List[IGate] = []
            for frequency in inverted_gate_frequency_dict.frequencies:
                if frequency >= front_threshold:
                    expansion_candidates.extend(inverted_gate_frequency_dict[frequency])

            for start_candidate_sequence in start_candidate_sequences:
                for expansion_candidate in expansion_candidates:

                    if start_candidate_sequence[0] == expansion_candidate:
                        continue

                    gate_sequence: List[IGate] = []
                    gate_sequence.extend(start_candidate_sequence)
                    gate_sequence.append(expansion_candidate)

                    frequency = calculate_gate_sequence_frequency(
                        gate_sequence=gate_sequence, inverted_index=inverted_gate_index
                    )

                    ngram_frequency_dict.add(gate_sequence, frequency)

                    # Add max check to avoid adding the frequency that has just been
                    # popped.
                    if (
                        frequency < max(gate_frequencies)
                        and frequency not in gate_frequencies
                    ):
                        gate_frequencies.append(frequency)
                        gate_frequencies.sort(reverse=False)

        return ngram_frequency_dict

    @log_duration
    def _simulate_using_cache(self, circuits: List[Circuit]) -> None:
        for circuit in circuits:
            if circuit.state is not None:
                continue

            state = np.zeros(2**circuit.qubit_num, dtype=np.complex128)
            state[0] = 1

            current_gate_sequence = []
            for gate in circuit.gates:
                current_gate_sequence.append(gate)

                # Since cache does not contain single gate
                # sequences, checking them does not suffice.
                if len(current_gate_sequence) == 1:
                    continue

                cache_value = self.cache.get(current_gate_sequence)

                # Case: gate sequence is in cache
                if cache_value is not None:
                    continue

                if len(current_gate_sequence) == 2:
                    unitary = create_unitary(
                        current_gate_sequence[0], qubit_num=circuit.qubit_num
                    )
                    state = np.matmul(unitary, state)

                    current_gate_sequence = current_gate_sequence[1:]

                else:
                    unitary = self.cache.get(current_gate_sequence[:-1])
                    state = np.matmul(unitary, state)

                    current_gate_sequence = current_gate_sequence[-1:]

            if len(current_gate_sequence) == 0:
                pass  # do nothing.

            if len(current_gate_sequence) == 1:
                unitary = create_unitary(
                    current_gate_sequence[0], qubit_num=circuit.qubit_num
                )
                state = np.matmul(unitary, state)

            elif len(current_gate_sequence) == 2:
                unitary = create_unitary(
                    current_gate_sequence, qubit_num=circuit.qubit_num
                )
                state = np.matmul(unitary, state)

            else:
                unitary = self.cache.get(current_gate_sequence[:-1])
                state = np.matmul(unitary, state)

                unitary = create_unitary(
                    current_gate_sequence[-1], qubit_num=circuit.qubit_num
                )
                state = np.matmul(unitary, state)

            circuit.set_state(state)

    @log_duration
    def _simulate_without_cache(self, circuits: List[Circuit]) -> None:
        for circuit in circuits:
            if circuit.state is not None:
                continue

            state = np.zeros(2**circuit.qubit_num, dtype=np.complex128)
            state[0] = 1

            state = np.matmul(circuit.unitary, state)

            circuit.set_state(state)
