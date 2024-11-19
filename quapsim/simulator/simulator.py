#!/usr/bin/env python3

from datetime import datetime
import logging
import numpy as np
from typing import List, Dict, Callable

from quapsim.circuit import Circuit
from quapsim.gates import IGate, Swap, Gate, CGate, CCGate, create_unitary
from .params import SimulatorParams, DEFAULT_PARAMS
from quapsim.cache import ICache


def log_duration(func: Callable):
    def wrapper(*args, **kwargs):
        start = datetime.now()
        res = func(*args, **kwargs)
        end = datetime.now()
        duration = end - start

        logging.debug(f"Executing {func.__name__} took {duration}.")

        return res

    return wrapper


class QuaPSim:
    def __init__(self, params: SimulatorParams = DEFAULT_PARAMS, cache: ICache = None):
        logging.info(
            f"Initializing QuaPSim with params {params} and cache of type {type(cache)}"
        )

        self.params = params
        self.cache = cache

    @log_duration
    def evaluate(self, circuits: List[Circuit]) -> None:
        """Evaluates a list of quantum circuits and stores the
        state at the end of each circuit in circuit.state."""

        logging.info(f"Starting to evaluate {len(circuits)} circuits.")

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

        # create gate count dict
        gate_frequencies = {}
        for circuit in circuits:
            for gate in circuit.gates:
                if gate in gate_frequencies:
                    gate_frequencies[gate] += 1
                else:
                    gate_frequencies[gate] = 1

        inverted_gate_counts = {}
        for gate in gate_frequencies:
            count = gate_frequencies[gate]

            if count in inverted_gate_counts:
                inverted_gate_counts[count].append(gate)
            else:
                inverted_gate_counts[count] = [gate]

        # TODO: permute circuits according to counts

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

        # collect ngram frequencies for cache building
        gate_frequencies: List[int] = list(inverted_gate_counts.keys())
        gate_frequencies.sort(reverse=False)

        ngram_dict = {}
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
                        for ngram in ngram_dict.keys()
                        if ngram_dict[ngram]["frequency"] >= front_threshold
                    ]
                )
                >= self.params.cache_size
            ):
                break

            start_candidate_sequences: List[List[IGate]] = []
            if front_threshold in inverted_gate_counts:
                for gate in inverted_gate_counts[front_threshold]:
                    start_candidate_sequences.append([gate])

            for ngram in ngram_dict.keys():
                if ngram_dict[ngram]["frequency"] >= front_threshold:
                    start_candidate_sequences.append(ngram_dict[ngram]["gates"])

            expansion_candidates: List[IGate] = []
            for key in inverted_gate_counts.keys():
                if key >= front_threshold:
                    expansion_candidates.extend(inverted_gate_counts[key])

            for start_candidate_sequence in start_candidate_sequences:
                for expansion_candidate in expansion_candidates:

                    if start_candidate_sequence[0] == expansion_candidate:
                        continue

                    gate_sequence: List[IGate] = []
                    gate_sequence.extend(start_candidate_sequence)
                    gate_sequence.append(expansion_candidate)

                    frequency = self._calculate_gate_sequence_frequecy(
                        gate_sequence=gate_sequence, inverted_index=inverted_index
                    )

                    ngram = "_".join([gate.__repr__() for gate in gate_sequence])
                    ngram_dict[ngram] = {"frequency": frequency, "gates": gate_sequence}

                    # Add max check to avoid adding the frequency that has just been
                    # popped.
                    if (
                        frequency < max(gate_frequencies)
                        and frequency not in gate_frequencies
                    ):
                        gate_frequencies.append(frequency)
                        gate_frequencies.sort(reverse=False)

        # TODO: Get all ngrams relevant based on cache size
        # while ensuring that within the same cache size,
        # shorter gate sequences are added first. (needed for
        # retrieval logic)

        inverse_ngram_dict = {}
        for ngram in ngram_dict:
            frequency = ngram_dict[ngram]["frequency"]
            gate_sequence = ngram_dict[ngram]["gates"]

            if frequency in inverse_ngram_dict:
                inverse_ngram_dict[frequency].append(gate_sequence)
            else:
                inverse_ngram_dict[frequency] = [gate_sequence]

        ngram_frequencies: List[int] = list(inverse_ngram_dict.keys())
        ngram_frequencies.sort(reverse=True)

        cached_unitaries = 0
        for frequency in ngram_frequencies:
            # If ngram only occurrs once, there is no gain in caching it.
            if frequency == 1:
                break

            gate_sequences: List[List[IGate]] = inverse_ngram_dict[frequency]

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

    def _calculate_gate_sequence_frequecy(
        self, gate_sequence: List[Gate], inverted_index: Dict
    ) -> int:
        for gate in gate_sequence:
            if gate not in inverted_index:
                return 0

        document_candidates = []
        for i, gate in enumerate(gate_sequence):
            if i == 0:
                document_candidates = list(inverted_index[gate].keys())

            else:
                documents_of_word = list(inverted_index[gate].keys())
                document_candidates = [
                    document
                    for document in document_candidates
                    if document in documents_of_word
                ]

            if len(document_candidates) == 0:
                return 0

        gate_sequence_frequency = 0
        for document_candidate in document_candidates:
            # TODO: refactor variable naming
            left_pred_occurrences = inverted_index[gate_sequence[0]][document_candidate]

            for succ_word in gate_sequence[1:]:
                succ_occurrences = inverted_index[succ_word][document_candidate]

                left_succ_occurrences = [
                    occurrence
                    for occurrence in succ_occurrences
                    if (occurrence - 1) in left_pred_occurrences
                ]

                left_pred_occurrences = left_succ_occurrences

                if len(left_pred_occurrences) == 0:
                    break

            gate_sequence_frequency += len(left_pred_occurrences)

        return gate_sequence_frequency

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
