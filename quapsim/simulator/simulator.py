#!/usr/bin/env python3

import logging
import numpy as np
from random import randint
from typing import List, Dict, Callable, Iterable, Tuple

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

        logging.info(
            f"Starting to evaluate {len(circuits)} circuits with params {self.params} and cache of type {type(self.cache)}."
        )

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
            self.simulate_without_cache(circuits)
        else:
            self.build_cache(circuits)
            self.simulate_using_cache(circuits)

    @log_duration
    def build_cache(self, circuits: List[Circuit]) -> None:
        logging.info(f"Starting to build cache.")

        qubit_num = circuits[0].qubit_num

        gate_frequency_dict = self._build_gate_frequency_dict(circuits)
        inverted_gate_frequency_dict = self._build_inverted_gate_frequency_dict(
            gate_frequency_dict
        )

        self._optimize_gate_order(circuits, gate_frequency_dict)

        inverted_gate_index = self._build_inverted_gate_index(circuits)
        ngram_frequency_dict = self._build_ngram_frequency_dict(
            circuits, gate_frequency_dict, inverted_gate_frequency_dict, inverted_gate_index
        )

        inverted_ngram_frequency_dict = self._build_inverted_ngram_frequency_dict(
            ngram_frequency_dict
        )
        self._fill_cache(inverted_ngram_frequency_dict, qubit_num)

    @log_duration
    def _build_gate_frequency_dict(self, circuits: List[Circuit]) -> GateFrequencyDict:
        return GateFrequencyDict().index(circuits)

    @log_duration
    def _build_inverted_gate_frequency_dict(
        self, gate_frequency_dict: GateFrequencyDict
    ) -> InvertedGateFrequencyDict:
        return gate_frequency_dict.invert()

    @log_duration
    def _build_inverted_gate_index(self, circuits: List[Circuit]) -> InvertedGateIndex:
        return InvertedGateIndex().index(circuits)

    @log_duration
    def _build_inverted_ngram_frequency_dict(
        self, ngram_frequency_dict: NgramFrequencyDict
    ) -> InvertedNgramFrequencyDict:
        return ngram_frequency_dict.invert()

    @log_duration
    def _optimize_gate_order(
        self, circuits: List[Circuit], gate_frequency_dict: GateFrequencyDict
    ) -> None:
        for circuit in circuits:
            for _ in range(self.params.reordering_steps):
                current_idx = randint(0, len(circuit.gates) - 1)
                current_gate = circuit.gates[current_idx]

                # determine left index of beam
                left_beam_idx = 0
                for left_idx in range(current_idx):
                    comparison_gate = circuit.gates[left_idx]

                    if (
                        len(
                            [
                                qubit
                                for qubit in current_gate.qubits
                                if qubit in comparison_gate.qubits
                            ]
                        )
                        > 0
                    ):
                        left_beam_idx = left_idx

                # determine right index of beam
                right_beam_idx = len(circuit.gates) - 1
                for j in range(len(circuit.gates) - current_idx - 1):
                    right_idx = len(circuit.gates) - 1 - j

                    comparison_gate = circuit.gates[right_idx]

                    if (
                        len(
                            [
                                qubit
                                for qubit in current_gate.qubits
                                if qubit in comparison_gate.qubits
                            ]
                        )
                        > 0
                    ):
                        right_beam_idx = right_idx + 1

                # select next highest frequ gate
                current_frequency = gate_frequency_dict[current_gate]

                min_frequency_distance = np.inf
                min_frequency_idx = None

                for i in range(left_beam_idx, right_beam_idx):
                    if i == current_idx:
                        continue

                    comparison_gate = circuit.gates[i]
                    frequency_distance = abs(
                        current_frequency - gate_frequency_dict[comparison_gate]
                    )

                    if frequency_distance < min_frequency_distance:
                        min_frequency_distance = frequency_distance
                        min_frequency_idx = i

                # update gate order
                if min_frequency_idx is None:
                    continue

                if min_frequency_idx < current_idx:
                    circuit.gates.insert(
                        min_frequency_idx + 1, circuit.gates.pop(current_idx)
                    )
                else:
                    circuit.gates.insert(
                        min_frequency_idx - 1, circuit.gates.pop(current_idx)
                    )

    @log_duration
    def _fill_cache(
        self, inverse_ngram_frequency_dict: InvertedNgramFrequencyDict, qubit_num: int
    ) -> None:
        ngram_frequencies: List[int] = inverse_ngram_frequency_dict.frequencies
        ngram_frequencies.sort(reverse=True)

        cached_unitaries = 0
        for frequency in ngram_frequencies:
            # If ngram only occurrs once or zero times (might still be investigated
            # in previous ngram generation), there is no gain in caching it.
            # if frequency <= 1:
            #     break

            gate_sequences: List[List[IGate]] = inverse_ngram_frequency_dict[frequency]

            if cached_unitaries + len(gate_sequences) < self.params.cache_size:

                for gate_sequence in gate_sequences:
                    unitary = create_unitary(gate_sequence, qubit_num)
                    logging.debug(
                        f"Adding {gate_sequence} (with a frequency of {frequency}) to cache."
                    )
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
                    logging.debug(
                        f"Adding {gate_sequence} (with a frequency of {frequency}) to cache."
                    )
                    self.cache.add(gate_sequence, unitary)

                break

    # @log_duration
    # def _build_ngram_frequency_dict(
    #     self,
    #     inverted_gate_frequency_dict: InvertedGateFrequencyDict,
    #     inverted_gate_index: InvertedGateIndex,
    # ) -> NgramFrequencyDict:
    #     gate_frequencies: List[int] = inverted_gate_frequency_dict.frequencies
    #     gate_frequencies.sort(reverse=False)

    #     ngram_frequency_dict = NgramFrequencyDict()

    #     stop_search = False
    #     while len(gate_frequencies) > 0:

    #         front_threshold = gate_frequencies.pop()

    #         # If a gate occurrs once or twice, there is no
    #         # gain in caching it, since no operations are
    #         # saved.
    #         if front_threshold <= 2:
    #             logging.debug(
    #                 f"Breaking ngram frequency dict generation. frequency threshold too low: {front_threshold}"
    #             )
    #             break

    #         if len(ngram_frequency_dict) >= self.params.cache_size:
    #             logging.debug(
    #                 (
    #                     f"Stopping construction of ngram frequency dict since it contains {len(ngram_frequency_dict)} entries "
    #                     f"with a cache size of {self.params.cache_size}."
    #                 )
    #             )
    #             stop_search = True
    #             break

    #         start_candidate_sequences: List[List[IGate]] = []
    #         if front_threshold in inverted_gate_frequency_dict:
    #             for gate in inverted_gate_frequency_dict[front_threshold]:
    #                 start_candidate_sequences.append([gate])

    #         for ngram in ngram_frequency_dict.ngrams:
    #             if ngram_frequency_dict.get_frequency(ngram) >= front_threshold:
    #                 start_candidate_sequences.append(
    #                     ngram_frequency_dict.get_gates(ngram)
    #                 )

    #         expansion_candidates: List[IGate] = []
    #         for frequency in inverted_gate_frequency_dict.frequencies:
    #             if frequency >= front_threshold:
    #                 expansion_candidates.extend(inverted_gate_frequency_dict[frequency])

    #         for start_candidate_sequence in start_candidate_sequences:
    #             for expansion_candidate in expansion_candidates:

    #                 if len(ngram_frequency_dict) >= self.params.cache_size:
    #                     logging.debug(
    #                         (
    #                             f"Stopping construction of ngram frequency dict since it contains {len(ngram_frequency_dict)} entries "
    #                             f"with a cache size of {self.params.cache_size}."
    #                         )
    #                     )
    #                     stop_search = True
    #                     break

    #                 gate_sequences: List[List[IGate]] = []

    #                 if start_candidate_sequence[-1] != expansion_candidate:
    #                     gate_sequences.append([])
    #                     gate_sequences[-1].extend(start_candidate_sequence)
    #                     gate_sequences[-1].append(expansion_candidate)

    #                 if start_candidate_sequence[0] != expansion_candidate:
    #                     gate_sequences.append([])
    #                     gate_sequences[-1].append(expansion_candidate)
    #                     gate_sequences[-1].extend(start_candidate_sequence)

    #                 for gate_sequence in gate_sequences:
    #                     frequency = calculate_gate_sequence_frequency(
    #                         gate_sequence=gate_sequence,
    #                         inverted_index=inverted_gate_index,
    #                     )

    #                     # No point in investigating it as a candidate, since
    #                     # no gain to be expected here.
    #                     if frequency <= 1:
    #                         continue

    #                     ngram_frequency_dict.add(gate_sequence, frequency)

    #                     # Add max-check to avoid adding the frequency that has just been
    #                     # popped.
    #                     if (
    #                         len(gate_frequencies) > 0
    #                         and frequency < max(gate_frequencies)
    #                         and frequency not in gate_frequencies
    #                     ):
    #                         gate_frequencies.append(frequency)
    #                         gate_frequencies.sort(reverse=False)

    #             if stop_search:
    #                 break

    #         if stop_search:
    #             break

    #     return ngram_frequency_dict

    @log_duration
    def _build_ngram_frequency_dict(
        self,
        circuits: List[Circuit],
        gate_frequency_dict: GateFrequencyDict,
        inverted_gate_frequency_dict: InvertedGateFrequencyDict,
        inverted_gate_index: InvertedGateIndex,
    ) -> NgramFrequencyDict:

        class NGram:
            gates: List[IGate]
            frequency: int
            expanded_by: List[IGate]

            def __init__(self, gates: List[IGate], frequency: int):
                self.gates = gates
                self.frequency = frequency
                self.expanded_by = []

            @property
            def gain(self) -> int:
                return (len(self.gates) - 1) * (self.frequency - 1)

            def __len__(self) -> int:
                return len(self.gates)

        # TODO: Refactor method structure and signature.
        def get_potential_gain(ngram: NGram, bigrams: Dict) -> Tuple[IGate, int]:
            last_gate = ngram.gates[-1]

            # TODO: add flag to ngram that no need to expand further.
            if last_gate not in bigrams:
                return None, 0

            max_successor_frequency = -1
            max_successor: IGate = None
            for successor in bigrams[last_gate]:
                if successor not in bigrams[last_gate]:
                    continue

                if successor in ngram.expanded_by:
                    continue

                successor_frequency = bigrams[last_gate][successor]
                if successor_frequency > max_successor_frequency:
                    max_successor_frequency = successor_frequency
                    max_successor = successor

            if max_successor is None:
                return None, 0

            potential_gain = (len(ngram) + 1 - 1) * (
                min(ngram.frequency, max_successor_frequency) - 1
            )
            return max_successor, potential_gain

        bigrams = {}
        for circuit in circuits:
            for gate, successor_gate in zip(circuit.gates[:-1], circuit.gates[1:]):
                if (
                    gate_frequency_dict[gate] < 2
                    or gate_frequency_dict[successor_gate] < 2
                ):
                    continue

                if gate in bigrams:
                    if successor_gate in bigrams[gate]:
                        bigrams[gate][successor_gate] += 1
                    else:
                        bigrams[gate][successor_gate] = 1
                else:
                    bigrams[gate] = {successor_gate: 1}

        gate_frequencies: List[int] = inverted_gate_frequency_dict.frequencies
        gate_frequencies.sort(reverse=False)

        ngrams: List[NGram] = []
        for gate_frequency in gate_frequencies:
            if gate_frequency < 2:
                break

            gates = inverted_gate_frequency_dict[gate_frequency]
            for gate in gates:
                gate_onegram = NGram([gate], gate_frequency)
                ngrams.append(gate_onegram)

        onegram_count = len(ngrams)

        def get_highest_potential_ngram(
            ngrams: List[NGram], bigrams: Dict
        ) -> Tuple[NGram, IGate]:

            highest_potential_gain = -1
            highest_ngram = None
            highest_expansion_gate = None

            for ngram in ngrams:
                expansion_gate, potential_gain = get_potential_gain(ngram, bigrams)

                if expansion_gate is None:
                    continue

                if potential_gain > highest_potential_gain:
                    highest_ngram = ngram
                    highest_expansion_gate = expansion_gate
                    highest_potential_gain = potential_gain

            return highest_ngram, highest_expansion_gate

        while True:

            ngram, expansion_gate = get_highest_potential_ngram(ngrams, bigrams)

            if ngram is None:
                break

            gate_sequence = []
            gate_sequence.extend(ngram.gates)
            gate_sequence.append(expansion_gate)

            new_ngram_frequency = calculate_gate_sequence_frequency(
                gate_sequence=gate_sequence,
                inverted_index=inverted_gate_index,
            )
            new_ngram = NGram(gates=gate_sequence, frequency=new_ngram_frequency)
            ngrams.append(new_ngram)

            ngram.expanded_by.append(expansion_gate)
            ngram.frequency -= new_ngram_frequency

            bigrams[ngram.gates[-1]][expansion_gate] -= new_ngram_frequency

            logging.debug(
                f"Adding {new_ngram.gates} with frequency {new_ngram_frequency} to ngram pool."
            )

            if len(ngrams) >= onegram_count + self.params.cache_size:
                logging.debug(f"Breaking ngram generation since cache size is met.")
                break

        ngram_frequency_dict = NgramFrequencyDict()

        for ngram in ngrams:
            if len(ngram) < 2:
                continue

            # if ngram.frequency <= 1:
            #     continue

            ngram_frequency_dict.add(ngram.gates, ngram.frequency)

        return ngram_frequency_dict

    @log_duration
    def simulate_using_cache(self, circuits: List[Circuit]) -> None:
        logging.info(f"Starting to simulate using the cache.")

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
                    logging.debug(f"Using {current_gate_sequence[:-1]} from cache.")
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
                logging.debug(f"Using {current_gate_sequence[:-1]} from cache.")
                state = np.matmul(unitary, state)

                unitary = create_unitary(
                    current_gate_sequence[-1], qubit_num=circuit.qubit_num
                )
                state = np.matmul(unitary, state)

            circuit.set_state(state)

    @log_duration
    def simulate_without_cache(self, circuits: List[Circuit]) -> None:
        logging.info(f"Starting to simulate without using the cache.")
        for circuit in circuits:
            if circuit.state is not None:
                continue

            state = np.zeros(2**circuit.qubit_num, dtype=np.complex128)
            state[0] = 1

            state = np.matmul(circuit.unitary, state)

            circuit.set_state(state)
