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


class NGram:
    gates: List[IGate]
    frequency: int

    def __init__(self, gates: List[IGate], frequency: int):
        self.gates = gates
        self.frequency = frequency

    @property
    def gain(self) -> int:
        return (len(self.gates) - 1) * (self.frequency - 1)


def compute_potential_gain(first_ngram: NGram, second_ngram: NGram) -> int:
    if first_ngram.gates[-1] != second_ngram.gates[0]:
        return 0

    if first_ngram.frequency <= 1:
        return 0

    if second_ngram.frequency <= 1:
        return 0

    potential_gain = (len(first_ngram.gates) + len(second_ngram.gates) - 1) * (
        min(first_ngram.frequency, second_ngram.frequency) - 1
    )
    return potential_gain


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
            circuits,
            gate_frequency_dict,
            inverted_gate_index,
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
            if frequency <= 1:
                break

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

    @log_duration
    def _build_ngram_frequency_dict(
        self,
        circuits: List[Circuit],
        gate_frequency_dict: GateFrequencyDict,
        inverted_gate_index: InvertedGateIndex,
    ) -> NgramFrequencyDict:
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

        inverted_bigrams = {}
        for gate in bigrams:
            for successor_gate in bigrams[gate]:
                frequency = bigrams[gate][successor_gate]

                if frequency in inverted_bigrams:
                    inverted_bigrams[frequency].append((gate, successor_gate))
                else:
                    inverted_bigrams[frequency] = [(gate, successor_gate)]

        bigram_frequencies = list(inverted_bigrams.keys())
        bigram_frequencies.sort(reverse=True)

        ngrams: List[NGram] = []
        for potential_frequency in bigram_frequencies:
            if potential_frequency <= 1:
                break

            current_bigrams = inverted_bigrams[potential_frequency]

            # Padding factor
            padding_factor = 3
            if (len(ngrams) + len(current_bigrams)) < (
                self.params.cache_size * padding_factor
            ):
                for gate, successor_gate in current_bigrams:
                    ngram_frequency = calculate_gate_sequence_frequency(
                        gate_sequence=[gate, successor_gate],
                        inverted_index=inverted_gate_index,
                    )
                    ngram = NGram(
                        gates=[gate, successor_gate], frequency=ngram_frequency
                    )
                    ngrams.append(ngram)

            else:
                for gate, successor_gate in current_bigrams[
                    : self.params.cache_size * padding_factor - len(ngrams)
                ]:
                    ngram_frequency = calculate_gate_sequence_frequency(
                        gate_sequence=[gate, successor_gate],
                        inverted_index=inverted_gate_index,
                    )
                    ngram = NGram(
                        gates=[gate, successor_gate], frequency=ngram_frequency
                    )
                    ngrams.append(ngram)

                break

        logging.debug(f"Starting ngram generation with {len(ngrams)} bigrams.")

        potential_gains: np.ndarray = np.zeros((len(ngrams), len(ngrams)), dtype=int)
        for i, first_ngram in enumerate(ngrams):
            for j, second_ngram in enumerate(ngrams):
                if first_ngram.gates[-1] == second_ngram.gates[0]:
                    potential_gain = compute_potential_gain(first_ngram, second_ngram)
                    potential_gains[i, j] = potential_gain

        merging_round = 0
        while True:

            merging_round += 1

            if merging_round >= 100:
                logging.debug(
                    f"Breaking ngram generation after {merging_round} rounds of merging."
                )
                break

            first_ngram_idx, second_ngram_idx = np.unravel_index(
                np.argmax(potential_gains, axis=None), potential_gains.shape
            )
            if potential_gains[first_ngram_idx, second_ngram_idx] < 1:
                logging.debug(
                    f"Breaking ngram generation since no more potential gains."
                )
                break

            first_ngram = ngrams[first_ngram_idx]
            second_ngram = ngrams[second_ngram_idx]

            gate_sequence = []
            gate_sequence.extend(first_ngram.gates)
            gate_sequence.extend(second_ngram.gates[1:])

            ngram_frequency = calculate_gate_sequence_frequency(
                gate_sequence=gate_sequence,
                inverted_index=inverted_gate_index,
            )

            new_ngram = NGram(gates=gate_sequence, frequency=ngram_frequency)

            ngrams.append(new_ngram)
            logging.debug(
                f"Adding {new_ngram.gates} with frequency {ngram_frequency} to ngram pool."
            )

            # Update potential gains array

            # Avoid evaluating the same ngram pair again
            potential_gains[first_ngram_idx, second_ngram_idx] = 0

            # Add new rows and columns to gains arr
            new_col = np.zeros((len(ngrams) - 1, 1), dtype=int)
            potential_gains = np.hstack([potential_gains, new_col])

            new_row = np.zeros((len(ngrams) - 1 + 1), dtype=int)
            potential_gains = np.vstack([potential_gains, new_row])

            # No need to compute potential gain or update
            # affected rows and columns.
            if ngram_frequency == 0:
                continue

            first_ngram.frequency -= ngram_frequency
            second_ngram.frequency -= ngram_frequency

            for i, ngram in enumerate(ngrams):
                if ngram.gates[-1] == new_ngram.gates[0]:
                    potential_gain = compute_potential_gain(ngram, new_ngram)
                    potential_gains[i, len(ngrams) - 1] = potential_gain

                if new_ngram.gates[-1] == ngram.gates[0]:
                    potential_gain = compute_potential_gain(ngram, new_ngram)
                    potential_gains[len(ngrams) - 1, i] = potential_gain

                # Else: keep potential gain as 0

            # Reevaluate the rows and columns that contained one of
            # the affected ngrams and have potential gain > 0.
            row = potential_gains[first_ngram_idx, :]
            for column_idx in np.where(row > 0)[0]:  # np.where returns a tuple.
                potential_gain = compute_potential_gain(first_ngram, ngrams[column_idx])
                potential_gains[first_ngram_idx, column_idx]

            column = potential_gains[:, first_ngram_idx]
            for row_idx in np.where(column > 0)[0]:
                potential_gain = compute_potential_gain(ngrams[row_idx], first_ngram)
                potential_gains[row_idx, first_ngram_idx]

            row = potential_gains[second_ngram_idx, :]
            for column_idx in np.where(row > 0)[0]:
                potential_gain = compute_potential_gain(
                    second_ngram, ngrams[column_idx]
                )
                potential_gains[second_ngram_idx, column_idx]

            column = potential_gains[:, second_ngram_idx]
            for row_idx in np.where(column > 0)[0]:
                potential_gain = compute_potential_gain(ngrams[row_idx], second_ngram)
                potential_gains[row_idx, second_ngram_idx]

        ngram_frequency_dict = NgramFrequencyDict()

        ngrams.sort(key=lambda ngram: ngram.gain, reverse=True)
        for ngram in ngrams[: self.params.cache_size]:
            if len(ngram.gates) < 2:
                continue

            if ngram.frequency <= 1:
                continue

            ngram_frequency_dict.add(ngram.gates, ngram.frequency)

        return ngram_frequency_dict

    @log_duration
    def simulate_using_cache(self, circuits: List[Circuit]) -> None:
        logging.info(f"Starting to simulate using the cache.")

        cache_entry_lengths: List[int] = list(self.cache.lengths)
        cache_entry_lengths.sort(reverse=True)

        for circuit in circuits:
            if circuit.state is not None:
                continue

            state = np.zeros(2**circuit.qubit_num, dtype=np.complex128)
            state[0] = 1

            i = 0
            while True:
                if i >= len(circuit.gates):
                    break

                for cache_window in cache_entry_lengths:
                    if i + cache_window > len(circuit.gates):
                        continue

                    gate_sequence = circuit.gates[i : i + cache_window]

                    cached_unitary = self.cache.get(gate_sequence)

                    if cached_unitary is not None:
                        logging.debug(f"Using {gate_sequence} from cache.")
                        state = np.matmul(cached_unitary, state)

                        i = i + cache_window
                        break

                else:
                    unitary = create_unitary(
                        circuit.gates[i], qubit_num=circuit.qubit_num
                    )
                    state = np.matmul(unitary, state)
                    i += 1

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
