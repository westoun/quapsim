#!/usr/bin/env python3

from datetime import datetime
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
    log_duration,
    NGram,
    compute_potential_gain,
    create_merged_ngram,
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

        self._optimize_gate_order(circuits, gate_frequency_dict)

        ngrams = self._generate_seed_bigrams(circuits, gate_frequency_dict)
        ngrams = self._consolidate_ngrams(ngrams)
        ngrams = self._select_ngrams_to_cache(ngrams)

        self._fill_cache(ngrams, qubit_num)

    @log_duration
    def _build_gate_frequency_dict(self, circuits: List[Circuit]) -> GateFrequencyDict:
        return GateFrequencyDict().index(circuits)

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
    def _generate_seed_bigrams(
        self,
        circuits: List[Circuit],
        gate_frequency_dict: GateFrequencyDict,
    ) -> List[NGram]:
        bigrams: Dict = {}
        for document_id, circuit in enumerate(circuits):
            for location in range(len(circuit.gates) - 1):
                gate = circuit.gates[location]
                successor_gate = circuit.gates[location + 1]
                if (
                    gate_frequency_dict[gate] < 2
                    or gate_frequency_dict[successor_gate] < 2
                ):
                    continue

                key = f"{gate.__repr__()}_{successor_gate.__repr__()}"

                if key in bigrams:
                    bigrams[key].add_location(document_id, location)
                    bigrams[key].frequency += 1
                else:
                    bigrams[key] = NGram(gates=[gate, successor_gate], frequency=1)
                    bigrams[key].add_location(document_id, location)

        bigrams: List[NGram] = list(bigrams.values())
        bigrams = [bigram for bigram in bigrams if bigram.frequency > 1]
        bigrams.sort(key=lambda bigram: bigram.frequency, reverse=True)

        return bigrams[: self.params.merging_pool_size]

    @log_duration
    def _consolidate_ngrams(
        self,
        ngrams: List[NGram],
    ) -> List[NGram]:

        logging.debug(f"Starting ngram generation with {len(ngrams)} bigrams.")

        potential_gains: np.ndarray = np.zeros((len(ngrams), len(ngrams)), dtype=int)
        for i, first_ngram in enumerate(ngrams):
            for j, second_ngram in enumerate(ngrams):
                if first_ngram.gates[-1] == second_ngram.gates[0]:
                    potential_gain = compute_potential_gain(first_ngram, second_ngram)
                    potential_gains[i, j] = potential_gain

        lookup_duration = datetime.now() - datetime.now()

        merging_round = 0
        while True:

            merging_round += 1

            if merging_round >= self.params.merging_rounds:
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

            start = datetime.now()
            new_ngram = create_merged_ngram(first_ngram, second_ngram)
            lookup_duration += datetime.now() - start

            ngrams.append(new_ngram)

            logging.debug(
                f"Adding {new_ngram.gates} with frequency {new_ngram.frequency} to ngram pool."
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
            if new_ngram.frequency == 0:
                continue

            first_ngram.frequency -= new_ngram.frequency
            second_ngram.frequency -= new_ngram.frequency

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

        logging.info(
            f"Time during merging spent on frequency lookup: {lookup_duration}"
        )
        return ngrams

    @log_duration
    def _select_ngrams_to_cache(self, ngrams: List[NGram]) -> List[NGram]:
        ngrams = [ngram for ngram in ngrams if ngram.frequency > 1]
        ngrams.sort(key=lambda ngram: ngram.gain, reverse=True)
        return ngrams[: self.params.cache_size]

    @log_duration
    def _fill_cache(self, ngrams: List[NGram], qubit_num: int) -> None:
        for ngram in ngrams:
            unitary = create_unitary(ngram.gates, qubit_num)
            logging.debug(
                f"Adding {ngram.gates} (with a frequency of {ngram.frequency}) to cache."
            )
            self.cache.add(ngram.gates, unitary)

    @log_duration
    def simulate_using_cache(self, circuits: List[Circuit]) -> None:
        logging.info(f"Starting to simulate using the cache.")

        for circuit in circuits:
            if circuit.state is not None:
                continue

            state = np.zeros(2**circuit.qubit_num, dtype=np.complex128)
            state[0] = 1

            i = 0
            while True:
                if i >= len(circuit.gates):
                    break

                cache_window = self.cache.get_prefix_in_cache_length(circuit.gates[i:])

                if cache_window == 0:
                    unitary = create_unitary(
                        circuit.gates[i], qubit_num=circuit.qubit_num
                    )
                    state = np.matmul(unitary, state)

                    i = i + 1

                else:
                    cached_unitary = self.cache.get(circuit.gates[i : i + cache_window])
                    state = np.matmul(cached_unitary, state)

                    logging.debug(
                        f"Using {circuit.gates[i : i + cache_window]} from cache."
                    )

                    i = i + cache_window

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
