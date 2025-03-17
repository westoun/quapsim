#!/usr/bin/env python3

from datetime import datetime
import logging
import numpy as np
from random import randint
from typing import List, Dict, Callable, Iterable, Tuple

from quapsim.circuit import Circuit
from quapsim.gates import IGate, create_unitary, create_identity_matrix
from .params import SimulatorParams, DEFAULT_PARAMS
from quapsim.cache import ICache

from .utils import (
    log_duration,
    NGram,
    compute_potential_gain,
    create_merged_ngram,
    compute_redundancy,
)


class QuaPSim:
    def __init__(self, params: SimulatorParams = DEFAULT_PARAMS, cache: ICache = None):
        self.params = params
        self.cache = cache

    @log_duration
    def evaluate(self, circuits: List[Circuit], state: np.ndarray = None, set_unitary: bool = True) -> None:
        """Evaluates a list of quantum circuits. If set_unitary is 
        True, it computes the unitary of the circuit and stores said 
        unitary in circuit.unitary. If set_unitary is false, it computes
        the action performed by the circuit for a single state. If no 
        state is provided by the user, |0...0> is used as default."""

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
            self.simulate_without_cache(circuits, state, set_unitary)
        else:
            self.build_cache(circuits)
            self.simulate_using_cache(circuits, state, set_unitary)

    @log_duration
    def build_cache(self, circuits: List[Circuit]) -> None:
        logging.info(f"Starting to build cache.")

        qubit_num = circuits[0].qubit_num

        ngrams = self._generate_seed_bigrams(circuits)
        ngrams = self._consolidate_ngrams(ngrams)
        ngrams = self._select_ngrams_to_cache(ngrams)

        self._fill_cache(ngrams, qubit_num)

    @log_duration
    def _generate_seed_bigrams(
        self,
        circuits: List[Circuit],
    ) -> List[NGram]:
        bigrams: Dict = {}
        for document_id, circuit in enumerate(circuits):
            for location in range(len(circuit.gates) - 1):
                gate = circuit.gates[location]
                successor_gate = circuit.gates[location + 1]

                key = f"{gate.__repr__()}_{successor_gate.__repr__()}"

                if key in bigrams:
                    bigrams[key].add_location(document_id, location)
                    bigrams[key].frequency += 1
                    bigrams[key].frequency_potential += 1
                else:
                    bigrams[key] = NGram(
                        gates=[gate, successor_gate], frequency=1)
                    bigrams[key].add_location(document_id, location)

        bigrams: List[NGram] = list(bigrams.values())
        return bigrams

    @log_duration
    def _consolidate_ngrams(
        self,
        ngrams: List[NGram],
    ) -> List[NGram]:

        logging.debug(f"Starting ngram generation with {len(ngrams)} bigrams.")

        ngrams.sort(key=lambda ngram: ngram.gain_potential, reverse=True)

        start_gate_dict = {}
        end_gate_dict = {}

        # Because ngrams have been sorted before, the original entries of each
        # dict are automatically sorted.
        for ngram in ngrams:
            if ngram.gain_potential < 1:
                break

            start_gate = ngram.gates[0]
            if start_gate in start_gate_dict:
                start_gate_dict[start_gate].append(ngram)
            else:
                start_gate_dict[start_gate] = [ngram]

            end_gate = ngram.gates[-1]
            if end_gate in end_gate_dict:
                end_gate_dict[end_gate].append(ngram)
            else:
                end_gate_dict[end_gate] = [ngram]

        checked_ngrams = set()
        for i in range(self.params.merging_rounds):

            top_ngram = None
            for ngram in ngrams:
                if ngram.gain_potential < 1:
                    break

                if ngram not in checked_ngrams:
                    top_ngram = ngram
                    break

            # No more ngrams to check
            if top_ngram is None:
                logging.debug(
                    f"Breaking ngram generation since no more potential gains."
                )
                break

            start_gate = top_ngram.gates[0]
            end_gate = top_ngram.gates[-1]

            if start_gate in end_gate_dict:
                left_candidates: List[NGram] = end_gate_dict[start_gate]

                for left_candidate in left_candidates:
                    potential_gain = compute_potential_gain(
                        left_candidate, top_ngram)

                    # Since end_gate_dict only includes bigrams sorted by frequency,
                    # any further bigrams cannot have a higher potential gain than the
                    # current one.
                    if potential_gain < 1:
                        break

                    new_ngram = create_merged_ngram(left_candidate, top_ngram)
                    ngrams.append(new_ngram)

                    logging.debug(
                        f"Adding {new_ngram.gates} with frequency {new_ngram.frequency} to ngram pool."
                    )

                    left_candidate.frequency_potential -= new_ngram.frequency
                    top_ngram.frequency_potential -= new_ngram.frequency

                end_gate_dict[start_gate].sort(
                    key=lambda ngram: ngram.frequency_potential, reverse=True)

            if end_gate in start_gate_dict:
                right_candidates: List[NGram] = start_gate_dict[end_gate]

                for right_candidate in right_candidates:
                    potential_gain = compute_potential_gain(
                        top_ngram, right_candidate)

                    # Since end_gate_dict only includes bigrams sorted by frequency,
                    # any further bigrams cannot have a higher potential gain than the
                    # current one.
                    if potential_gain < 1:
                        break

                    new_ngram = create_merged_ngram(top_ngram, right_candidate)
                    ngrams.append(new_ngram)

                    logging.debug(
                        f"Adding {new_ngram.gates} with frequency {new_ngram.frequency} to ngram pool."
                    )

                    right_candidate.frequency_potential -= new_ngram.frequency
                    top_ngram.frequency_potential -= new_ngram.frequency

                start_gate_dict[end_gate].sort(
                    key=lambda ngram: ngram.frequency_potential, reverse=True)

            checked_ngrams.add(top_ngram)

            ngrams.sort(
                key=lambda ngram: ngram.gain_potential, reverse=True)

        return ngrams

    @log_duration
    def _select_ngrams_to_cache(self, ngrams: List[NGram]) -> List[NGram]:
        ngrams = [ngram for ngram in ngrams if ngram.frequency > 1]
        ngrams.sort(key=lambda ngram: ngram.gain, reverse=True)
        return ngrams[: self.params.cache_size]

    @log_duration
    def _fill_cache(self, ngrams: List[NGram], qubit_num: int) -> None:
        # Add shorter ngrams first to cache, so that later ngrams can use
        # them.
        ngrams.sort(key=lambda ngram: len(ngram.gates))

        for ngram in ngrams:
            ngram_unitary = None

            i = 0
            while True:
                if i >= len(ngram.gates):
                    break

                cache_window = self.cache.get_prefix_in_cache_length(
                    ngram.gates[i:])

                if cache_window == 0:
                    unitary = create_unitary(
                        ngram.gates[i], qubit_num=qubit_num
                    )

                    if ngram_unitary is None:
                        ngram_unitary = unitary
                    else:
                        ngram_unitary = np.matmul(unitary, ngram_unitary)

                    i = i + 1

                else:
                    cached_unitary = self.cache.get(
                        ngram.gates[i: i + cache_window])

                    if ngram_unitary is None:
                        ngram_unitary = cached_unitary
                    else:
                        ngram_unitary = np.matmul(
                            cached_unitary, ngram_unitary)

                    i = i + cache_window

            logging.debug(
                f"Adding {ngram.gates} (with a frequency of {ngram.frequency}) to cache."
            )
            self.cache.add(ngram.gates, ngram_unitary)

    @log_duration
    def simulate_using_cache(self, circuits: List[Circuit], state: np.ndarray = None, set_unitary: bool = False) -> None:
        if not set_unitary:
            logging.info(
                f"Starting to simulate using the cache, mode: set_state.")
            self._simulate_using_cache_set_state(circuits, state)
        else:
            logging.info(
                f"Starting to simulate using the cache, mode: set_unitary. Ignoring any provided states.")
            self._simulate_using_cache_set_unitary(circuits)

    def _simulate_using_cache_set_state(self, circuits: List[Circuit], state: np.ndarray = None) -> None:
        lookup_duration = datetime.now() - datetime.now()

        for circuit in circuits:

            circuit_state = self._init_circuit_state(
                qubit_num=circuit.qubit_num, default_state=state)

            i = 0
            while True:
                if i >= len(circuit.gates):
                    break

                start = datetime.now()
                cache_window = self.cache.get_prefix_in_cache_length(
                    circuit.gates[i:])
                lookup_duration += datetime.now() - start

                if cache_window == 0:
                    unitary = create_unitary(
                        circuit.gates[i], qubit_num=circuit.qubit_num
                    )
                    circuit_state = np.matmul(unitary, circuit_state)

                    i = i + 1

                else:
                    cached_unitary = self.cache.get(
                        circuit.gates[i: i + cache_window])

                    circuit_state = np.matmul(cached_unitary, circuit_state)

                    logging.debug(
                        f"Using {circuit.gates[i: i + cache_window]} from cache."
                    )

                    i = i + cache_window

            circuit.set_state(circuit_state)

        logging.info(
            f"Time during merging spent on trie lookup: {lookup_duration}")

    def _simulate_using_cache_set_unitary(self, circuits: List[Circuit]) -> None:
        lookup_duration = datetime.now() - datetime.now()

        for circuit in circuits:
            circuit_unitary = create_identity_matrix(dim=2**circuit.qubit_num)

            i = 0
            while True:
                if i >= len(circuit.gates):
                    break

                start = datetime.now()
                cache_window = self.cache.get_prefix_in_cache_length(
                    circuit.gates[i:])
                lookup_duration += datetime.now() - start

                if cache_window == 0:
                    unitary = create_unitary(
                        circuit.gates[i], qubit_num=circuit.qubit_num
                    )
                    circuit_unitary = np.matmul(unitary, circuit_unitary)

                    i = i + 1

                else:
                    cached_unitary = self.cache.get(
                        circuit.gates[i: i + cache_window])

                    circuit_unitary = np.matmul(
                        cached_unitary, circuit_unitary)

                    logging.debug(
                        f"Using {circuit.gates[i: i + cache_window]} from cache."
                    )

                    i = i + cache_window

            circuit.set_unitary(circuit_unitary)

        logging.info(
            f"Time during merging spent on trie lookup: {lookup_duration}")

    @log_duration
    def simulate_without_cache(self, circuits: List[Circuit], state: np.ndarray = None, set_unitary: bool = False) -> None:
        if not set_unitary:
            logging.info(
                f"Starting to simulate without using the cache, mode: set_state.")
            self._simulate_without_cache_set_state(circuits, state)
        else:
            logging.info(
                f"Starting to simulate without using the cache, mode: set_unitary. Ignoring any provided states.")
            self._simulate_without_cache_set_unitary(circuits)

    def _simulate_without_cache_set_state(self, circuits: List[Circuit], state: np.ndarray = None) -> None:
        for circuit in circuits:
            circuit_state = self._init_circuit_state(
                qubit_num=circuit.qubit_num, default_state=state)

            for gate in circuit.gates:
                unitary = create_unitary(
                    gate, qubit_num=circuit.qubit_num
                )
                circuit_state = np.matmul(unitary, circuit_state)

            circuit.set_state(circuit_state)

    def _init_circuit_state(self, qubit_num: int, default_state: np.ndarray = None) -> np.ndarray:
        if default_state is not None:
            return default_state.copy()

        state = np.zeros(
            2**qubit_num, dtype=np.complex128)
        state[0] = 1
        return state

    def _simulate_without_cache_set_unitary(self, circuits: List[Circuit]) -> None:
        for circuit in circuits:
            circuit_unitary = create_identity_matrix(dim=2**circuit.qubit_num)

            for gate in circuit.gates:
                unitary = create_unitary(
                    gate, qubit_num=circuit.qubit_num
                )
                circuit_unitary = np.matmul(unitary, circuit_unitary)

            circuit.set_unitary(circuit_unitary)
