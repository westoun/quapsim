#!/usr/bin/env python3

from datetime import datetime
import logging
from typing import List, Dict, Iterable, Callable

from quapsim.circuit import Circuit
from quapsim.gates import IGate


class NGram:
    gates: List[IGate]
    frequency: int
    locations: Dict

    def __init__(self, gates: List[IGate], frequency: int):
        self.gates = gates
        self.frequency = frequency
        self.locations = {}

    @property
    def gain(self) -> int:
        if self.frequency == 0:
            return 0

        return (len(self.gates) - 1) * (self.frequency - 1)

    @property
    def length(self) -> int:
        return len(self.gates)

    @property
    def documents(self) -> List[int]:
        return list(self.locations.keys())

    def add_location(self, document_id: int, location: int) -> None:
        if document_id not in self.locations:
            self.locations[document_id] = [location]
        else:
            if location not in self.locations[document_id]:
                self.locations[document_id].append(location)


def create_merged_ngram(ngram1: NGram, ngram2: NGram) -> NGram:
    document_candidates = [
        document for document in ngram1.documents if document in ngram2.documents
    ]

    merged_locations = {}
    sequence_frequency = 0

    for document_id in document_candidates:
        ngram1_locations = ngram1.locations[document_id]
        ngram2_locations = ngram2.locations[document_id]

        overlapping_locations = [
            location
            for location in ngram1_locations
            if (location + ngram1.length - 1) in ngram2_locations
        ]

        merged_locations[document_id] = overlapping_locations
        sequence_frequency += len(overlapping_locations)

    combined_gates = []
    combined_gates.extend(ngram1.gates)
    combined_gates.extend(ngram2.gates[1:])

    new_ngram = NGram(gates=combined_gates, frequency=sequence_frequency)
    new_ngram.locations = merged_locations
    return new_ngram


def compute_potential_gain(first_ngram: NGram, second_ngram: NGram) -> int:
    if first_ngram.gates[-1] != second_ngram.gates[0]:
        return 0

    if first_ngram.frequency <= 1:
        return 0

    if second_ngram.frequency <= 1:
        return 0

    potential_gain = (len(first_ngram.gates) + len(second_ngram.gates) - 2) * (
        min(first_ngram.frequency, second_ngram.frequency) - 1
    )
    return potential_gain


def log_duration(func: Callable):
    def wrapper(*args, **kwargs):
        start = datetime.now()
        res = func(*args, **kwargs)
        end = datetime.now()
        duration = end - start

        logging.info(f"Executing {func.__name__} took {duration}.")

        return res

    return wrapper


class GateFrequencyDict:
    _dict: Dict

    def __init__(self):
        self._dict = {}

    @property
    def gates(self) -> List[IGate]:
        return list(self._dict.keys())

    def index(self, circuits: List[Circuit]) -> "GateFrequencyDict":
        for circuit in circuits:
            for gate in circuit.gates:
                if gate in self._dict:
                    self._dict[gate] += 1
                else:
                    self._dict[gate] = 1
        return self

    def __getitem__(self, gate: IGate) -> int:
        if gate in self._dict:
            return self._dict[gate]
        else:
            return -1


class InvertedGateIndex:
    _dict: Dict

    def __init__(self):
        self._dict = {}

    def index(self, circuits: List[Circuit]) -> "InvertedGateIndex":
        for i, circuit in enumerate(circuits):
            for j, gate in enumerate(circuit.gates):
                if gate in self._dict:
                    if i in self._dict[gate]:
                        self._dict[gate][i].append(j)
                    else:
                        self._dict[gate][i] = [j]
                else:
                    self._dict[gate] = {i: [j]}

        return self

    def __contains__(self, gate: IGate) -> bool:
        return gate in self._dict

    def get_documents(self, gate: IGate) -> List[int]:
        return self._dict[gate].keys()

    def get_locations(self, gate: IGate, document: int) -> List[int]:
        if gate not in self._dict:
            return []

        if document not in self._dict[gate]:
            return []

        return self._dict[gate][document]


def calculate_gate_sequence_frequency(
    gate_sequence: List[IGate], inverted_index: InvertedGateIndex
) -> int:
    for gate in gate_sequence:
        if gate not in inverted_index:
            return 0

    document_candidates = []
    for i, gate in enumerate(gate_sequence):
        if i == 0:
            document_candidates = inverted_index.get_documents(gate)

        else:
            documents_of_word = inverted_index.get_documents(gate)
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
        left_pred_occurrences = inverted_index.get_locations(
            gate_sequence[0], document_candidate
        )

        for succ_word in gate_sequence[1:]:
            succ_occurrences = inverted_index.get_locations(
                succ_word, document_candidate
            )

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
