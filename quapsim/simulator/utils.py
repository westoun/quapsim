#!/usr/bin/env python3

from datetime import datetime
import logging
from typing import List, Dict, Iterable, Callable

from quapsim.circuit import Circuit
from quapsim.gates import IGate


def compute_redundancy(circuits: List[Circuit]) -> float:
    all_bigrams = count_all_bigrams(circuits)
    unique_bigrams = count_unique_bigrams(circuits)

    return 1 - (unique_bigrams - 1) / (all_bigrams - 1)


def count_all_bigrams(circuits: List[Circuit]) -> float:
    return len(circuits) * (len(circuits[0].gates) - 1)


def count_unique_bigrams(circuits: List[Circuit]) -> float:
    unique_bigrams = set()
    for circuit in circuits:
        for i in range(len(circuit.gates) - 1):
            gate = circuit.gates[i]
            succ_gate = circuit.gates[i + 1]

            bigram = f"{gate.__repr__()}_{succ_gate.__repr__()}"
            unique_bigrams.add(bigram)
    return len(unique_bigrams)


class NGram:
    gates: List[IGate]
    frequency: int
    # The amount of occurrences not yet bound in other
    # ngrams.
    frequency_potential: int
    locations: Dict

    def __init__(self, gates: List[IGate], frequency: int):
        self.gates = gates
        self.frequency = frequency
        self.frequency_potential = frequency
        self.locations = {}

    @property
    def gain(self) -> int:
        if self.frequency == 0:
            return 0

        return (len(self.gates) - 1) * (self.frequency - 1)

    @property
    def gain_potential(self) -> int:
        # The maximal gain this ngram could have when merged with
        # another bigram.
        if self.frequency_potential == 0:
            return 0

        return (len(self.gates) - 1 + 1) * (self.frequency_potential - 1)

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

    if first_ngram.frequency_potential <= 1:
        return 0

    if second_ngram.frequency_potential <= 1:
        return 0

    potential_gain = (len(first_ngram.gates) + len(second_ngram.gates) - 2) * (
        min(first_ngram.frequency_potential,
            second_ngram.frequency_potential) - 1
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
