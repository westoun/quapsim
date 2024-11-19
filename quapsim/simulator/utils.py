#!/usr/bin/env python3

from datetime import datetime
import logging
from typing import List, Dict, Iterable, Callable

from quapsim.circuit import Circuit
from quapsim.gates import IGate


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

    def invert(self) -> "InvertedGateFrequencyDict":
        inverted_gate_frequency_dict = InvertedGateFrequencyDict()
        for gate in self._dict:
            if gate in self._dict:
                frequency = self._dict[gate]
            else:
                frequency = 0

            inverted_gate_frequency_dict.add(frequency, gate)
        return inverted_gate_frequency_dict

    def __getitem__(self, gate: IGate) -> int:
        if gate in self._dict:
            return self._dict[gate]
        else:
            return -1


class InvertedGateFrequencyDict:
    _dict: Dict

    def __init__(self):
        self._dict = {}

    def add(self, frequency: int, gate: IGate) -> None:
        if frequency in self._dict:
            self._dict[frequency].append(gate)
        else:
            self._dict[frequency] = [gate]

    @property
    def frequencies(self) -> List[int]:
        return list(self._dict.keys())

    def __contains__(self, frequency: int) -> bool:
        return frequency in self._dict

    def __getitem__(self, frequency: int) -> List[IGate]:
        if frequency in self._dict:
            return self._dict[frequency]
        else:
            return None


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


class NgramFrequencyDict:
    _dict: Dict
    _frequencies: List[int]

    def __init__(self):
        self._dict = {}
        self._frequencies = []

    def __contains__(self, ngram: str) -> bool:
        return ngram in self._dict

    def get_frequency(self, ngram: str) -> int:
        if ngram not in self._dict:
            return 0
        else:
            return self._dict[ngram]["frequency"]

    def get_gates(self, ngram: str) -> List[IGate]:
        if ngram not in self._dict:
            return []
        else:
            return self._dict[ngram]["gates"]

    @property
    def ngrams(self) -> List[str]:
        return list(self._dict.keys())

    def add(self, gates: List[IGate], frequency: int):
        ngram = "_".join([gate.__repr__() for gate in gates])
        self._dict[ngram] = {"frequency": frequency, "gates": gates}
        self._frequencies.append(frequency)

    def invert(self) -> "InvertedNgramFrequencyDict":
        inverted_ngram_dict = InvertedGateFrequencyDict()

        for ngram in self._dict:
            frequency = self.get_frequency(ngram)
            gate_sequence = self.get_gates(ngram)

            inverted_ngram_dict.add(frequency, gate_sequence)

        return inverted_ngram_dict

    def __len__(self) -> int:
        return len(self._dict.keys())

    def frequency_at(self, idx: int) -> int:
        """Return the nth largest frequency"""
        self._frequencies.sort(reverse=True)

        if len(self._frequencies) >= idx:
            return self._frequencies[idx]
        else:
            return -1


class InvertedNgramFrequencyDict:
    _dict: Dict

    def __init__(self):
        self._dict = {}

    def add(self, frequency: int, gates: List[IGate]) -> None:
        if frequency in self._dict:
            self._dict[frequency].append(gates)
        else:
            self._dict[frequency] = [gates]

    @property
    def frequencies(self) -> List[int]:
        return list(self._dict.keys())

    def __getitem__(self, frequency: int) -> List[IGate]:
        if frequency in self._dict:
            return self._dict[frequency]
        else:
            return []
