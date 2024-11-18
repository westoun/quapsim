#!/usr/bin/env python3

import numpy as np
from typing import List, Union

from .interface import ICache
from quapsim.gates import IGate


def create_key(gates: List[IGate]) -> str:
    key = "_".join([gate.__repr__() for gate in gates])
    return key


class SimpleDictCache(ICache):
    """Base class for all gate sequence caches."""

    def __init__(self):
        self._dict = {}

    def add(self, gate_sequence: List[IGate], unitary: np.ndarray) -> None:
        """Add the unitary of a sequence of gates to the cache."""
        key = create_key(gate_sequence)
        self._dict[key] = unitary

    def retrieve(self, gate_sequence: List[IGate]) -> Union[None, np.ndarray]:
        """Retrieve the unitary of a sequence of gates from the cache
        if it exists. Else return None."""
        key = create_key(gate_sequence)

        if key not in self._dict:
            return None
        else:
            return self._dict[key]

    def reset(self) -> None:
        """Clear the cache."""
        self._dict = {}
