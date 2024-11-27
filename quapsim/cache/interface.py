#!/usr/bin/env python3

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Union, Set

from quapsim.gates import IGate


class ICache(ABC):
    """Base class for all gate sequence caches."""

    @abstractmethod
    def could_be_in_cache(self, gate_sequence: List[IGate]) -> bool:
        """Return if a gate sequence or its children could be in
        the cache (based on internal trie structure).
        """
        ...

    @abstractmethod
    def add(self, gate_sequence: List[IGate], unitary: np.ndarray) -> None:
        """Add the unitary of a sequence of gates to the cache."""
        ...

    @abstractmethod
    def get(self, gate_sequence: List[IGate]) -> Union[None, np.ndarray]:
        """Retrieve the unitary of a sequence of gates from the cache
        if it exists. Else return None."""
        ...
