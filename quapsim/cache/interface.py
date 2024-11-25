#!/usr/bin/env python3

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Union, Set

from quapsim.gates import IGate


class ICache(ABC):
    """Base class for all gate sequence caches."""

    lengths: Set[int]
    """Return a set containing the unique lengths of all cache entries.
        Can be used to guide cache lookups."""

    @abstractmethod
    def add(self, gate_sequence: List[IGate], unitary: np.ndarray) -> None:
        """Add the unitary of a sequence of gates to the cache."""
        ...

    @abstractmethod
    def get(self, gate_sequence: List[IGate]) -> Union[None, np.ndarray]:
        """Retrieve the unitary of a sequence of gates from the cache
        if it exists. Else return None."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear the cache."""
        ...
