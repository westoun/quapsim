#!/usr/bin/env python3

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Union, Set

from quapsim.gates import IGate


class ICache(ABC):
    """Base class for all gate sequence caches."""

    @abstractmethod
    def add(self, gate_sequence: List[IGate], unitary: np.ndarray) -> None:
        """Add the unitary of a sequence of gates to the cache."""
        ...

    @abstractmethod
    def get_prefix_in_cache_length(self, gate_sequence: List[IGate]) -> int:
        """Return the maximum sequence length that is in the cache,
        measured from the beginning of the sequence. This method is
        use full for performing cache lookup."""
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
