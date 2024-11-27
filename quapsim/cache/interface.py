#!/usr/bin/env python3

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Union, Set, Tuple

from quapsim.gates import IGate


class ICache(ABC):
    """Base class for all gate sequence caches."""

    @abstractmethod
    def get_cache_potential(self, gate_sequence: List[IGate]) -> Tuple[bool, bool]:
        """The first boolean of the returned tuple answers whether the
        any child sequences could be contained in the cache.
        The second one answers if it actually is."""
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
