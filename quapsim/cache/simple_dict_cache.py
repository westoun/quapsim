#!/usr/bin/env python3

import numpy as np
from typing import List, Union

from .interface import ICache
from quapsim.gates import IGate


class SimpleDictCache(ICache):
    """Base class for all gate sequence caches."""

    def add(self, gate_sequence: List[IGate], unitary: np.ndarray) -> None:
        """Add the unitary of a sequence of gates to the cache."""
        raise NotImplementedError()

    def retrieve(
        self, gate_sequence: List[IGate], unitary: np.ndarray
    ) -> Union[None, np.ndarray]:
        """Retrieve the unitary of a sequence of gates from the cache
        if it exists. Else return None."""
        raise NotImplementedError()

    def reset(self) -> None:
        """Clear the cache."""
        raise NotImplementedError()
