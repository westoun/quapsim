#!/usr/bin/env python3

from abc import ABC, abstractmethod
import numpy as np
from typing import List


class IGate(ABC):
    """Base class of all quantum gates."""

    matrix: np.ndarray

    @property
    @abstractmethod
    def qubits(self) -> List[int]:
        """Return which qubits the specific gate
        affects."""
        ...

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string that uniquely represents the gate."""
        ...

    def __hash__(self) -> int:
        return hash(self.__repr__())

    def __eq__(self, value: "IGate") -> bool:
        return self.__hash__() == value.__hash__()
