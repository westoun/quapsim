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
