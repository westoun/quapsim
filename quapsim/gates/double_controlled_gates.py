#!/usr/bin/env python3

import numpy as np
from typing import List

from .interface import IGate
from ._matrices import X_MATRIX, Z_MATRIX


class CCGate(IGate):
    """Base class for all double controlled qubit gates."""

    target_qubit: int
    control_qubit1: int
    control_qubit2: int
    matrix: np.ndarray = None

    def __init__(
        self, control_qubit1: int, control_qubit2: int, target_qubit: int
    ) -> None:
        self.control_qubit1 = control_qubit1
        self.control_qubit2 = control_qubit2
        self.target_qubit = target_qubit

    @property
    def qubits(self) -> List[int]:
        return [self.target_qubit, self.control_qubit1, self.control_qubit2]

    def __repr__(self) -> str:
        gate_name = str(type(self)).split(".")[-1].replace("'>", "")
        return f"{gate_name}(control1={self.control_qubit1}, control2={self.control_qubit2}, target={self.target_qubit})"


class CCX(CCGate):
    """Double Controlled Pauli-X gate.

    Applies the pauli-x gate to the target_qubit
    if both control_qubits are in a state of |1>.
    """

    matrix: np.ndarray = X_MATRIX


class CCZ(CCGate):
    """Double Controlled Pauli-Z gate.

    Applies the pauli-z gate to the target_qubit
    if both control_qubits are in a state of |1>.
    """

    matrix: np.ndarray = Z_MATRIX
