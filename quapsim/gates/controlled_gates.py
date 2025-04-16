#!/usr/bin/env python3

import numpy as np
from typing import List

from .interface import IGate
from ._matrices import (
    H_MATRIX,
    X_MATRIX,
    Y_MATRIX,
    S_MATRIX,
    Z_MATRIX,
    T_MATRIX,
    RX_MATRIX,
    RY_MATRIX,
    RZ_MATRIX,
    PHASE_MATRIX,
)


class CGate(IGate):
    """Base class for all controlled qubit gates."""

    target_qubit: int
    control_qubit: int
    matrix: np.ndarray = None

    def __init__(self, control_qubit: int, target_qubit: int) -> None:
        self.control_qubit = control_qubit
        self.target_qubit = target_qubit

    @property
    def qubits(self) -> List[int]:
        return [self.target_qubit, self.control_qubit]

    def __repr__(self) -> str:
        gate_name = str(type(self)).split(".")[-1].replace("'>", "")
        return f"{gate_name}(control={self.control_qubit}, target={self.target_qubit})"


class CH(CGate):
    """Controlled Hadamard gate.

    Applies the hadamard gate to the target_qubit
    if the control_qubit is in a state of |1>.
    """

    matrix: np.ndarray = H_MATRIX


class CS(CGate):
    """Controlled S gate.

    Applies the S gate to the target_qubit
    if the control_qubit is in a state of |1>.
    The S gate induces a phase of pi/2.
    """

    matrix: np.ndarray = S_MATRIX


class CT(CGate):
    """Controlled T gate.

    Applies the T gate to the target_qubit
    if the control_qubit is in a state of |1>.
    """

    matrix: np.ndarray = T_MATRIX


class CX(CGate):
    """Controlled Pauli-X gate.

    Applies the pauli-x gate to the target_qubit
    if the control_qubit is in a state of |1>.
    """

    matrix: np.ndarray = X_MATRIX


class CY(CGate):
    """Controlled Pauli-Y gate.

    Applies the pauli-y gate to the target_qubit
    if the control_qubit is in a state of |1>.
    """

    matrix: np.ndarray = Y_MATRIX


class CZ(CGate):
    """Controlled Pauli-Z gate.

    Applies the pauli-z gate to the target_qubit
    if the control_qubit is in a state of |1>.
    """

    matrix: np.ndarray = Z_MATRIX


class CRX(CGate):
    """Controlled Rotational-X gate.

    Performs a rotation by theta/2 degrees around the X axis
    of the target_qubit if the control_qubit is in a state
    of |1>.
    """

    matrix: np.ndarray
    theta: float

    def __init__(self, control_qubit: int, target_qubit: int, theta: float) -> None:
        self.control_qubit = control_qubit
        self.target_qubit = target_qubit
        self.theta = theta
        self.matrix = RX_MATRIX(theta)

    def __repr__(self) -> str:
        gate_name = str(type(self)).split(".")[-1].replace("'>", "")
        return f"{gate_name}(control={self.control_qubit}, target={self.target_qubit}, theta={round(self.theta, 3)})"


class CRY(CGate):
    """Controlled Rotational-Y gate.

    Performs a rotation by theta/2 degrees around the Y axis
    of the target_qubit if the control_qubit is in a state
    of |1>.
    """

    matrix: np.ndarray
    theta: float

    def __init__(self, control_qubit: int, target_qubit: int, theta: float) -> None:
        self.control_qubit = control_qubit
        self.target_qubit = target_qubit
        self.theta = theta
        self.matrix = RY_MATRIX(theta)

    def __repr__(self) -> str:
        gate_name = str(type(self)).split(".")[-1].replace("'>", "")
        return f"{gate_name}(control={self.control_qubit}, target={self.target_qubit}, theta={round(self.theta, 3)})"


class CRZ(CGate):
    """Controlled Rotational-Z gate.

    Performs a rotation by theta/2 degrees around the Z axis
    of the target_qubit if the control_qubit is in a state
    of |1>.
    """

    matrix: np.ndarray
    theta: float

    def __init__(self, control_qubit: int, target_qubit: int, theta: float) -> None:
        self.control_qubit = control_qubit
        self.target_qubit = target_qubit
        self.theta = theta
        self.matrix = RZ_MATRIX(theta)

    def __repr__(self) -> str:
        gate_name = str(type(self)).split(".")[-1].replace("'>", "")
        return f"{gate_name}(control={self.control_qubit}, target={self.target_qubit}, theta={round(self.theta, 3)})"


class CPhase(CGate):
    """Controlled Phase gate.

    Applies the phase gate to the target_qubit
    if the control_qubit is in a state of |1>.
    """

    matrix: np.ndarray
    theta: float

    def __init__(self, control_qubit: int, target_qubit: int, theta: float) -> None:
        self.control_qubit = control_qubit
        self.target_qubit = target_qubit
        self.theta = theta
        self.matrix = PHASE_MATRIX(theta)

    def __repr__(self) -> str:
        gate_name = str(type(self)).split(".")[-1].replace("'>", "")
        return f"{gate_name}(control={self.control_qubit}, target={self.target_qubit}, theta={round(self.theta, 3)})"
