#!/usr/bin/env python3

import numpy as np
from typing import List

from .interface import IGate
from ._matrices import (
    I_MATRIX,
    H_MATRIX,
    X_MATRIX,
    Y_MATRIX,
    Z_MATRIX,
    S_MATRIX,
    T_MATRIX,
    RX_MATRIX,
    RY_MATRIX,
    RZ_MATRIX,
    PHASE_MATRIX,
)


class Gate(IGate):
    """Base class for all single qubit gates."""

    target_qubit: int
    matrix: np.ndarray = None

    def __init__(self, target_qubit: int) -> None:
        self.target_qubit = target_qubit

    @property
    def qubits(self) -> List[int]:
        return [self.target_qubit]

    def __repr__(self) -> str:
        gate_name = str(type(self)).split(".")[-1].replace("'>", "")
        return f"{gate_name}(target={self.target_qubit})"


class Identity(Gate):
    """Identity gate."""

    matrix: np.ndarray = I_MATRIX


class S(Gate):
    """S gate.

    Introduces a phase of pi/2."""

    matrix: np.ndarray = S_MATRIX


class T(Gate):
    """T gate.

    Introduces a phase of pi/4."""

    matrix: np.ndarray = T_MATRIX


class H(Gate):
    """Hadamard gate."""

    matrix: np.ndarray = H_MATRIX


class X(Gate):
    """Pauli-X gate."""

    matrix: np.ndarray = X_MATRIX


class Y(Gate):
    """Pauli-Y gate."""

    matrix: np.ndarray = Y_MATRIX


class Z(Gate):
    """Pauli-Z gate."""

    matrix: np.ndarray = Z_MATRIX


class RX(Gate):
    """Rotational-X gate.

    Performs a rotation by theta/2 degrees around the X axis."""

    matrix: np.ndarray
    theta: float

    def __init__(self, target_qubit: int, theta: float) -> None:
        self.target_qubit = target_qubit
        self.theta = theta
        self.matrix = RX_MATRIX(theta)

    def __repr__(self) -> str:
        gate_name = str(type(self)).split(".")[-1].replace("'>", "")
        return f"{gate_name}(target={self.target_qubit}, theta={round(self.theta, 3)})"


class RY(Gate):
    """Rotational-Y gate.

    Performs a rotation by theta/2 degrees around the Y axis."""

    matrix: np.ndarray
    theta: float

    def __init__(self, target_qubit: int, theta: float) -> None:
        self.target_qubit = target_qubit
        self.theta = theta
        self.matrix = RY_MATRIX(theta)

    def __repr__(self) -> str:
        gate_name = str(type(self)).split(".")[-1].replace("'>", "")
        return f"{gate_name}(target={self.target_qubit}, theta={round(self.theta, 3)})"


class RZ(Gate):
    """Rotational-Z gate.

    Performs a rotation by theta/2 degrees around the Z axis."""

    matrix: np.ndarray
    theta: float

    def __init__(self, target_qubit: int, theta: float) -> None:
        self.target_qubit = target_qubit
        self.theta = theta
        self.matrix = RZ_MATRIX(theta)

    def __repr__(self) -> str:
        gate_name = str(type(self)).split(".")[-1].replace("'>", "")
        return f"{gate_name}(target={self.target_qubit}, theta={round(self.theta, 3)})"


class Phase(Gate):
    """Phase gate."""

    matrix: np.ndarray
    theta: float

    def __init__(self, target_qubit: int, theta: float) -> None:
        self.target_qubit = target_qubit
        self.theta = theta
        self.matrix = PHASE_MATRIX(theta)

    def __repr__(self) -> str:
        gate_name = str(type(self)).split(".")[-1].replace("'>", "")
        return f"{gate_name}(target={self.target_qubit}, theta={round(self.theta, 3)})"
