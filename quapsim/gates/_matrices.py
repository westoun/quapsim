#!/usr/bin/env python3

import cmath
import math
import numpy as np

H_MATRIX = 1.0 / (2.0**0.5) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
X_MATRIX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Z_MATRIX = np.array([[1, 0], [0, -1]], dtype=np.complex128)
Y_MATRIX = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
S_MATRIX = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
T_MATRIX = np.array([[1, 0], [0, cmath.exp(1j * np.pi / 4)]], dtype=np.complex128)


def RX_MATRIX(theta: float) -> np.ndarray:
    return np.array(
        [
            [math.cos(theta / 2), -1j * math.sin(theta / 2)],
            [-1j * math.sin(theta / 2), math.cos(theta / 2)],
        ],
        dtype=np.complex128,
    )


def RY_MATRIX(theta: float) -> np.ndarray:
    return np.array(
        [
            [math.cos(theta / 2), -1 * math.sin(theta / 2)],
            [math.sin(theta / 2), math.cos(theta / 2)],
        ],
        dtype=np.complex128,
    )


def RZ_MATRIX(theta: float) -> np.ndarray:
    return np.array(
        [[cmath.exp(-1j * theta / 2), 0], [0, cmath.exp(1j * theta / 2)]],
        dtype=np.complex128,
    )


def PHASE_MATRIX(theta: float) -> np.ndarray:
    return np.array(
        [[1, 0], [0, cmath.exp(1j * theta)]],
        dtype=np.complex128,
    )
