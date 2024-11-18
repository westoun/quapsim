#!/usr/bin/env python3

import numpy as np
from typing import List


def create_identity(dim: int = 2) -> np.ndarray:
    """Creates an identity matrix of the specified
    dimensionality."""
    return np.eye(dim, dtype=np.complex128)


def create_matrix(
    base_matrix: np.ndarray, target_qubit: int, qubit_num: int
) -> np.ndarray:
    """Creates a composed matrix by padding the specified
    base matrix with identity matrices until the desired
    dimensionality of 2^qubit_num is reached.
    """

    matrix = base_matrix

    if qubit_num == 1:
        return matrix

    qubits_before = target_qubit
    if qubits_before > 0:
        identity_before = create_identity(dim=2**qubits_before)
        matrix = np.kron(identity_before, matrix)

    qubits_after = qubit_num - (target_qubit + 1)
    if qubits_after > 0:
        identity_after = create_identity(dim=2**qubits_after)
        matrix = np.kron(matrix, identity_after)

    return matrix


# Corresponds to |0> <0|
PROJECTOR_0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)

# Corresponds to |1> <1|
PROJECTOR_1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)


def create_controlled_matrix(
    base_matrix: np.ndarray, control_qubit: int, target_qubit: int, qubit_num: int
) -> np.ndarray:
    """Creates a controlled matrix based on a base matrix and
    a specified control and target qubit.
    """

    # Based on https://quantumcomputing.stackexchange.com/a/4255
    control_matrix = create_matrix(
        PROJECTOR_0, target_qubit=control_qubit, qubit_num=qubit_num
    )

    target_matrix = None
    for i in range(qubit_num):
        # first matrix to be added
        if target_matrix is None:
            if i == control_qubit:
                target_matrix = PROJECTOR_1
            elif i == target_qubit:
                target_matrix = base_matrix
            else:
                target_matrix = create_identity(2)

        else:
            if i == control_qubit:
                target_matrix = np.kron(target_matrix, PROJECTOR_1)
            elif i == target_qubit:
                target_matrix = np.kron(target_matrix, base_matrix)
            else:
                target_matrix = np.kron(target_matrix, create_identity(2))

    # projector matrix at position of control qubit
    # base matrix at position of target qubit

    matrix = control_matrix + target_matrix
    return matrix


def create_double_controlled_matrix(
    base_matrix: np.ndarray,
    control_qubit1: int,
    control_qubit2: int,
    target_qubit: int,
    qubit_num: int,
) -> np.ndarray:
    """Creates a controlled matrix based on a base matrix, two
    control qubits, and target qubit.
    """

    control_matrix00 = None
    for i in range(qubit_num):
        if control_matrix00 is None:
            if i == control_qubit1:
                control_matrix00 = PROJECTOR_0
            elif i == control_qubit2:
                control_matrix00 = PROJECTOR_0
            else:
                control_matrix00 = create_identity(2)

        else:
            if i == control_qubit1:
                control_matrix00 = np.kron(control_matrix00, PROJECTOR_0)
            elif i == control_qubit2:
                control_matrix00 = np.kron(control_matrix00, PROJECTOR_0)
            else:
                control_matrix00 = np.kron(control_matrix00, create_identity(2))

    control_matrix01 = None
    for i in range(qubit_num):
        if control_matrix01 is None:
            if i == control_qubit1:
                control_matrix01 = PROJECTOR_0
            elif i == control_qubit2:
                control_matrix01 = PROJECTOR_1
            else:
                control_matrix01 = create_identity(2)

        else:
            if i == control_qubit1:
                control_matrix01 = np.kron(control_matrix01, PROJECTOR_0)
            elif i == control_qubit2:
                control_matrix01 = np.kron(control_matrix01, PROJECTOR_1)
            else:
                control_matrix01 = np.kron(control_matrix01, create_identity(2))

    control_matrix10 = None
    for i in range(qubit_num):
        if control_matrix10 is None:
            if i == control_qubit1:
                control_matrix10 = PROJECTOR_1
            elif i == control_qubit2:
                control_matrix10 = PROJECTOR_0
            else:
                control_matrix10 = create_identity(2)

        else:
            if i == control_qubit1:
                control_matrix10 = np.kron(control_matrix10, PROJECTOR_1)
            elif i == control_qubit2:
                control_matrix10 = np.kron(control_matrix10, PROJECTOR_0)
            else:
                control_matrix10 = np.kron(control_matrix10, create_identity(2))

    target_matrix = None
    for i in range(qubit_num):
        if target_matrix is None:
            if i == control_qubit1:
                target_matrix = PROJECTOR_1
            elif i == control_qubit2:
                target_matrix = PROJECTOR_1
            elif i == target_qubit:
                target_matrix = base_matrix
            else:
                target_matrix = create_identity(2)

        else:
            if i == control_qubit1:
                target_matrix = np.kron(target_matrix, PROJECTOR_1)
            elif i == control_qubit2:
                target_matrix = np.kron(target_matrix, PROJECTOR_1)
            elif i == target_qubit:
                target_matrix = np.kron(target_matrix, base_matrix)
            else:
                target_matrix = np.kron(target_matrix, create_identity(2))

    matrix = control_matrix00 + control_matrix01 + control_matrix10 + target_matrix
    return matrix
