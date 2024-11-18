#!/usr/bin/env python3

import numpy as np
from typing import List, Union, Dict

from quapsim.gates import IGate, Swap, Gate, CGate, CCGate, CX
from quapsim.gates._matrices import X_MATRIX
from .utils import (
    probabilities_from_state,
    probability_dict_from_state,
    state_dict_from_state,
)
from quapsim.gates.utils import (
    create_double_controlled_matrix,
    create_controlled_matrix,
    create_matrix,
    create_identity,
)


class Circuit:
    """A quantum circuit consisting of a list of gates
    and a specified amount of qubits.
    """

    gates: List[IGate]
    qubit_num: int

    _state: np.ndarray = None
    _probabilities: np.ndarray = None
    _probability_dict: Dict = None
    _state_dict: Dict = None

    def __init__(self, qubit_num: int) -> None:
        self.gates = []

        self.qubit_num = qubit_num

    def apply(self, gate: IGate) -> None:
        """Appends the specified gate to the list of
        gates already in the circuit."""
        self._state, self._probabilities = None, None

        self.gates.append(gate)

    @property
    def state(self) -> Union[np.ndarray, None]:
        """Returns the state of the circuit after all
        quantum gates have been applied.

        If the circuit has not been evaluated by the
        simulator, None is returned.
        """
        return self._state

    def set_state(self, state: np.ndarray) -> None:
        self._state = state

    @property
    def probabilities(self) -> Union[np.ndarray, None]:
        """Returns the probabilities corresponding to the
        state of the circuit after all
        quantum gates have been applied.

        If the circuit has not been evaluated by the
        simulator, None is returned.
        """

        if self._state is None:
            return None

        if self._probabilities is not None:
            return self._probabilities
        else:
            self._probabilities = probabilities_from_state(self.state)
            return self._probabilities

    @property
    def probability_dict(self) -> Union[Dict, None]:
        """Returns a dictionary of the probabilities
        corresponding to the state of the circuit after all
        quantum gates have been applied. States with a
        probability of 0 are omitted.

        If the circuit has not been evaluated by the
        simulator, None is returned.
        """

        if self._state is None:
            return None

        if self._probability_dict is not None:
            return self._probability_dict
        else:
            self._probability_dict = probability_dict_from_state(self.state)
            return self._probability_dict

    @property
    def state_dict(self) -> Union[Dict, None]:
        """Returns a dictionary of the state coefficients
        corresponding to the state of the circuit after all
        quantum gates have been applied. States with a
        probability of 0 are omitted.

        If the circuit has not been evaluated by the
        simulator, None is returned.
        """

        if self._state is None:
            return None

        if self._state_dict is not None:
            return self._state_dict
        else:
            self._state_dict = state_dict_from_state(self.state)
            return self._state_dict

    def __repr__(self) -> str:
        return f"[{', '.join([str(gate) for gate in self.gates])}]"


def get_unitary(circuit: Circuit) -> np.ndarray:
    """Computes the unitary matrix of a specified circuit."""
    assert len(circuit.gates) > 0, "Empty circuit encountered!"

    unitary = create_identity(dim=2**circuit.qubit_num)
    for gate in circuit.gates:
        if type(gate) == Swap:
            cnot1 = create_controlled_matrix(
                X_MATRIX, gate.qubit1, gate.qubit2, circuit.qubit_num
            )
            cnot2 = create_controlled_matrix(
                X_MATRIX, gate.qubit2, gate.qubit1, circuit.qubit_num
            )

            unitary = np.matmul(cnot1, unitary)
            unitary = np.matmul(cnot2, unitary)
            unitary = np.matmul(cnot1, unitary)

        elif issubclass(gate.__class__, Gate):
            matrix = create_matrix(gate.matrix, gate.target_qubit, circuit.qubit_num)
            unitary = np.matmul(matrix, unitary)

        elif issubclass(gate.__class__, CGate):
            matrix = create_controlled_matrix(
                gate.matrix, gate.control_qubit, gate.target_qubit, circuit.qubit_num
            )
            unitary = np.matmul(matrix, unitary)

        elif issubclass(gate.__class__, CCGate):
            matrix = create_double_controlled_matrix(
                gate.matrix,
                gate.control_qubit1,
                gate.control_qubit2,
                gate.target_qubit,
                circuit.qubit_num,
            )
            unitary = np.matmul(matrix, unitary)

        else:
            raise NotImplementedError(f"Unknown gate type for {gate} ({type(gate)})")

    return unitary
