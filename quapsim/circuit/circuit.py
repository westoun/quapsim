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
from quapsim.gates.utils import create_unitary, create_identity_matrix


def get_unitary(circuit: "Circuit") -> np.ndarray:
    """Computes the unitary matrix of a specified circuit."""
    assert len(circuit.gates) > 0, "Empty circuit encountered!"

    unitary = create_identity_matrix(dim=2**circuit.qubit_num)
    for gate in circuit.gates:
        gate_unitary = create_unitary(gate, circuit.qubit_num)
        unitary = np.matmul(gate_unitary, unitary)

    return unitary


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

    @property
    def unitary(self) -> np.ndarray:
        return get_unitary(self)

    def __repr__(self) -> str:
        return f"[{', '.join([str(gate) for gate in self.gates])}]"
