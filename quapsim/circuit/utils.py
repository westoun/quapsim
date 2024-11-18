#!/usr/bin/env python3

import math
import numpy as np
import warnings


def probabilities_from_state(state: np.ndarray) -> np.ndarray:
    """Returns the probabilities corresponding to a quantum
    system state."""

    conjugate = state.conjugate()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Ignore "np.complex128Warning: Casting np.complex128 values to real discards the imaginary part"
        # since that is precisely what we want.
        probabilities = np.multiply(state, conjugate).astype(float)

    return probabilities


def state_dict_from_state(state: np.ndarray) -> np.ndarray:
    """Returns a dictionary of the states where the coefficients of
    the states are different from 0.
    """
    qubit_num = int(math.log2(len(state)))

    state_dict = {}

    for i, state in enumerate(state):

        if state == 0:
            continue

        qubit_state: str = ""

        remainder = i
        for j in reversed(range(qubit_num)):

            if remainder >= 2**j:
                qubit_state += "1"
                remainder -= 2**j
            else:
                qubit_state += "0"

        state_dict[qubit_state] = state

    return state_dict


def probability_dict_from_state(state: np.ndarray) -> np.ndarray:
    """Returns a dictionary of the probabilities corresponding
    to the specified state. States with a probability of 0 are
    omitted.
    """

    qubit_num = int(math.log2(len(state)))

    probabilities = probabilities_from_state(state)

    probability_dict = {}
    for i, probability in enumerate(probabilities):

        if probability == 0:
            continue

        state: str = ""

        remainder = i
        for j in reversed(range(qubit_num)):

            if remainder >= 2**j:
                state += "1"
                remainder -= 2**j
            else:
                state += "0"

        probability_dict[state] = probability

    return probability_dict
