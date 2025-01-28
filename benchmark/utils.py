#!/usr/bin/env python3

import logging
import numpy as np
from random import choice, randint, sample, random, choices
from typing import Tuple, List

from quapsim import Circuit as QuapsimCircuit
import quapsim.gates as QuapsimGates

from quasim import Circuit as QuasimCircuit
import quasim.gates as QuasimGates

SINGLE_GATES = [
    "H",
    "X",
    "Y",
    "Z",
    "S",
    "T",
]
SINGLE_PARAMETERIZED_GATES = [
    "RX",
    "RY",
    "RZ",
    "PHASE",
]
CONTROLLED_GATES = [
    "SWAP",
    "CS",
    "CH",
    "CX",
    "CY",
    "CZ",
]
CONTROLLED_PARAMETERIZED_GATES = [
    "CRX",
    "CRY",
    "CRZ",
    "CPhase",
]
DOUBLE_CONTROLLED_GATES = [
    "CCX",
    "CCZ",
]


def compute_redundancy(circuits: List[QuapsimCircuit]) -> float:
    all_bigram_count = len(circuits) * (len(circuits[0].gates) - 1)

    unique_bigrams = set()
    for circuit in circuits:
        for i in range(len(circuit.gates) - 1):
            gate = circuit.gates[i]
            succ_gate = circuit.gates[i + 1]

            bigram = f"{gate.__repr__()}_{succ_gate.__repr__()}"
            unique_bigrams.add(bigram)

    return 1 - (len(unique_bigrams) - 1) / (all_bigram_count - 1)


def create_random_gate_configs(
    gate_count: int = 10,
    qubit_num: int = 4,
    entangle_first: bool = False,
    parameter_amount: int = 10,  # Specifies how many distinct values the parameterized
    # gates can exhibit.
) -> List[Tuple]:
    gate_configs: List[Tuple] = []

    existing_gates: int = 0
    if entangle_first:
        gate_configs.append(("H", 0))
        existing_gates += 1

        for control_qubit in range(qubit_num - 1):
            target_qubit = control_qubit + 1
            gate_configs.append(("CX", control_qubit, target_qubit))
            existing_gates += 1

    assert (
        existing_gates <= gate_count
    ), "If you set entangle_first=True, you need to raise the desired gate count."

    single_gate_type_count = len(SINGLE_GATES) * qubit_num
    single_parameterized_gate_type_count = (
        len(SINGLE_PARAMETERIZED_GATES) * parameter_amount * qubit_num
    )
    controlled_gate_type_count = len(CONTROLLED_GATES) * qubit_num * (qubit_num - 1)
    controlled_parameterized_gate_type_count = (
        len(CONTROLLED_PARAMETERIZED_GATES)
        * parameter_amount
        * qubit_num
        * (qubit_num - 1)
    )
    double_controlled_gate_type_count = (
        len(DOUBLE_CONTROLLED_GATES) * qubit_num * (qubit_num - 1) * (qubit_num - 2)
    )

    # The amount of unique gate configurations
    total_gate_type_count = (
        single_gate_type_count
        + single_parameterized_gate_type_count
        + controlled_gate_type_count
        + controlled_parameterized_gate_type_count
        + double_controlled_gate_type_count
    )

    # Avoid logging the total amount of gate configs more than once without having
    # to pass around additional params.
    if not hasattr(create_random_gate_configs, "_logged_total"):
        logging.debug(
            f"Considering a total amount of {total_gate_type_count} unique gate configurations."
        )
        create_random_gate_configs._logged_total = True

    # First pick a gate set based on the amount of unique
    # gate types within the set in relation to the total amount
    # of gate types. Then, within a set, each gate type occurrs
    # equally often.
    candidate_sets: List[List[str]] = choices(
        population=[
            SINGLE_GATES,
            SINGLE_PARAMETERIZED_GATES,
            CONTROLLED_GATES,
            CONTROLLED_PARAMETERIZED_GATES,
            DOUBLE_CONTROLLED_GATES,
        ],
        weights=[
            single_gate_type_count / total_gate_type_count,
            single_parameterized_gate_type_count / total_gate_type_count,
            controlled_gate_type_count / total_gate_type_count,
            controlled_parameterized_gate_type_count / total_gate_type_count,
            double_controlled_gate_type_count / total_gate_type_count,
        ],
        k=gate_count - existing_gates,
    )

    for gate_set in candidate_sets:
        gate_type = choice(gate_set)

        if gate_type == "H":
            target_qubit = randint(0, qubit_num - 1)
            gate_configs.append((gate_type, target_qubit))

        elif gate_type == "X":
            target_qubit = randint(0, qubit_num - 1)
            gate_configs.append((gate_type, target_qubit))

        elif gate_type == "Y":
            target_qubit = randint(0, qubit_num - 1)
            gate_configs.append((gate_type, target_qubit))

        elif gate_type == "Z":
            target_qubit = randint(0, qubit_num - 1)
            gate_configs.append((gate_type, target_qubit))

        elif gate_type == "S":
            target_qubit = randint(0, qubit_num - 1)
            gate_configs.append((gate_type, target_qubit))

        elif gate_type == "T":
            target_qubit = randint(0, qubit_num - 1)
            gate_configs.append((gate_type, target_qubit))

        elif gate_type == "CS":
            target_qubit, control_qubit = sample(range(0, qubit_num), 2)
            gate_configs.append((gate_type, control_qubit, target_qubit))

        elif gate_type == "CH":
            target_qubit, control_qubit = sample(range(0, qubit_num), 2)
            gate_configs.append((gate_type, control_qubit, target_qubit))

        elif gate_type == "CX":
            target_qubit, control_qubit = sample(range(0, qubit_num), 2)
            gate_configs.append((gate_type, control_qubit, target_qubit))

        elif gate_type == "CY":
            target_qubit, control_qubit = sample(range(0, qubit_num), 2)
            gate_configs.append((gate_type, control_qubit, target_qubit))

        elif gate_type == "CZ":
            target_qubit, control_qubit = sample(range(0, qubit_num), 2)
            gate_configs.append((gate_type, control_qubit, target_qubit))

        elif gate_type == "RX":
            target_qubit = randint(0, qubit_num - 1)

            theta = randint(0, parameter_amount - 1) * (4 * np.pi) / (parameter_amount)

            gate_configs.append((gate_type, target_qubit, theta))

        elif gate_type == "RY":
            target_qubit = randint(0, qubit_num - 1)

            theta = randint(0, parameter_amount - 1) * (4 * np.pi) / (parameter_amount)

            gate_configs.append((gate_type, target_qubit, theta))

        elif gate_type == "RZ":
            target_qubit = randint(0, qubit_num - 1)

            theta = randint(0, parameter_amount - 1) * (4 * np.pi) / (parameter_amount)

            gate_configs.append((gate_type, target_qubit, theta))

        elif gate_type == "CRX":
            target_qubit, control_qubit = sample(range(0, qubit_num), 2)

            theta = randint(0, parameter_amount - 1) * (4 * np.pi) / (parameter_amount)

            gate_configs.append((gate_type, control_qubit, target_qubit, theta))

        elif gate_type == "CRY":
            target_qubit, control_qubit = sample(range(0, qubit_num), 2)

            theta = randint(0, parameter_amount - 1) * (4 * np.pi) / (parameter_amount)

            gate_configs.append((gate_type, control_qubit, target_qubit, theta))

        elif gate_type == "CPhase":
            target_qubit, control_qubit = sample(range(0, qubit_num), 2)

            theta = randint(0, parameter_amount - 1) * (4 * np.pi) / (parameter_amount)

            gate_configs.append((gate_type, control_qubit, target_qubit, theta))

        elif gate_type == "CRZ":
            target_qubit, control_qubit = sample(range(0, qubit_num), 2)

            theta = randint(0, parameter_amount - 1) * (4 * np.pi) / (parameter_amount)

            gate_configs.append((gate_type, control_qubit, target_qubit, theta))

        elif gate_type == "CCX":
            target_qubit, control_qubit1, control_qubit2 = sample(
                range(0, qubit_num), 3
            )

            gate_configs.append(
                (gate_type, control_qubit1, control_qubit2, target_qubit)
            )

        elif gate_type == "CCZ":
            target_qubit, control_qubit1, control_qubit2 = sample(
                range(0, qubit_num), 3
            )

            gate_configs.append(
                (gate_type, control_qubit1, control_qubit2, target_qubit)
            )

        elif gate_type == "PHASE":
            target_qubit = randint(0, qubit_num - 1)

            theta = randint(0, parameter_amount - 1) * (4 * np.pi) / (parameter_amount)

            gate_configs.append((gate_type, target_qubit, theta))

        elif gate_type == "SWAP":
            qubit1, qubit2 = sample(range(0, qubit_num), 2)

            gate_configs.append((gate_type, qubit1, qubit2))

        else:
            raise NotImplementedError()

    return gate_configs


def build_quasim_circuit(gate_configs: List[Tuple], qubit_num: int) -> QuasimCircuit:
    quasim_circuit = QuasimCircuit(qubit_num)

    for gate_config in gate_configs:
        gate_type = gate_config[0]

        if gate_type == "H":
            target_qubit = gate_config[1]
            quasim_circuit.apply(QuasimGates.H(target_qubit))

        elif gate_type == "X":
            target_qubit = gate_config[1]
            quasim_circuit.apply(QuasimGates.X(target_qubit))

        elif gate_type == "Y":
            target_qubit = gate_config[1]
            quasim_circuit.apply(QuasimGates.Y(target_qubit))

        elif gate_type == "Z":
            target_qubit = gate_config[1]
            quasim_circuit.apply(QuasimGates.Z(target_qubit))

        elif gate_type == "S":
            target_qubit = gate_config[1]
            quasim_circuit.apply(QuasimGates.S(target_qubit))

        elif gate_type == "T":
            target_qubit = gate_config[1]
            quasim_circuit.apply(QuasimGates.T(target_qubit))

        elif gate_type == "CS":
            control_qubit, target_qubit = gate_config[1], gate_config[2]
            quasim_circuit.apply(QuasimGates.CS(control_qubit, target_qubit))

        elif gate_type == "CH":
            control_qubit, target_qubit = gate_config[1], gate_config[2]
            quasim_circuit.apply(QuasimGates.CH(control_qubit, target_qubit))

        elif gate_type == "CX":
            control_qubit, target_qubit = gate_config[1], gate_config[2]
            quasim_circuit.apply(QuasimGates.CX(control_qubit, target_qubit))

        elif gate_type == "CY":
            control_qubit, target_qubit = gate_config[1], gate_config[2]
            quasim_circuit.apply(QuasimGates.CY(control_qubit, target_qubit))

        elif gate_type == "CZ":
            control_qubit, target_qubit = gate_config[1], gate_config[2]
            quasim_circuit.apply(QuasimGates.CZ(control_qubit, target_qubit))

        elif gate_type == "RX":
            target_qubit, theta = gate_config[1], gate_config[2]
            quasim_circuit.apply(QuasimGates.RX(target_qubit, theta))

        elif gate_type == "RY":
            target_qubit, theta = gate_config[1], gate_config[2]
            quasim_circuit.apply(QuasimGates.RY(target_qubit, theta))

        elif gate_type == "RZ":
            target_qubit, theta = gate_config[1], gate_config[2]
            quasim_circuit.apply(QuasimGates.RZ(target_qubit, theta))

        elif gate_type == "CRX":
            control_qubit, target_qubit, theta = (
                gate_config[1],
                gate_config[2],
                gate_config[3],
            )
            quasim_circuit.apply(QuasimGates.CRX(control_qubit, target_qubit, theta))

        elif gate_type == "CRY":
            control_qubit, target_qubit, theta = (
                gate_config[1],
                gate_config[2],
                gate_config[3],
            )
            quasim_circuit.apply(QuasimGates.CRY(control_qubit, target_qubit, theta))

        elif gate_type == "CPhase":
            control_qubit, target_qubit, theta = (
                gate_config[1],
                gate_config[2],
                gate_config[3],
            )
            quasim_circuit.apply(QuasimGates.CPhase(control_qubit, target_qubit, theta))

        elif gate_type == "CRZ":
            control_qubit, target_qubit, theta = (
                gate_config[1],
                gate_config[2],
                gate_config[3],
            )
            quasim_circuit.apply(QuasimGates.CRZ(control_qubit, target_qubit, theta))

        elif gate_type == "CCX":
            control_qubit1, control_qubit2, target_qubit = (
                gate_config[1],
                gate_config[2],
                gate_config[3],
            )
            quasim_circuit.apply(
                QuasimGates.CCX(control_qubit1, control_qubit2, target_qubit)
            )

        elif gate_type == "CCZ":
            control_qubit1, control_qubit2, target_qubit = (
                gate_config[1],
                gate_config[2],
                gate_config[3],
            )
            quasim_circuit.apply(
                QuasimGates.CCZ(control_qubit1, control_qubit2, target_qubit)
            )

        elif gate_type == "PHASE":
            target_qubit, theta = gate_config[1], gate_config[2]
            quasim_circuit.apply(QuasimGates.Phase(target_qubit, theta))

        elif gate_type == "SWAP":
            qubit1, qubit2 = gate_config[1], gate_config[2]
            quasim_circuit.apply(QuasimGates.Swap(qubit1, qubit2))

        else:
            raise NotImplementedError()

    return quasim_circuit


def build_quapsim_circuit(gate_configs: List[Tuple], qubit_num: int) -> QuapsimCircuit:
    quapsim_circuit = QuapsimCircuit(qubit_num)

    for gate_config in gate_configs:
        gate_type = gate_config[0]

        if gate_type == "H":
            target_qubit = gate_config[1]
            quapsim_circuit.apply(QuapsimGates.H(target_qubit))

        elif gate_type == "X":
            target_qubit = gate_config[1]
            quapsim_circuit.apply(QuapsimGates.X(target_qubit))

        elif gate_type == "Y":
            target_qubit = gate_config[1]
            quapsim_circuit.apply(QuapsimGates.Y(target_qubit))

        elif gate_type == "Z":
            target_qubit = gate_config[1]
            quapsim_circuit.apply(QuapsimGates.Z(target_qubit))

        elif gate_type == "S":
            target_qubit = gate_config[1]
            quapsim_circuit.apply(QuapsimGates.S(target_qubit))

        elif gate_type == "T":
            target_qubit = gate_config[1]
            quapsim_circuit.apply(QuapsimGates.T(target_qubit))

        elif gate_type == "CS":
            control_qubit, target_qubit = gate_config[1], gate_config[2]
            quapsim_circuit.apply(QuapsimGates.CS(control_qubit, target_qubit))

        elif gate_type == "CH":
            control_qubit, target_qubit = gate_config[1], gate_config[2]
            quapsim_circuit.apply(QuapsimGates.CH(control_qubit, target_qubit))

        elif gate_type == "CX":
            control_qubit, target_qubit = gate_config[1], gate_config[2]
            quapsim_circuit.apply(QuapsimGates.CX(control_qubit, target_qubit))

        elif gate_type == "CY":
            control_qubit, target_qubit = gate_config[1], gate_config[2]
            quapsim_circuit.apply(QuapsimGates.CY(control_qubit, target_qubit))

        elif gate_type == "CZ":
            control_qubit, target_qubit = gate_config[1], gate_config[2]
            quapsim_circuit.apply(QuapsimGates.CZ(control_qubit, target_qubit))

        elif gate_type == "RX":
            target_qubit, theta = gate_config[1], gate_config[2]
            quapsim_circuit.apply(QuapsimGates.RX(target_qubit, theta))

        elif gate_type == "RY":
            target_qubit, theta = gate_config[1], gate_config[2]
            quapsim_circuit.apply(QuapsimGates.RY(target_qubit, theta))

        elif gate_type == "RZ":
            target_qubit, theta = gate_config[1], gate_config[2]
            quapsim_circuit.apply(QuapsimGates.RZ(target_qubit, theta))

        elif gate_type == "CRX":
            control_qubit, target_qubit, theta = (
                gate_config[1],
                gate_config[2],
                gate_config[3],
            )
            quapsim_circuit.apply(QuapsimGates.CRX(control_qubit, target_qubit, theta))

        elif gate_type == "CRY":
            control_qubit, target_qubit, theta = (
                gate_config[1],
                gate_config[2],
                gate_config[3],
            )
            quapsim_circuit.apply(QuapsimGates.CRY(control_qubit, target_qubit, theta))

        elif gate_type == "CPhase":
            control_qubit, target_qubit, theta = (
                gate_config[1],
                gate_config[2],
                gate_config[3],
            )
            quapsim_circuit.apply(
                QuapsimGates.CPhase(control_qubit, target_qubit, theta)
            )

        elif gate_type == "CRZ":
            control_qubit, target_qubit, theta = (
                gate_config[1],
                gate_config[2],
                gate_config[3],
            )
            quapsim_circuit.apply(QuapsimGates.CRZ(control_qubit, target_qubit, theta))

        elif gate_type == "CCX":
            control_qubit1, control_qubit2, target_qubit = (
                gate_config[1],
                gate_config[2],
                gate_config[3],
            )
            quapsim_circuit.apply(
                QuapsimGates.CCX(control_qubit1, control_qubit2, target_qubit)
            )

        elif gate_type == "CCZ":
            control_qubit1, control_qubit2, target_qubit = (
                gate_config[1],
                gate_config[2],
                gate_config[3],
            )
            quapsim_circuit.apply(
                QuapsimGates.CCZ(control_qubit1, control_qubit2, target_qubit)
            )

        elif gate_type == "PHASE":
            target_qubit, theta = gate_config[1], gate_config[2]
            quapsim_circuit.apply(QuapsimGates.Phase(target_qubit, theta))

        elif gate_type == "SWAP":
            qubit1, qubit2 = gate_config[1], gate_config[2]
            quapsim_circuit.apply(QuapsimGates.Swap(qubit1, qubit2))

        else:
            raise NotImplementedError()

    return quapsim_circuit
