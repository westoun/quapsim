#!/usr/bin/env python3

import numpy as np
from random import choice, randint, sample, random
from typing import Tuple, List

from quapsim import Circuit as QuapsimCircuit
import quapsim.gates as QuapsimGates

from quasim import Circuit as QuasimCircuit
import quasim.gates as QuasimGates

GATES = [
    "H",
    "X",
    "Y",
    "Z",
    "CH",
    "CX",
    "CY",
    "CZ",
    "CRX",
    "CRY",
    "CRZ",
    "RX",
    "RY",
    "RZ",
    "CCX",
    "CCZ",
    "PHASE",
    "SWAP",
    "S",
    "T",
    "CS",
    "CPhase",
]


def create_random_circuits(
    gate_count: int = 10, qubit_num: int = 4
) -> Tuple[QuapsimCircuit, QuasimCircuit]:
    quapsim_circuit = QuapsimCircuit(qubit_num)
    quasim_circuit = QuasimCircuit(qubit_num)

    for _ in range(gate_count):
        gate_type = choice(GATES)

        if gate_type == "H":
            target_qubit = randint(0, qubit_num - 1)
            quapsim_circuit.apply(QuapsimGates.H(target_qubit))
            quasim_circuit.apply(QuasimGates.H(target_qubit))

        elif gate_type == "X":
            target_qubit = randint(0, qubit_num - 1)
            quapsim_circuit.apply(QuapsimGates.X(target_qubit))
            quasim_circuit.apply(QuasimGates.X(target_qubit))

        elif gate_type == "Y":
            target_qubit = randint(0, qubit_num - 1)
            quapsim_circuit.apply(QuapsimGates.Y(target_qubit))
            quasim_circuit.apply(QuasimGates.Y(target_qubit))

        elif gate_type == "Z":
            target_qubit = randint(0, qubit_num - 1)
            quapsim_circuit.apply(QuapsimGates.Z(target_qubit))
            quasim_circuit.apply(QuasimGates.Z(target_qubit))

        elif gate_type == "S":
            target_qubit = randint(0, qubit_num - 1)
            quapsim_circuit.apply(QuapsimGates.S(target_qubit))
            quasim_circuit.apply(QuasimGates.S(target_qubit))

        elif gate_type == "T":
            target_qubit = randint(0, qubit_num - 1)
            quapsim_circuit.apply(QuapsimGates.T(target_qubit))
            quasim_circuit.apply(QuasimGates.T(target_qubit))

        elif gate_type == "CS":
            target_qubit, control_qubit = sample(range(0, qubit_num), 2)
            quapsim_circuit.apply(QuapsimGates.CS(control_qubit, target_qubit))
            quasim_circuit.apply(QuasimGates.CS(control_qubit, target_qubit))

        elif gate_type == "CH":
            target_qubit, control_qubit = sample(range(0, qubit_num), 2)
            quapsim_circuit.apply(QuapsimGates.CH(control_qubit, target_qubit))
            quasim_circuit.apply(QuasimGates.CH(control_qubit, target_qubit))

        elif gate_type == "CX":
            target_qubit, control_qubit = sample(range(0, qubit_num), 2)
            quapsim_circuit.apply(QuapsimGates.CX(control_qubit, target_qubit))
            quasim_circuit.apply(QuasimGates.CX(control_qubit, target_qubit))

        elif gate_type == "CY":
            target_qubit, control_qubit = sample(range(0, qubit_num), 2)
            quapsim_circuit.apply(QuapsimGates.CY(control_qubit, target_qubit))
            quasim_circuit.apply(QuasimGates.CY(control_qubit, target_qubit))

        elif gate_type == "CZ":
            target_qubit, control_qubit = sample(range(0, qubit_num), 2)
            quapsim_circuit.apply(QuapsimGates.CZ(control_qubit, target_qubit))
            quasim_circuit.apply(QuasimGates.CZ(control_qubit, target_qubit))

        elif gate_type == "RX":
            target_qubit = randint(0, qubit_num - 1)
            theta = random() * 2 * np.pi - np.pi

            quapsim_circuit.apply(QuapsimGates.RX(target_qubit, theta))
            quasim_circuit.apply(QuasimGates.RX(target_qubit, theta))

        elif gate_type == "RY":
            target_qubit = randint(0, qubit_num - 1)
            theta = random() * 2 * np.pi - np.pi

            quapsim_circuit.apply(QuapsimGates.RY(target_qubit, theta))
            quasim_circuit.apply(QuasimGates.RY(target_qubit, theta))

        elif gate_type == "RZ":
            target_qubit = randint(0, qubit_num - 1)
            theta = random() * 2 * np.pi - np.pi

            quapsim_circuit.apply(QuapsimGates.RZ(target_qubit, theta))
            quasim_circuit.apply(QuasimGates.RZ(target_qubit, theta))

        elif gate_type == "CRX":
            target_qubit, control_qubit = sample(range(0, qubit_num), 2)
            theta = random() * 2 * np.pi - np.pi

            quapsim_circuit.apply(QuapsimGates.CRX(control_qubit, target_qubit, theta))
            quasim_circuit.apply(QuasimGates.CRX(control_qubit, target_qubit, theta))

        elif gate_type == "CRY":
            target_qubit, control_qubit = sample(range(0, qubit_num), 2)
            theta = random() * 2 * np.pi - np.pi

            quapsim_circuit.apply(QuapsimGates.CRY(control_qubit, target_qubit, theta))
            quasim_circuit.apply(QuasimGates.CRY(control_qubit, target_qubit, theta))

        elif gate_type == "CPhase":
            target_qubit, control_qubit = sample(range(0, qubit_num), 2)
            theta = random() * 2 * np.pi - np.pi

            quapsim_circuit.apply(
                QuapsimGates.CPhase(control_qubit, target_qubit, theta)
            )
            quasim_circuit.apply(QuasimGates.CPhase(control_qubit, target_qubit, theta))

        elif gate_type == "CRZ":
            target_qubit, control_qubit = sample(range(0, qubit_num), 2)
            theta = random() * 2 * np.pi - np.pi

            quapsim_circuit.apply(QuapsimGates.CRZ(control_qubit, target_qubit, theta))
            quasim_circuit.apply(QuasimGates.CRZ(control_qubit, target_qubit, theta))

        elif gate_type == "CCX":
            target_qubit, control_qubit1, control_qubit2 = sample(
                range(0, qubit_num), 3
            )
            quapsim_circuit.apply(
                QuapsimGates.CCX(control_qubit1, control_qubit2, target_qubit)
            )
            quasim_circuit.apply(
                QuasimGates.CCX(control_qubit1, control_qubit2, target_qubit)
            )

        elif gate_type == "CCZ":
            target_qubit, control_qubit1, control_qubit2 = sample(
                range(0, qubit_num), 3
            )
            quapsim_circuit.apply(
                QuapsimGates.CCZ(control_qubit1, control_qubit2, target_qubit)
            )
            quasim_circuit.apply(
                QuasimGates.CCZ(control_qubit1, control_qubit2, target_qubit)
            )

        elif gate_type == "PHASE":
            target_qubit = randint(0, qubit_num - 1)
            theta = random() * 2 * np.pi - np.pi

            quapsim_circuit.apply(QuapsimGates.Phase(target_qubit, theta))
            quasim_circuit.apply(QuasimGates.Phase(target_qubit, theta))

        elif gate_type == "SWAP":
            qubit1, qubit2 = sample(range(0, qubit_num), 2)

            quapsim_circuit.apply(QuapsimGates.Swap(qubit1, qubit2))
            quasim_circuit.apply(QuasimGates.Swap(qubit1, qubit2))

        else:
            raise NotImplementedError()

    return (quapsim_circuit, quasim_circuit)
