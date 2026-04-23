import numpy as np
from random import choice, sample, choices, randint
from typing import List, Type

from quapsim.gates import IGate, H, CPhase, Swap, Phase, \
    X, Y, Z, CX, CY, CZ, Identity

from quapsim import Circuit

PARAM_COUNT = 10

SINGLE_QUBIT_GATES = [Identity, H, X, Y, Z]
SINGLE_QUBIT_PARAMETERIZED_GATES = [Phase]
TWO_QUBIT_GATES = [CX, CY, CZ, Swap]
TWO_QUBIT_PARAMETERIZED_GATES = [CPhase]


def _random_gate_uniform_by_config(qubit_num: int, param_count: int = PARAM_COUNT) -> IGate:
    # TODO: Refactor to avoid generating whole list.
    allowed_thetas = [
        2 * np.pi / 2 ** i for i in range(param_count + 1)
    ]

    single_gate_type_count = len(SINGLE_QUBIT_GATES) * qubit_num
    single_parameterized_gate_type_count = (
        len(SINGLE_QUBIT_PARAMETERIZED_GATES) * param_count * qubit_num
    )
    controlled_gate_type_count = len(
        TWO_QUBIT_GATES) * qubit_num * (qubit_num - 1)
    controlled_parameterized_gate_type_count = (
        len(TWO_QUBIT_PARAMETERIZED_GATES)
        * param_count
        * qubit_num
        * (qubit_num - 1)
    )

    total_gate_type_count = (
        single_gate_type_count
        + single_parameterized_gate_type_count
        + controlled_gate_type_count
        + controlled_parameterized_gate_type_count
    )

    gate_type_family = choices(
        population=[
            SINGLE_QUBIT_GATES,
            SINGLE_QUBIT_PARAMETERIZED_GATES,
            TWO_QUBIT_GATES,
            TWO_QUBIT_PARAMETERIZED_GATES,
        ],
        weights=[
            single_gate_type_count / total_gate_type_count,
            single_parameterized_gate_type_count / total_gate_type_count,
            controlled_gate_type_count / total_gate_type_count,
            controlled_parameterized_gate_type_count / total_gate_type_count,
        ],
        k=1,
    )[0]

    if gate_type_family == SINGLE_QUBIT_GATES:
        GateType = choice(SINGLE_QUBIT_GATES)
        target_qubit = randint(0, qubit_num - 1)
        return GateType(target_qubit)

    elif gate_type_family == SINGLE_QUBIT_PARAMETERIZED_GATES:
        GateType = choice(SINGLE_QUBIT_PARAMETERIZED_GATES)
        target_qubit = randint(0, qubit_num - 1)
        theta = choice(allowed_thetas)
        return GateType(target_qubit, theta)

    elif gate_type_family == TWO_QUBIT_GATES:
        GateType = choice(TWO_QUBIT_GATES)
        target_qubit, control_qubit = sample(range(0, qubit_num), 2)
        return GateType(control_qubit, target_qubit)

    elif gate_type_family == TWO_QUBIT_PARAMETERIZED_GATES:
        GateType = choice(TWO_QUBIT_PARAMETERIZED_GATES)
        target_qubit, control_qubit = sample(range(0, qubit_num), 2)
        theta = choice(allowed_thetas)
        return GateType(control_qubit, target_qubit, theta)

    else:
        raise NotImplementedError()


def _random_gate_uniform_by_gatetype(qubit_num: int, param_count: int = PARAM_COUNT) -> IGate:
    allowed_thetas = [
        2 * np.pi / 2 ** i for i in range(param_count + 1)
    ]

    GateType = choice(SINGLE_QUBIT_GATES + SINGLE_QUBIT_PARAMETERIZED_GATES +
                      TWO_QUBIT_GATES + TWO_QUBIT_PARAMETERIZED_GATES)

    if GateType in SINGLE_QUBIT_GATES:
        target_qubit = randint(0, qubit_num - 1)
        return GateType(target_qubit)

    elif GateType in SINGLE_QUBIT_PARAMETERIZED_GATES:
        target_qubit = randint(0, qubit_num - 1)
        theta = choice(allowed_thetas)
        return GateType(target_qubit, theta)

    elif GateType in TWO_QUBIT_GATES:
        target_qubit, control_qubit = sample(range(0, qubit_num), 2)
        return GateType(control_qubit, target_qubit)

    elif GateType in TWO_QUBIT_PARAMETERIZED_GATES:
        target_qubit, control_qubit = sample(range(0, qubit_num), 2)
        theta = choice(allowed_thetas)
        return GateType(control_qubit, target_qubit, theta)

    else:
        raise NotImplementedError(
            f"No implementation found for gate type '{GateType}'")


def random_gate(qubit_num: int, param_count: int = 10, uniform_configuration_choice: bool = True) -> IGate:
    if uniform_configuration_choice:
        return _random_gate_uniform_by_config(qubit_num, param_count)
    else:
        return _random_gate_uniform_by_gatetype(qubit_num, param_count)


def random_circuit(qubit_num: int, gate_count: int, param_count: int) -> Circuit:
    gates = []

    for _ in range(gate_count):
        gate = random_gate(qubit_num, param_count)
        gates.append(gate)

    circuit = Circuit(gates, qubit_num)
    return circuit
