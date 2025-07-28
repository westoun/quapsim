#!/usr/bin/env python3

import click
import logging
import numpy as np
import numpy as np
import random
from random import choice
from typing import List, Tuple, Union, Type
from uuid import uuid4

from ga4qc.ga import GA
from ga4qc.seeder import RandomSeeder, ISeeder
from ga4qc.callback import (
    ICallback,
    FitnessStatsCallback,
    BestCircuitCallback,
    UniqueCircuitCountCallback,
)
from ga4qc.circuit import Circuit
from ga4qc.processors import (
    ISimulator,
    AbsoluteUnitaryDistance
)
from ga4qc.mutation import RandomGateMutation, ParameterChangeMutation, IMutation
from ga4qc.crossover import OnePointCrossover
from ga4qc.selection import ISelection, TournamentSelection
from ga4qc.circuit.gates import Identity, CX, S, T, H, X, RX, CCZ, CZ, CCX, \
    Z, Y, IGate, OracleConstructor, Oracle, CY, RY, RZ, CRX, CRY, CRZ, \
    CLIFFORD_PLUS_T, Phase, IOptimizableGate, Swap
from ga4qc.params import GAParams

from quapsim import QuaPSim, SimulatorParams, SimpleDictCache
from quapsim import Circuit as QuapsimCircuit
from quapsim.gates import Gate as QuapsimGate
import quapsim.gates
from quapsim.simulator.utils import (
    compute_redundancy,
)


def log_gate_types(circuits: List[Circuit]) -> None:
    gate_type_dict = {}

    total_gate_count = 0

    for circuit in circuits:
        for gate in circuit.gates:
            GateType = type(gate).__name__
            if GateType in gate_type_dict:
                gate_type_dict[GateType] += 1
            else:
                gate_type_dict[GateType] = 1

        total_gate_count += len(circuit.gates)

    logging.info(f"Distribution of gate types: {gate_type_dict}")

# Class will be added in future version of ga4qc.


class CPhase(IOptimizableGate):
    controll: int
    target: int
    theta: float

    def __init__(self, controll: int = 0, target: int = 1, theta: float = 0.0):
        self.controll = controll
        self.target = target
        self.theta = theta

    def randomize(self, qubit_num: int) -> IGate:
        assert (
            qubit_num > 1
        ), "The CPhase Gate requires at least 2 qubits to operate as intended."

        self.target, self.controll = random.sample(range(0, qubit_num), 2)

        # Choose theta randomly, since theta = 0 is often a stationary
        # point and fails numerical optimizers to progress.
        self.theta = random.random() * 2 * np.pi - np.pi

        return self

    @property
    def params(self) -> List[float]:
        return [self.theta]

    def set_params(self, params: List[float]) -> None:
        assert len(params) == 1, "The CPhase gate requires exactly one parameter!"

        self.theta = params[0]

    def __repr__(self):
        return f"CPhase(control={self.controll}, target={self.target}, theta={round(self.theta, 3)})"


def create_qft_unitary(qubit_num: int) -> np.ndarray:
    dim = 2 ** qubit_num

    dft_matrix = np.zeros((dim, dim), dtype=np.complex128)

    w = np.pow(np.e, 2 * np.pi * 1j / dim)

    for i in range(dim):
        for j in range(dim):
            dft_matrix[i, j] = np.pow(w, i * j)

    unitary = 1 / np.pow(dim, 0.5) * dft_matrix
    return unitary


class LogFitnessStats(FitnessStatsCallback):
    def handle(
        self, fit_means, fit_mins, fit_maxs, fit_stdevs, generation=None
    ) -> None:
        logging.info(
            f"Best fitness at generation {generation}: {fit_mins[0]}")
        logging.info(
            f"Mean fitness at generation {generation}: {fit_means[0]}")


class LogBestCircuit(BestCircuitCallback):
    def handle(self, circuits: List[Circuit], generation=None):
        logging.info(
            f"Best circuit at generation {generation}: {circuits[0]}")


def get_quapsim_gate(gate: IGate) -> QuapsimGate:
    if type(gate) is X:
        return quapsim.gates.X(gate.target)
    if type(gate) is Y:
        return quapsim.gates.Y(gate.target)
    if type(gate) is Z:
        return quapsim.gates.Z(gate.target)
    elif type(gate) is H:
        return quapsim.gates.H(gate.target)
    elif type(gate) is RX:
        return quapsim.gates.RX(gate.target, gate.theta)
    elif type(gate) is RY:
        return quapsim.gates.RY(gate.target, gate.theta)
    elif type(gate) is RZ:
        return quapsim.gates.RZ(gate.target, gate.theta)
    elif type(gate) is CRX:
        return quapsim.gates.CRX(control_qubit=gate.controll, target_qubit=gate.target, theta=gate.theta)
    elif type(gate) is CRY:
        return quapsim.gates.CRY(control_qubit=gate.controll, target_qubit=gate.target, theta=gate.theta)
    elif type(gate) is CRZ:
        return quapsim.gates.CRZ(control_qubit=gate.controll, target_qubit=gate.target, theta=gate.theta)
    elif type(gate) is CX:
        return quapsim.gates.CX(control_qubit=gate.controll, target_qubit=gate.target)
    elif type(gate) is CZ:
        return quapsim.gates.CZ(control_qubit=gate.controll, target_qubit=gate.target)
    elif type(gate) is CY:
        return quapsim.gates.CY(control_qubit=gate.controll, target_qubit=gate.target)
    elif type(gate) is Swap:
        return quapsim.gates.Swap(qubit1=gate.target1, qubit2=gate.target2)
    elif type(gate) is CCZ:
        return quapsim.gates.CCZ(
            control_qubit1=gate.controll1,
            control_qubit2=gate.controll2,
            target_qubit=gate.target
        )
    elif type(gate) is CCX:
        return quapsim.gates.CCX(
            control_qubit1=gate.controll1,
            control_qubit2=gate.controll2,
            target_qubit=gate.target
        )
    elif type(gate) is S:
        return quapsim.gates.S(gate.target)
    elif type(gate) is Phase:
        return quapsim.gates.Phase(gate.target, gate.theta)
    elif type(gate) is CPhase:
        return quapsim.gates.CPhase(gate.controll, gate.target, gate.theta)
    elif type(gate) is T:
        return quapsim.gates.T(gate.target)
    elif type(gate) is Identity:
        return quapsim.gates.Identity(gate.target)
    else:
        raise NotImplementedError(
            f"The gate of type {type(gate)} does not "
            "have a corresponding mapping in quapsim specified."
        )


class QuapsimSimulator(ISimulator):
    simulator: QuaPSim

    def __init__(self, simulator: QuaPSim):
        self.simulator = simulator

    def process(self, circuits: List[Circuit], generation: int) -> None:
        quapsim_circuits: List[QuapsimCircuit] = [
            ga4qc_to_quapsim(circuit) for circuit in circuits
        ]

        logging.info(
            f"Population redundancy in generation {generation}: {compute_redundancy(quapsim_circuits)}"
        )

        log_gate_types(quapsim_circuits)

        if self.simulator.params.cache_size > 0 and (generation - 1 ) % 10 == 0:
            self.simulator.build_cache(quapsim_circuits)

        if self.simulator.params.cache_size > 0:
            self.simulator.simulate_using_cache(
                quapsim_circuits, set_unitary=True)
        else:
            self.simulator.simulate_without_cache(
                quapsim_circuits, set_unitary=True)

        for circuit, quapsim_circuit in zip(circuits, quapsim_circuits):
            circuit.unitaries = [
                quapsim_circuit.unitary
            ]


def ga4qc_to_quapsim(circuit: Circuit) -> QuapsimCircuit:
    quapsim_circuit = QuapsimCircuit(circuit.qubit_num)

    for gate in circuit.gates:
        quapsim_gate = get_quapsim_gate(gate)
        quapsim_circuit.apply(quapsim_gate)

    return quapsim_circuit


ALLOWED_THETAS = [
    2 * np.pi / 2 ** i for i in range(5 + 1)
]


def random_gate(gate_types: Type[IGate], qubit_num: int) -> IGate:
    GateType = choice(gate_types)
    gate = GateType().randomize(qubit_num)

    if issubclass(gate.__class__, IOptimizableGate):
        theta = random.choice(ALLOWED_THETAS)
        gate.set_params([theta])

    return gate


class RandomSeeder2(ISeeder):
    params: GAParams

    def __init__(self, params: GAParams):
        self.params = params

    def seed(self, population_size: int) -> List[Circuit]:
        population = []

        for _ in range(population_size):
            gates = []

            for _ in range(self.params.chromosome_length):
                gate = random_gate(self.params.gate_set, self.params.qubit_num)
                gates.append(gate)

            circuit = Circuit(gates, self.params.qubit_num)

            population.append(circuit)

        return population


class RandomGateMutation2(IMutation):

    prob: float
    gate_prob: float

    ga_params: GAParams

    def __init__(
        self,
        params: GAParams,
        circ_prob: float = 1.0,
        gate_prob: float = 0.1,
    ):
        self.ga_params = params

        self.prob = circ_prob
        self.gate_prob = gate_prob

    def mutate(self, circuit: Circuit, generation: int) -> None:
        for i, gate in enumerate(circuit.gates):
            if random.random() < self.gate_prob:
                circuit.gates[i] = random_gate(
                    self.ga_params.gate_set, self.ga_params.qubit_num)


@click.command()
@click.option(
    "--cache-size",
    "-cs",
    type=click.INT,
    help="The amount of unitaries stored in the cache.",
)
@click.option(
    "--merging-rounds",
    "-mr",
    type=click.INT,
    help="The amount of merging rounds used to build the cache.",
)
@click.option(
    "--seed",
    "-s",
    type=click.INT,
    default=0,
    help="The seed value used for pythons random module.",
)
@click.option(
    "--tag",
    "-t",
    type=click.STRING,
    default=None,
    help="An optional tag that is logged alongside the experiment config for later identification.",
)
def run_experiment(
    cache_size,
    merging_rounds,
    seed,
    tag,
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        filename=f"results/experiment_{tag}_{str(uuid4())}.log",
        filemode="w",
    )

    if seed is not None:
        random.seed(seed)

    cache = SimpleDictCache()
    params = SimulatorParams(
        processes=1,
        cache_size=cache_size,
        merging_rounds=merging_rounds,
    )
    simulator = QuaPSim(params, cache)

    qubit_num = 4  # 6

    # Required gate count of gold solution is
    # (#qubits/2 + 0.5) * #qubits + #qubits/2

    ga_params = GAParams(
        population_size=5000,
        chromosome_length=15,
        generations=10000,
        qubit_num=qubit_num,
        ancillary_qubit_num=0,
        elitism_count=10,
        gate_set=[Identity, H, CPhase, Swap, Phase, X, Y, Z, CX, CY, CZ]
    )

    target_unitary = create_qft_unitary(qubit_num)

    logging.info(
        (
            f"Starting experiment with cache_size={cache_size}, "
            f"merging_rounds={merging_rounds}, seed={seed}, tag={tag}, population_size={ga_params.population_size}, "
            f"generations={ga_params.generations}, chromosome_length={ga_params.chromosome_length}"
        )
    )

    seeder = RandomSeeder2(ga_params)

    ga = GA(
        seeder=seeder,
        mutations=[
            RandomGateMutation2(ga_params,
                                circ_prob=1, gate_prob=0.02)
        ],
        crossovers=[OnePointCrossover(0.5)],
        processors=[
            QuapsimSimulator(simulator),
            AbsoluteUnitaryDistance(
                params=ga_params,
                target_unitaries=[target_unitary]
            ),
        ],
        selection=TournamentSelection(tourn_size=2, objective_i=0),
    )

    ga.on_after_generation(LogFitnessStats())
    ga.on_after_generation(LogBestCircuit())

    # Avoid myrrad of warnings after recent macos update
    # https://github.com/numpy/numpy/issues/28687
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        ga.run(ga_params)


if __name__ == "__main__":
    run_experiment()
