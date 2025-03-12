#!/usr/bin/env python3

import click
import logging
import numpy as np
import random
from uuid import uuid4
from typing import List

from ga4qc.ga import GA
from ga4qc.seeder import RandomSeeder
from ga4qc.callback import (
    ICallback,
    FitnessStatsCallback,
    BestCircuitCallback,
    UniqueCircuitCountCallback,
)
from ga4qc.processors import (
    JensenShannonFitness,
    ISimulator
)
from ga4qc.mutation import RandomGateMutation, ParameterChangeMutation
from ga4qc.crossover import OnePointCrossover
from ga4qc.selection import ISelection, TournamentSelection
from ga4qc.circuit import Circuit
from ga4qc.circuit.gates import Identity, CX, S, T, H, X, RX, CCZ, CZ, CCX, \
    Z, Y, IGate, OracleConstructor, Oracle, CY, RY, RZ, CRX, CRY, CRZ

from quapsim import QuaPSim, SimulatorParams, SimpleDictCache
from quapsim import Circuit as QuapsimCircuit
from quapsim.gates import Gate as QuapsimGate
import quapsim.gates
from benchmark.utils import (
    compute_redundancy,
)


def construct_oracle_circuit(target_state: List[int]) -> List[IGate]:
    # 4 qubits => 3 ancillas

    circuit = []

    for i, qubit_state in enumerate(target_state):
        if qubit_state == 0:
            circuit.append(X(i))

    circuit.append(CCX(0, 1, 4))
    circuit.append(CCX(2, 3, 5))
    circuit.append(CCX(4, 5, 6))
    circuit.append(CZ(6, 0))
    circuit.append(CCX(4, 5, 6))
    circuit.append(CCX(2, 3, 5))
    circuit.append(CCX(0, 1, 4))

    for i, qubit_state in enumerate(target_state):
        if qubit_state == 0:
            circuit.append(X(i))

    return circuit


def state_to_distribution(target_state: List[int]) -> List[float]:
    vectors = []
    for qubit_state in target_state:
        if qubit_state == 0:
            vectors.append([1, 0])
        else:
            vectors.append([0, 1])

    distribution = vectors[0]
    for vector in vectors[1:]:
        distribution = np.kron(distribution, vector)

    return distribution.tolist()


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


def ga4qc_to_quapsim(circuit: Circuit) -> List[QuapsimCircuit]:
    quapsim_circuits = []

    for case_i in range(circuit.case_count):
        quapsim_circuit = QuapsimCircuit(circuit.qubit_num)

        for gate in circuit.gates:
            if type(gate) is Oracle:
                for gate_ in gate.get_gates(case_i):
                    quapsim_gate = get_quapsim_gate(gate_)
                    quapsim_circuit.apply(quapsim_gate)
            else:
                quapsim_gate = get_quapsim_gate(gate)
                quapsim_circuit.apply(quapsim_gate)

        quapsim_circuits.append(quapsim_circuit)

    return quapsim_circuits


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
    elif type(gate) is T:
        return quapsim.gates.T(gate.target)
    elif type(gate) is Identity:
        # Workaround because quapsim does not implement an identity
        # gate as of now.
        return quapsim.gates.Phase(gate.target, theta=2 * np.pi)
    else:
        raise NotImplementedError(
            f"The gate of type {type(gate)} does not "
            "have a corresponding mapping in quapsim specified."
        )


class QuapsimSimulator(ISimulator):
    simulator: QuaPSim
    generation: int = -1  # TODO: Remove once generation has been added
    # to process() signature.

    def __init__(self, simulator: QuaPSim):
        self.simulator = simulator

    def process(self, circuits: List[Circuit]) -> None:
        self.generation += 1

        quapsim_circuits: List[QuapsimCircuit] = []
        states_per_circuit: List[int] = []

        for circuit in circuits:
            flattened_circuits: List[QuapsimCircuit] = ga4qc_to_quapsim(
                circuit)

            quapsim_circuits.extend(flattened_circuits)
            states_per_circuit.append(len(flattened_circuits))

        if self.generation % 5 == 0 and self.simulator.params.cache_size > 0:
            self.simulator.build_cache(quapsim_circuits)

        # Add 1 to generation count since ga4qc starts counting at 1.
        logging.info(
            f"Population redundancy in generation {self.generation + 1}: {compute_redundancy(quapsim_circuits)}"
        )

        if self.simulator.params.cache_size > 0:
            self.simulator.simulate_using_cache(
                quapsim_circuits, set_unitary=True)
        else:
            self.simulator.simulate_without_cache(
                quapsim_circuits, set_unitary=True)

        for circuit in circuits:
            circuit.unitaries = []

            states_count = states_per_circuit.pop(0)

            sel_quapsim_circuits = quapsim_circuits[:states_count]
            quapsim_circuits = quapsim_circuits[states_count:]

            for quapsim_circuit in sel_quapsim_circuits:
                circuit.unitaries.append(quapsim_circuit.unitary)


@click.command()
@click.option(
    "--cache-size",
    "-cs",
    type=click.INT,
    help="The amount of unitaries stored in the cache.",
)
@click.option(
    "--reordering-steps",
    "-rs",
    type=click.INT,
    help="The amount of reordering steps used per circuit.",
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
    reordering_steps,
    merging_rounds,
    seed,
    tag,
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        filename=f"experiment_{tag}_{str(uuid4())}.log",
        filemode="w",
    )

    if seed is not None:
        random.seed(seed)

    cache = SimpleDictCache()
    params = SimulatorParams(
        processes=1,
        cache_size=cache_size,
        reordering_steps=reordering_steps,
        merging_rounds=merging_rounds,
    )
    simulator = QuaPSim(params, cache)

    GATE_SET = [H, CX, T, S, CZ, Z, X, Y, CY, CCX, CCZ, Identity,
                RX, RY, RZ, CRX, CRY, CRZ]
    POPULATION_SIZE = 1000
    GENERATIONS = 500
    CHROMOSOME_LENGTH = 50
    ELITISM_COUNT = 50

    logging.info(
        (
            f"Starting experiment with cache_size={cache_size}, reordering_steps={reordering_steps}, "
            f"merging_rounds={merging_rounds}, seed={seed}, tag={tag}, population_size={POPULATION_SIZE}, "
            f"generations={GENERATIONS}, chromosome_length={CHROMOSOME_LENGTH}"
        )
    )

    target_states = [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ]
    target_dists = [state_to_distribution(state) for state in target_states]

    GroverOracle = OracleConstructor(
        sub_circuits=[
            construct_oracle_circuit(state) for state in target_states
        ],
        name="Oracle"
    )
    GATE_SET.append(GroverOracle)

    seeder = RandomSeeder(GATE_SET, gate_count=CHROMOSOME_LENGTH, qubit_num=7)

    ga = GA(
        seeder=seeder,
        mutations=[
            RandomGateMutation(GATE_SET, qubit_num=7,
                               circ_prob=1, gate_prob=0.05)
        ],
        crossovers=[OnePointCrossover()],
        processors=[
            QuapsimSimulator(simulator),
            JensenShannonFitness(target_dists=target_dists,
                                 ancillary_qubit_num=3),
        ],
        selection=TournamentSelection(tourn_size=2),
    )

    ga.on_after_generation(LogFitnessStats())
    ga.on_after_generation(LogBestCircuit())

    ga.run(population_size=POPULATION_SIZE, gate_count=CHROMOSOME_LENGTH,
           generations=GENERATIONS, elitism_count=ELITISM_COUNT)


if __name__ == "__main__":
    run_experiment()
