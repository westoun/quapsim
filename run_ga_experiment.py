#!/usr/bin/env python3

import click
from datetime import datetime
from functools import partial
import logging
import numpy as np
import random
from scipy.optimize import minimize, OptimizeResult
from typing import List, Tuple
from uuid import uuid4

from ga4qc.ga import GA
from ga4qc.seeder import RandomSeeder
from ga4qc.callback import (
    ICallback,
    FitnessStatsCallback,
    BestCircuitCallback,
    UniqueCircuitCountCallback,
)
from ga4qc.circuit import Circuit, update_params, extract_params
from ga4qc.processors import (
    IFitness,
    JensenShannonFitness,
    ISimulator,
    AbsoluteUnitaryDistance,
    WilliamsRankingFitness,
    WeightedSumFitness,
    RemoveDuplicates,
    ICircuitProcessor,
    GateCountFitness,
    NumericalOptimizer
)
from ga4qc.mutation import RandomGateMutation, ParameterChangeMutation
from ga4qc.crossover import OnePointCrossover
from ga4qc.selection import ISelection, TournamentSelection
from ga4qc.circuit import Circuit
from ga4qc.circuit.gates import Identity, CX, S, T, H, X, RX, CCZ, CZ, CCX, \
    Z, Y, IGate, OracleConstructor, Oracle, CY, RY, RZ, CRX, CRY, CRZ, \
    CLIFFORD_PLUS_T, Phase, IOptimizableGate, Swap
from ga4qc.params import GAParams

from quapsim import QuaPSim, SimulatorParams, SimpleDictCache
from quapsim import Circuit as QuapsimCircuit
from quapsim.gates import Gate as QuapsimGate
import quapsim.gates
from benchmark.utils import (
    compute_redundancy,
)


def get_bounds(params: List[float]) -> List[Tuple[float, float]]:
    bounds = []
    for _ in params:
        bounds.append((-2 * np.pi, 2 * np.pi))
    return bounds

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

        self.target, self.control = random.sample(range(0, qubit_num), 2)

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


def create_cnx_unitary(qubit_num: int) -> np.ndarray:
    dim = 2 ** qubit_num

    unitary = np.zeros((dim, dim), dtype=np.complex128)

    for i in range(dim):
        if i == dim - 1:
            unitary[i, i - 1] = 1
        elif i == dim - 2:
            unitary[i, i + 1] = 1
        else:
            unitary[i, i] = 1

    return unitary


def evaluate(
    params: List[float], circuit: Circuit, simulator: QuaPSim, fitness: IFitness, generation: int
) -> float:
    circuit.reset()
    update_params(circuit, params)
    simulate(circuits=[circuit], simulator=simulator)
    fitness.process([circuit], generation)
    score = circuit.fitness_values[0]
    return score


def map_circuits(circuits: List[Circuit]) -> List[QuapsimCircuit]:
    quapsim_circuits: List[QuapsimCircuit] = []

    for circuit in circuits:
        flattened_circuits: List[QuapsimCircuit] = ga4qc_to_quapsim(
            circuit)

        quapsim_circuits.extend(flattened_circuits)

    return quapsim_circuits


def build_cache(circuits: List[Circuit], simulator: QuaPSim) -> None:
    quapsim_circuits = map_circuits(circuits)
    simulator.build_cache(quapsim_circuits)


def simulate(circuits: List[Circuit], simulator: QuaPSim) -> None:
    quapsim_circuits = map_circuits(circuits)

    if simulator.params.cache_size > 0:
        simulator.simulate_using_cache(
            quapsim_circuits, set_unitary=True)
    else:
        simulator.simulate_without_cache(
            quapsim_circuits, set_unitary=True)

    for circuit, quapsim_circuit in zip(circuits, quapsim_circuits):
        circuit.unitaries = [quapsim_circuit.unitary]


def log_redundancy(circuits: List[Circuit], generation: int) -> None:
    quapsim_circuits = map_circuits(circuits)
    logging.info(
        f"Population redundancy in generation {generation}: {compute_redundancy(quapsim_circuits)}"
    )


class QuapsimNumericalOptimizer(NumericalOptimizer):
    simulator: QuaPSim
    fitness: IFitness
    rounds: int

    def __init__(self, simulator: QuaPSim, fitness: IFitness, rounds: int = 10):
        self.simulator = simulator
        self.fitness = fitness
        self.rounds = rounds

    def process(self, circuits: List[Circuit], generation: int) -> None:
        log_redundancy(circuits, generation)

        # GA4QC starts counting at 1.
        if (generation - 1) % 10 == 0 and self.simulator.params.cache_size > 0:
            build_cache(circuits, self.simulator)

        # Workaround to avoid logging calls from simulator whenever a single
        # circuit is simulated.
        logging.disable(logging.CRITICAL)
        start = datetime.now()

        for circuit in circuits:
            initial_params = extract_params(circuit)

            if len(initial_params) == 0:
                simulate(circuits=[circuit], simulator=self.simulator)
                self.fitness.process([circuit], generation)
                continue

            bounds = get_bounds(initial_params)

            objective_function = partial(
                evaluate,
                circuit=circuit,
                simulator=self.simulator,
                fitness=self.fitness,
                generation=generation
            )

            optimization_result: OptimizeResult = minimize(
                objective_function,
                x0=initial_params,
                method="Nelder-Mead",
                bounds=bounds,
                tol=0,
                options={"maxiter": self.rounds, "disp": False},
            )

            best_params = optimization_result.x
            update_params(circuit, best_params)

        end = datetime.now()
        duration = end - start
        logging.disable(logging.NOTSET)

        if self.simulator.params.cache_size > 0:
            logging.info(f"Executing simulate_using_cache took {duration}.")
        else:
            logging.info(f"Executing simulate_without_cache took {duration}.")


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
    elif type(gate) is Phase:
        return quapsim.gates.Phase(gate.target, gate.theta)
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
    elif type(gate) is Swap:
        return quapsim.gates.Swap(qubit1=gate.target1, qubit2=gate.target2)
    elif type(gate) is CY:
        return quapsim.gates.CY(control_qubit=gate.controll, target_qubit=gate.target)
    elif type(gate) is CPhase:
        return quapsim.gates.CPhase(control_qubit=gate.controll, target_qubit=gate.target, theta=gate.theta)
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
        return quapsim.gates.Identity(gate.target)
    else:
        raise NotImplementedError(
            f"The gate of type {type(gate)} does not "
            "have a corresponding mapping in quapsim specified."
        )


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
        filename=f"experiment_{tag}_{str(uuid4())}.log",
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

    qubit_num = 4

    # Required gate count of gold solution is
    # (#qubits/2 + 0.5) * #qubits

    ga_params = GAParams(
        population_size=500,
        chromosome_length=qubit_num**2,
        generations=200,
        qubit_num=qubit_num,
        ancillary_qubit_num=0,
        elitism_count=10,
        gate_set=[Identity, H,
                  Phase, CPhase, Swap]
    )

    target_unitary = create_qft_unitary(qubit_num)

    logging.info(
        (
            f"Starting experiment with cache_size={cache_size}, "
            f"merging_rounds={merging_rounds}, seed={seed}, tag={tag}, population_size={ga_params.population_size}, "
            f"generations={ga_params.generations}, chromosome_length={ga_params.chromosome_length}"
        )
    )

    seeder = RandomSeeder(ga_params)

    ga = GA(
        seeder=seeder,
        mutations=[
            RandomGateMutation(ga_params,
                               circ_prob=0.3, gate_prob=0.2)
        ],
        crossovers=[OnePointCrossover(prob=0.5)],
        processors=[
            RemoveDuplicates(seeder),
            QuapsimNumericalOptimizer(
                simulator=simulator,
                fitness=AbsoluteUnitaryDistance(
                    params=ga_params,
                    target_unitaries=[target_unitary]
                ),
                rounds=5
            ),
            # GateCountFitness(),
            # WeightedSumFitness(weights=[1, 0.01])
        ],
        selection=TournamentSelection(tourn_size=2, objective_i=0),
    )

    ga.on_after_generation(LogFitnessStats())
    ga.on_after_generation(LogBestCircuit())

    ga.run(ga_params)


if __name__ == "__main__":
    run_experiment()
