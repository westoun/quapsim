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
from ga4qc.circuit.gates import Identity, CX, CS, CT, S, T, H, X, RX, CCZ, CZ, CCX, \
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


def construct_bell_state_dist(qubit_num: int) -> np.ndarray:
    dim = 2 ** qubit_num

    dist = np.zeros(dim)

    dist[0] = 0.5
    dist[dim - 1] = 0.5

    return dist


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


class QuapsimSimulator(ISimulator):
    simulator: QuaPSim

    def __init__(self, simulator: QuaPSim):
        self.simulator = simulator

    def process(self, circuits: List[Circuit], generation: int) -> None:
        log_gate_types(circuits)

        quapsim_circuits: List[QuapsimCircuit] = [
            ga4qc_to_quapsim(circuit) for circuit in circuits
        ]

        logging.info(
            f"Population redundancy in generation {generation}: {compute_redundancy(quapsim_circuits)}"
        )

        if self.simulator.params.cache_size > 0:
            self.simulator.build_cache(quapsim_circuits)

        if self.simulator.params.cache_size > 0:
            self.simulator.simulate_using_cache(
                quapsim_circuits, set_unitary=True)
        else:
            self.simulator.simulate_without_cache(
                quapsim_circuits, set_unitary=True)

        for circuit, quapsim_circuit in zip(circuits, quapsim_circuits):
            circuit.unitaries = [quapsim_circuit.unitary]


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


def ga4qc_to_quapsim(circuit: Circuit) -> QuapsimCircuit:
    quapsim_circuit = QuapsimCircuit(circuit.qubit_num)

    for gate in circuit.gates:
        quapsim_gate = get_quapsim_gate(gate)
        quapsim_circuit.apply(quapsim_gate)

    return quapsim_circuit


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
    elif type(gate) is CS:
        return quapsim.gates.CS(control_qubit=gate.controll, target_qubit=gate.target)
    elif type(gate) is CT:
        return quapsim.gates.CT(control_qubit=gate.controll, target_qubit=gate.target)
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

    qubit_num = 9

    ga_params = GAParams(
        population_size=1000,
        chromosome_length=15,
        generations=200,
        qubit_num=qubit_num,
        ancillary_qubit_num=0,
        elitism_count=5,
        gate_set=[Identity, H, X, Y, Z, CX, CY, CZ, S, T, CS, CT, Swap]
        # gate_set=CLIFFORD_PLUS_T + [Identity]
    )

    target_dist = construct_bell_state_dist(qubit_num)

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
                               circ_prob=0.3, gate_prob=0.1)
        ],
        crossovers=[OnePointCrossover(prob=0.5)],
        processors=[
            QuapsimSimulator(simulator=simulator),
            JensenShannonFitness(
                params=ga_params,
                target_dists=[target_dist]
            ),
        ],
        selection=TournamentSelection(tourn_size=2, objective_i=0),
    )

    ga.on_after_generation(LogFitnessStats())
    ga.on_after_generation(LogBestCircuit())

    ga.run(ga_params)


if __name__ == "__main__":
    run_experiment()
