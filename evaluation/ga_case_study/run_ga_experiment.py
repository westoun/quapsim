#!/usr/bin/env python3

import click
import logging
import numpy as np
import random
from random import choice
from typing import List, Tuple, Union, Type
from uuid import uuid4
import warnings

from quapsim import QuaPSim, SimulatorParams, SimpleDictCache
from quapsim import Circuit as QuapsimCircuit
from quapsim.gates import Gate as QuapsimGate
import quapsim.gates
from quapsim.simulator.utils import (
    compute_redundancy,
)

from ga.utils.random_ import random_circuit
from ga import ExperimentParams, GeneticAlgorithm


def create_random_unitary(qubit_num: int, gate_count: int) -> np.ndarray:
    random_circuit_ = random_circuit(qubit_num, gate_count)
    return random_circuit_.unitary


def create_qft_unitary(qubit_num: int) -> np.ndarray:
    dim = 2 ** qubit_num

    dft_matrix = np.zeros((dim, dim), dtype=np.complex128)

    w = np.pow(np.e, 2 * np.pi * 1j / dim)

    for i in range(dim):
        for j in range(dim):
            dft_matrix[i, j] = np.pow(w, i * j)

    unitary = 1 / np.pow(dim, 0.5) * dft_matrix
    return unitary


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
    "--rebuild-frequency",
    "-rf",
    type=click.INT,
    default=10,
    help="The rebuild frequency of the cache. Default is every 10 generations.",
)
@click.option(
    "--qubit-num",
    "-qn",
    type=click.INT,
    default=6,
    help="The number of qubits per circuit. Default is 6.",
)
@click.option(
    "--gate-count",
    "-gc",
    type=click.INT,
    default=20,
    help="The number of gates per circuit. Default is 20.",
)
@click.option(
    "--selection-strategy",
    "-ss",
    type=click.STRING,
    default="roulette",
    help=("The selection strategy to be used within the GA (must be 'roulette', 'tournament', or 'nsga'). "
          "Default is 'roulette'."),
)
@click.option(
    "--synthesis-target",
    "-st",
    type=click.STRING,
    default="random",
    help="The target of the synthesis (must be 'random' or 'qft'). Default is 'qft'."
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
    rebuild_frequency,
    qubit_num,
    gate_count,
    selection_strategy,
    synthesis_target,
    seed,
    tag,
):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s: %(message)s",
        filename=f"results/experiment_{tag}_{str(uuid4())}.log",
        filemode="w",
    )

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if synthesis_target == "random":
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            target_unitary = create_random_unitary(
                qubit_num=qubit_num, gate_count=gate_count)

    elif synthesis_target == "qft":
        # Required gate count of gold solution is
        # (#qubits/2 + 0.5) * #qubits + #qubits/2
        target_unitary = create_qft_unitary(qubit_num)
    else:
        raise NotImplementedError(
            f"Unknown synthesis target: '{synthesis_target}'")

    simulator_params = SimulatorParams(
        processes=1,
        cache_size=cache_size,
        merging_rounds=merging_rounds,
    )

    experiment_params = ExperimentParams(
        qubit_num=qubit_num,
        gate_count=gate_count,
        population_size=5000,
        mutation_prob=0.02,
        crossover_prob=0.5,
        max_generations=20,
        simulator_params=simulator_params,
        cache_rebuild_frequency=rebuild_frequency,
        target_unitary=target_unitary,
        selection_strategy=selection_strategy,
        seed=seed,
        results_path_prefix=f"results/experiment_{tag}",
        synthesis_target=synthesis_target
    )

    ga = GeneticAlgorithm(
        params=experiment_params
    )

    # Avoid myrrad of warnings after recent macos update
    # https://github.com/numpy/numpy/issues/28687
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        ga.run()


if __name__ == "__main__":
    run_experiment()
