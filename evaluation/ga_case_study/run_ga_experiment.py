#!/usr/bin/env python3

import click
import logging
import numpy as np
import random
from random import choice
from typing import List, Tuple, Union, Type
from uuid import uuid4

from quapsim import QuaPSim, SimulatorParams, SimpleDictCache
from quapsim import Circuit as QuapsimCircuit
from quapsim.gates import Gate as QuapsimGate
import quapsim.gates
from quapsim.simulator.utils import (
    compute_redundancy,
)

from ga import GaParams, GeneticAlgorithm


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
    "--selection-strategy",
    "-ss",
    type=click.STRING,
    default="roulette",
    help=("The selection strategy to be used within the GA (must be 'roulette', 'tournament', or 'nsga'). "
          "Default is 'roulette'."),
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
    selection_strategy,
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
        np.random.seed(seed)

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
    target_unitary = create_qft_unitary(qubit_num)

    ga_params = GaParams(
        qubit_num=qubit_num,
        gate_count=15,
        population_size=5000,
        mutation_prob=0.02,
        crossover_prob=0.5,
        max_generations=100,
        simulator=simulator,
        target_unitary=target_unitary,
        selection_strategy=selection_strategy,
        results_path=f"results/experiment_{tag}_fitness.csv"
    )

    ga = GeneticAlgorithm(
        params=ga_params
    )

    # Avoid myrrad of warnings after recent macos update
    # https://github.com/numpy/numpy/issues/28687
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        ga.run()


if __name__ == "__main__":
    run_experiment()
