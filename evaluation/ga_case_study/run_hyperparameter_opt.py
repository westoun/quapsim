#!/usr/bin/env python3

from bayes_opt import BayesianOptimization
import click
from functools import partial
import logging
import numpy as np
from os import path
import random
from statistics import median
from typing import List, Tuple, Union, Type
from uuid import uuid4
import warnings

from quapsim import SimulatorParams

from ga import ExperimentParams, GeneticAlgorithm
from ga.utils.logging_ import fetch_best_fitness, remove_ga_log, \
    get_timestamp


def log_optimization_results(
        qubit_num: int,
        gate_count: int,
        selection_strategy: str,
        best_fitness: float,
        cross_prob: float,
        mut_prob: float,
        seed_count: int,
        start_timestamp: str,
        end_timestamp: str,
        target_path: str) -> None:

    add_header = not path.exists(target_path)

    with open(target_path, "a") as target_file:

        if add_header:
            header = "qubit_num; gate_count; selection_strategy; "
            header += "best_fitness; cross_prob; mut_prob; "
            header += "seed_count; start_timestamp; end_timestamp"
            target_file.write(header + "\n")

        line = f"{qubit_num}; {gate_count}; {selection_strategy}; "
        line += f"{best_fitness}; {cross_prob}; {mut_prob}; "
        line += f"{seed_count}; {start_timestamp}; {end_timestamp}"
        target_file.write(line + "\n")


def create_qft_unitary(qubit_num: int) -> np.ndarray:
    dim = 2 ** qubit_num

    dft_matrix = np.zeros((dim, dim), dtype=np.complex128)

    w = np.pow(np.e, 2 * np.pi * 1j / dim)

    for i in range(dim):
        for j in range(dim):
            dft_matrix[i, j] = np.pow(w, i * j)

    unitary = 1 / np.pow(dim, 0.5) * dft_matrix
    return unitary


def estimate_ga_performance(mut_prob: float,
                            cross_prob: float,
                            qubit_num: int,
                            gate_count: int,
                            selection_strategy: str,
                            seed_count: int) -> float:

    log_file_prefix = f"results/hyperparameter_opt_{str(uuid4())}"

    best_fitness_per_seed = []

    for seed in range(seed_count):
        random.seed(seed)
        np.random.seed(seed)

        log_file_path = f"{log_file_prefix}.log"
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s: %(message)s",
            filename=log_file_path,
            filemode="w",
        )

        # Ensure that log file exists, even if it has
        # been removed at end of previous ga run.
        with open(log_file_path, "w") as log_file:
            log_file.write("")

        simulator_params = SimulatorParams(
            cache_size=0
        )

        target_unitary = create_qft_unitary(qubit_num)

        experiment_params = ExperimentParams(
            qubit_num=qubit_num,
            gate_count=gate_count,
            population_size=5000,
            mutation_prob=mut_prob,
            crossover_prob=cross_prob,
            max_generations=100,
            simulator_params=simulator_params,
            cache_rebuild_frequency=1,
            target_unitary=target_unitary,
            selection_strategy=selection_strategy,
            seed=seed,
            results_path_prefix=log_file_prefix
        )

        ga = GeneticAlgorithm(
            params=experiment_params
        )

        # Avoid myrrad of warnings after recent macos update
        # https://github.com/numpy/numpy/issues/28687
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            ga.run()

        fitness_scores: Tuple = fetch_best_fitness(experiment_params)
        best_fitness_per_seed.append(fitness_scores[0])

        remove_ga_log(experiment_params)

    # Return -1 * median since bayesian opt framework was
    # designed for maximization problems.
    return -1 * median(best_fitness_per_seed)


@click.command()
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
    "--seed-count",
    "-sc",
    type=click.INT,
    default=15,
    help="The number of seeds to use for each parameter config. Default is 15.",
)
def run_optimization(
        qubit_num: int,
        gate_count: int,
        selection_strategy: str,
        seed_count: int
):

    start_timestamp = get_timestamp()

    black_box_func = partial(estimate_ga_performance, qubit_num=qubit_num, gate_count=gate_count,
                             selection_strategy=selection_strategy, seed_count=seed_count)

    optimizer = BayesianOptimization(
        f=black_box_func,
        pbounds={
            "cross_prob": (0.0, 1.0),
            "mut_prob": (0.0, 0.1)
        },
        random_state=1,
    )

    optimizer.maximize(
        init_points=5,
        n_iter=15
    )

    best_fitness = -1 * optimizer.max["target"]
    cross_prob = optimizer.max["params"]["cross_prob"]
    mut_prob = optimizer.max["params"]["mut_prob"]

    end_timestamp = get_timestamp()

    log_optimization_results(
        qubit_num=qubit_num,
        gate_count=gate_count,
        selection_strategy=selection_strategy,
        best_fitness=best_fitness,
        cross_prob=cross_prob,
        mut_prob=mut_prob,
        seed_count=seed_count,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        target_path="results/hyperparameter_tuning_results.csv"
    )


if __name__ == "__main__":
    run_optimization()
