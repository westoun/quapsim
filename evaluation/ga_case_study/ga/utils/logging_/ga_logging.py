from dataclasses import dataclass
import os
from typing import Tuple, Any

from ga.params import ExperimentParams
from .simulator_logging import SimulatorResults
from .utils import get_timestamp, save_to_json, load_from_json


@dataclass
class GaResults:
    best_fitness: Tuple[float]
    population_redundancy: float


def log_epoch_results(generation: int, ga_results: GaResults, simulator_results: SimulatorResults, target_path_prefix: str) -> None:
    target_path = f"{target_path_prefix}_results.csv"

    add_header = not os.path.exists(target_path)

    with open(target_path, "a") as target_file:

        if add_header:
            header = "generation"

            for i in range(len(ga_results.best_fitness)):
                header += f"; fitness #{i}"

            header += "; population redundancy; build cache duration; simulate duration; cache hits; bigram hit count; average cache hit length;"

            target_file.write(header + "\n")

        line = f"{generation}"

        for value in ga_results.best_fitness:
            line += f"; {value}"

        line += f"; {ga_results.population_redundancy}; {simulator_results.build_cache_duration}; {simulator_results.simulate_duration}"
        line += f"; {simulator_results.cache_hits}; {simulator_results.bigram_hit_count}; {simulator_results.avg_cache_hit_length}"

        target_file.write(line + "\n")


def log_experiment_params(params: ExperimentParams) -> None:
    target_path = f"{params.results_path_prefix}_config.json"

    config = {
        "meta": {
            "start": get_timestamp(),
            "results_path_prefix": params.results_path_prefix,
            "seed": params.seed,
        },
        "qubit_num": params.qubit_num,
        "gate_count": params.gate_count,
        "target": params.synthesis_target,
        "ga_params": {
            "population_size": params.population_size,
            "mutation_prob": params.mutation_prob,
            "crossover_prob": params.crossover_prob,
            "max_generations": params.max_generations,
            "selection_strategy": params.selection_strategy,
        },
        "caching_params": {
            "cache_rebuild_frequency": params.cache_rebuild_frequency,
            "cache_size": params.simulator_params.cache_size,
            "merging_rounds": params.simulator_params.merging_rounds
        },
    }

    save_to_json(config, target_path)


def log_end_date(params: ExperimentParams) -> None:
    target_path = f"{params.results_path_prefix}_config.json"
    config = load_from_json(target_path)

    config["meta"]["end"] = get_timestamp()

    save_to_json(config, target_path)


def fetch_best_fitness(params: ExperimentParams) -> Tuple:
    results_path = f"{params.results_path_prefix}_results.csv"

    with open(results_path, "r") as results_file:

        lines = results_file.readlines()

        fitness_col_ids = []

        col_names = lines[0].split("; ")
        for j, col_name in enumerate(col_names):
            if "fitness" in col_name:
                fitness_col_ids.append(j)

        fitness_values = []

        last_entries = lines[-1].split("; ")
        for col_id in fitness_col_ids:
            col_entry = last_entries[col_id]

            col_entry = col_entry.strip()

            fitness_score = float(col_entry)
            fitness_values.append(fitness_score)

        return fitness_values


def remove_ga_log(params: ExperimentParams) -> None:
    config_path = f"{params.results_path_prefix}_config.json"
    results_path = f"{params.results_path_prefix}_results.csv"

    os.remove(config_path)
    os.remove(results_path)
