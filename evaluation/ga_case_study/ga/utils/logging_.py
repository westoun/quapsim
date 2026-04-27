from dataclasses import dataclass
from datetime import datetime
import json
import logging
import os
import re
from statistics import mean
from typing import Tuple, Any

from ga.params import ExperimentParams


def duration_to_seconds(duration: str) -> float:
    hours = int(duration.split(":")[0])
    minutes = int(duration.split(":")[1])
    seconds = float(duration.split(":")[2])

    return seconds + minutes * 60 + hours * 60 * 60


@dataclass
class GaResults:
    best_fitness: Tuple[float]
    population_redundancy: float


@dataclass
class SimulatorResults:

    build_cache_duration: float
    simulate_duration: float

    cache_hits: int
    bigram_hit_count: int
    avg_cache_hit_length: float


def fetch_simulator_results() -> SimulatorResults:
    log_path: str = logging.getLoggerClass().root.handlers[0].baseFilename

    simulator_results = SimulatorResults(
        build_cache_duration=0,
        simulate_duration=0,
        cache_hits=0,
        bigram_hit_count=0,
        avg_cache_hit_length=0,
    )

    cache_hit_lengths = []

    with open(log_path, "r") as log_file:
        for line in log_file:
            build_cache_duration = re.search(
                r"Executing build_cache took ([0-9\.\:]+)\.", line
            )
            if build_cache_duration is not None:
                simulator_results.build_cache_duration = (
                    duration_to_seconds(build_cache_duration.group(1))
                )
                continue

            simulate_without_cache_duration = re.search(
                r"Executing simulate_without_cache took ([0-9\.\:]+).", line
            )
            if simulate_without_cache_duration is not None:
                simulator_results.simulate_duration = duration_to_seconds(
                    simulate_without_cache_duration.group(1)
                )
                continue

            simulate_using_cache_duration = re.search(
                r"Executing simulate_using_cache took ([0-9\.\:]+).", line
            )
            if simulate_using_cache_duration is not None:
                simulator_results.simulate_duration = duration_to_seconds(
                    simulate_using_cache_duration.group(1)
                )
                continue

            simulate_using_cache_step = re.search(
                r"Using (\[.+\]) from cache.", line)
            if simulate_using_cache_step is not None:
                ngram = simulate_using_cache_step.group(1)
                ngram_length = ngram.count(")")

                cache_hit_lengths.append(ngram_length)
                continue

    simulator_results.cache_hits = len(cache_hit_lengths)

    if len(cache_hit_lengths) > 1:
        simulator_results.avg_cache_hit_length = mean(cache_hit_lengths)

    simulator_results.bigram_hit_count = len([
        length for length in cache_hit_lengths if length == 2
    ])

    return simulator_results


def remove_simulator_log() -> None:
    log_path: str = logging.getLoggerClass().root.handlers[0].baseFilename
    os.remove(log_path)


def reset_simulator_log() -> None:
    log_path: str = logging.getLoggerClass().root.handlers[0].baseFilename
    with open(log_path, "w") as log_file:
        log_file.write("")


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


def save_to_json(obj, path: str) -> None:
    with open(path, "w") as config_file:
        json.dump(obj, config_file)


def load_from_json(path: str) -> Any:
    with open(path, "r") as config_file:
        return json.load(config_file)


def get_timestamp() -> str:
    return str(datetime.now())


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
