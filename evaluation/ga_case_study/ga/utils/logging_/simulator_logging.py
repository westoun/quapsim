from dataclasses import dataclass
import logging
import os
import re
from statistics import mean

from .utils import duration_to_seconds


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
