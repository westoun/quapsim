#!/usr/bin/env python3

import json
from os import listdir
import re
from typing import List, Dict


def duration_to_seconds(duration: str) -> float:
    hours = int(duration.split(":")[0])
    minutes = int(duration.split(":")[1])
    seconds = float(duration.split(":")[2])

    return seconds + minutes * 60 + hours * 60 * 60


log_file_paths = [
    path
    for path in listdir(".")
    if path.startswith("experiment") and path.endswith(".log")
]

experiment_data: List[Dict] = []

for log_file_path in log_file_paths:
    experiment = {
        "file_path": log_file_path,
        "duration_in": "seconds",
        "started_at": None,
        "params": {
            "circuit_count": None,
            "gate_count": None,
            "qubit_num": None,
            "redundancy": None,
            "cache_size": None,
            "reordering_steps": None,
            "merging_pool_size": None,
            "merging_rounds": None,
            "seed": None,
            "tag": None,
        },
        "stats": {
            "unique_gate_configs": None,
            "redundancy_before_adjustment": None,
            "redundancy_after_adjustment": None,
        },
        "build_gate_frequency_dict": {"duration": None},
        "optimize_gate_order": {
            "duration": None,
            "redundancy_before_optimization": None,
            "redundancy_after_optimization": None,
        },
        "generate_seed_bigrams": {
            "duration": None,
        },
        "consolidate_ngrams": {
            "duration": None,
            "ngram_lengths": [],
            "ngram_frequencies": [],
        },
        "select_ngrams_to_cache": {"duration": None},
        "fill_cache": {"duration": None, "ngram_lengths": [], "ngram_frequencies": []},
        "simulate_using_cache": {"duration": None, "hit_ngram_lenghts": []},
        "total_duration": None,
    }

    with open(log_file_path, "r") as log_file:
        for i, line in enumerate(log_file):
            if i == 0:
                timestamp = " ".join(line.split(" ")[:2])

                experiment_setup = re.search(
                    r"circuit_count=([0-9]+), gate_count=([0-9]+), qubit_num=([0-9]+), redundancy=([0-9\.]+|None), cache_size=([0-9]+), reordering_steps=([0-9]+), merging_pool_size=([0-9]+), merging_rounds=([0-9]+), seed=([0-9]+|None), tag=([a-zA-Z_\-]|None)",
                    line,
                )

                circuit_count = experiment_setup.group(1)
                gate_count = experiment_setup.group(2)
                qubit_num = experiment_setup.group(3)
                redundancy = experiment_setup.group(4)
                cache_size = experiment_setup.group(5)
                reordering_steps = experiment_setup.group(6)
                merging_pool_size = experiment_setup.group(7)
                merging_rounds = experiment_setup.group(8)
                seed = experiment_setup.group(9)
                tag = experiment_setup.group(10)

                experiment["started_at"] = timestamp

                experiment["params"]["circuit_count"] = int(circuit_count)
                experiment["params"]["gate_count"] = int(gate_count)
                experiment["params"]["qubit_num"] = int(qubit_num)

                if redundancy != "None":
                    experiment["params"]["redundancy"] = float(redundancy)

                experiment["params"]["cache_size"] = int(cache_size)
                experiment["params"]["reordering_steps"] = int(reordering_steps)
                experiment["params"]["merging_pool_size"] = int(merging_pool_size)
                experiment["params"]["merging_rounds"] = int(merging_rounds)

                if seed != "None":
                    experiment["params"]["seed"] = int(seed)

                if tag != "None":
                    experiment["params"]["tag"] = tag

                continue

            # TODO: Avoid checking regex that have been parsed
            unique_gate_configs = re.search(
                r"total amount of ([0-9]+) unique gate configurations", line
            )
            if unique_gate_configs != None:
                experiment["stats"]["unique_gate_configs"] = int(
                    unique_gate_configs.group(1)
                )
                continue

            redundancy_before_adjustment = re.search(
                r"Population redundancy before adjustment: ([0-9\.]+)", line
            )
            if redundancy_before_adjustment is not None:
                experiment["stats"]["redundancy_before_adjustment"] = float(
                    redundancy_before_adjustment.group(1)
                )
                continue

            redundancy_after_adjustment = re.search(
                r"Population redundancy after adjustment: ([0-9\.]+)", line
            )
            if redundancy_after_adjustment is not None:
                experiment["stats"]["redundancy_after_adjustment"] = float(
                    redundancy_after_adjustment.group(1)
                )
                continue

            build_gate_frequency_dict_duration = re.search(
                r"Executing _build_gate_frequency_dict took ([0-9\.\:]+).", line
            )
            if build_gate_frequency_dict_duration is not None:
                experiment["build_gate_frequency_dict"]["duration"] = (
                    duration_to_seconds(build_gate_frequency_dict_duration.group(1))
                )
                continue

            optimize_gate_order_duration = re.search(
                r"Executing _optimize_gate_order took ([0-9\.\:]+).", line
            )
            if optimize_gate_order_duration is not None:
                experiment["optimize_gate_order"]["duration"] = duration_to_seconds(
                    optimize_gate_order_duration.group(1)
                )
                continue

            redundancy_before_optimization = re.search(
                r"Population redundancy before optimization ([0-9\.]+).", line
            )
            if redundancy_before_optimization is not None:
                experiment["optimize_gate_order"]["redundancy_before_optimization"] = (
                    float(redundancy_before_optimization.group(1))
                )
                continue

            redundancy_after_optimization = re.search(
                r"Population redundancy after optimization ([0-9\.]+).", line
            )
            if redundancy_after_optimization is not None:
                experiment["optimize_gate_order"]["redundancy_after_optimization"] = (
                    float(redundancy_after_optimization.group(1))
                )
                continue

            generate_seed_bigrams_duration = re.search(
                r"Executing _generate_seed_bigrams took ([0-9\.\:]+).", line
            )
            if generate_seed_bigrams_duration is not None:
                experiment["generate_seed_bigrams"]["duration"] = duration_to_seconds(
                    generate_seed_bigrams_duration.group(1)
                )
                continue

            consolidate_ngrams_duration = re.search(
                r"Executing _consolidate_ngrams took ([0-9\.\:]+).", line
            )
            if consolidate_ngrams_duration is not None:
                experiment["consolidate_ngrams"]["duration"] = duration_to_seconds(
                    consolidate_ngrams_duration.group(1)
                )
                continue

            consolidate_ngrams_step = re.search(
                r"Adding (\[.+\]) with frequency ([0-9]+) to ngram pool.", line
            )
            if consolidate_ngrams_step is not None:
                ngram = consolidate_ngrams_step.group(1)
                ngram_length = ngram.count(")")
                experiment["consolidate_ngrams"]["ngram_lengths"].append(ngram_length)

                frequency = int(consolidate_ngrams_step.group(2))
                experiment["consolidate_ngrams"]["ngram_frequencies"].append(frequency)
                continue

            select_ngrams_to_cache_duration = re.search(
                r"Executing _select_ngrams_to_cache took ([0-9\.\:]+).", line
            )
            if select_ngrams_to_cache_duration is not None:
                experiment["select_ngrams_to_cache"]["duration"] = duration_to_seconds(
                    select_ngrams_to_cache_duration.group(1)
                )
                continue

            fill_cache_duration = re.search(
                r"Executing _fill_cache took ([0-9\.\:]+).", line
            )
            if fill_cache_duration is not None:
                experiment["fill_cache"]["duration"] = duration_to_seconds(
                    fill_cache_duration.group(1)
                )
                continue

            fill_cache_step = re.search(
                r"Adding (\[.+\]) \(with a frequency of ([0-9]+)\) to cache.", line
            )
            if fill_cache_step is not None:
                ngram = fill_cache_step.group(1)
                ngram_length = ngram.count(")")
                experiment["fill_cache"]["ngram_lengths"].append(ngram_length)

                frequency = int(fill_cache_step.group(2))
                experiment["fill_cache"]["ngram_frequencies"].append(frequency)
                continue

            simulate_using_cache_duration = re.search(
                r"Executing simulate_using_cache took ([0-9\.\:]+).", line
            )
            if simulate_using_cache_duration is not None:
                experiment["simulate_using_cache"]["duration"] = duration_to_seconds(
                    simulate_using_cache_duration.group(1)
                )
                continue

            simulate_using_cache_step = re.search(r"Using (\[.+\]) from cache.", line)
            if simulate_using_cache_step is not None:
                ngram = simulate_using_cache_step.group(1)
                ngram_length = ngram.count(")")

                experiment["simulate_using_cache"]["hit_ngram_lenghts"].append(
                    ngram_length
                )
                continue

            total_duration = re.search(r"Executing evaluate took ([0-9\.\:]+).", line)
            if total_duration is not None:
                experiment["total_duration"] = duration_to_seconds(
                    total_duration.group(1)
                )
                continue

    # research how to serialize datetime/duration
    experiment_data.append(experiment)


with open("experiment_results.json", "w") as target_file:
    json.dump(experiment_data, target_file)
