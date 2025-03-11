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


log_file_folder = "results_ga"

log_file_paths = [
    log_file_folder + "/" + path
    for path in listdir(log_file_folder)
    if path.startswith("experiment") and path.endswith(".log")
]

experiment_data: List[Dict] = []

for log_file_path in log_file_paths:
    experiment = {
        "file_path": log_file_path,
        "duration_in": "seconds",
        "started_at": None,
        "params": {
            "cache_size": None,
            "reordering_steps": None,
            "merging_rounds": None,
            "seed": None,
            "tag": None,
        },
        "generations": []
    }

    new_generation = True
    with open(log_file_path, "r") as log_file:
        for i, line in enumerate(log_file):
            if new_generation:
                experiment["generations"].append({
                    "i": len(experiment["generations"]),
                    "population_redundancy": None,
                    "build_gate_frequency_dict": {"duration": None},
                    "optimize_gate_order": {
                        "duration": None,
                    },
                    "generate_seed_bigrams": {
                        "duration": None,
                    },
                    "consolidate_ngrams": {
                        "duration": None,
                    },
                    "select_ngrams_to_cache": {"duration": None},
                    "fill_cache": {"duration": None, },
                    "simulate_using_cache": {
                        "duration": None,
                        "trie_lookup_duration": None,
                    },
                    "simulate_without_cache": {"duration": None},
                    "mean_fitness": None,
                    "best_fitness": None,
                    "best_circuit": None
                })
                new_generation = False

            experiment_setup = re.search(
                r"cache_size=([0-9]+), reordering_steps=([0-9]+), merging_rounds=([0-9]+), seed=([0-9]+|None), tag=([a-zA-Z_\-]+|None)",
                line,
            )
            if experiment_setup is not None:
                timestamp = " ".join(line.split(" ")[:2])

                cache_size = experiment_setup.group(1)
                reordering_steps = experiment_setup.group(2)
                merging_rounds = experiment_setup.group(3)
                seed = experiment_setup.group(4)
                tag = experiment_setup.group(5)

                experiment["started_at"] = timestamp

                experiment["params"]["cache_size"] = int(cache_size)
                experiment["params"]["reordering_steps"] = int(
                    reordering_steps)
                experiment["params"]["merging_rounds"] = int(merging_rounds)

                if seed != "None":
                    experiment["params"]["seed"] = int(seed)

                if tag != "None":
                    experiment["params"]["tag"] = tag

                continue

            build_gate_frequency_dict_duration = re.search(
                r"Executing _build_gate_frequency_dict took ([0-9\.\:]+).", line
            )
            if build_gate_frequency_dict_duration is not None:
                experiment["generations"][-1]["build_gate_frequency_dict"]["duration"] = (
                    duration_to_seconds(
                        build_gate_frequency_dict_duration.group(1))
                )
                continue

            optimize_gate_order_duration = re.search(
                r"Executing _optimize_gate_order took ([0-9\.\:]+).", line
            )
            if optimize_gate_order_duration is not None:
                experiment["generations"][-1]["optimize_gate_order"]["duration"] = duration_to_seconds(
                    optimize_gate_order_duration.group(1)
                )
                continue

            generate_seed_bigrams_duration = re.search(
                r"Executing _generate_seed_bigrams took ([0-9\.\:]+).", line
            )
            if generate_seed_bigrams_duration is not None:
                experiment["generations"][-1]["generate_seed_bigrams"]["duration"] = duration_to_seconds(
                    generate_seed_bigrams_duration.group(1)
                )
                continue

            consolidate_ngrams_duration = re.search(
                r"Executing _consolidate_ngrams took ([0-9\.\:]+).", line
            )
            if consolidate_ngrams_duration is not None:
                experiment["generations"][-1]["consolidate_ngrams"]["duration"] = duration_to_seconds(
                    consolidate_ngrams_duration.group(1)
                )
                continue

            select_ngrams_to_cache_duration = re.search(
                r"Executing _select_ngrams_to_cache took ([0-9\.\:]+).", line
            )
            if select_ngrams_to_cache_duration is not None:
                experiment["generations"][-1]["select_ngrams_to_cache"]["duration"] = duration_to_seconds(
                    select_ngrams_to_cache_duration.group(1)
                )
                continue

            fill_cache_duration = re.search(
                r"Executing _fill_cache took ([0-9\.\:]+).", line
            )
            if fill_cache_duration is not None:
                experiment["generations"][-1]["fill_cache"]["duration"] = duration_to_seconds(
                    fill_cache_duration.group(1)
                )
                continue

            simulate_without_cache_duration = re.search(
                r"Executing simulate_without_cache took ([0-9\.\:]+).", line
            )
            if simulate_without_cache_duration is not None:
                experiment["generations"][-1]["simulate_without_cache"]["duration"] = duration_to_seconds(
                    simulate_without_cache_duration.group(1)
                )
                continue

            simulate_using_cache_duration = re.search(
                r"Executing simulate_using_cache took ([0-9\.\:]+).", line
            )
            if simulate_using_cache_duration is not None:
                experiment["generations"][-1]["simulate_using_cache"]["duration"] = duration_to_seconds(
                    simulate_using_cache_duration.group(1)
                )
                continue

            simulate_using_cache_step = re.search(
                r"Using (\[.+\]) from cache.", line)
            if simulate_using_cache_step is not None:
                ngram = simulate_using_cache_step.group(1)
                ngram_length = ngram.count(")")

                experiment["generations"][-1]["simulate_using_cache"]["hit_ngram_lenghts"].append(
                    ngram_length
                )
                continue

            trie_lookup_duration = re.search(
                r"Time during merging spent on trie lookup: ([0-9\.\:]+)", line
            )
            if trie_lookup_duration is not None:
                experiment["generations"][-1]["simulate_using_cache"]["trie_lookup_duration"] = (
                    duration_to_seconds(trie_lookup_duration.group(1))
                )
                continue

            population_redundancy = re.search(
                r"Population redundancy in generation [0-9]+: ([0-9]+\.[0-9]+)", line)
            if population_redundancy is not None:
                experiment["generations"][-1]["population_redundancy"] = float(
                    population_redundancy.group(1)
                )
                continue

            best_fitness = re.search(
                r"Best fitness at generation [0-9]+: ([0-9]+\.[0-9]+)", line)
            if best_fitness is not None:
                experiment["generations"][-1]["best_fitness"] = float(
                    best_fitness.group(1)
                )
                continue

            mean_fitness = re.search(
                r"Mean fitness at generation [0-9]+: ([0-9]+\.[0-9]+)", line)
            if mean_fitness is not None:
                experiment["generations"][-1]["mean_fitness"] = float(
                    mean_fitness.group(1)
                )
                continue

            best_circuit = re.search(
                r"Best circuit at generation [0-9]+: (\[.+\])", line)
            if best_circuit is not None:
                experiment["generations"][-1]["best_circuit"] = best_circuit.group(
                    1)
                new_generation = True
                continue

    experiment_data.append(experiment)

with open("experiment_ga_results.json", "w") as target_file:
    json.dump(experiment_data, target_file)
