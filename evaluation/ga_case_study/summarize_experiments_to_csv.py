from csv import DictWriter
import json
import os
import pandas as pd
from typing import Dict


def load_json(path: str) -> Dict:
    with open(path, "r") as json_file:
        return json.load(json_file)


def write_experiment_data_to_csv(results_dir: str = "results", target_path: str = "experiments.csv") -> None:
    with open(target_path, "w") as target_file:
        writer = DictWriter(target_file, fieldnames=[
            "path", "tag", "start", "end", "seed", "qubit_num", "gate_count", "target", "mutation_prob", "crossover_prob", "selection_strategy", "cache_rebuild_frequency", "cache_size", "merging_rounds", "actual_generations", "failed"
        ], delimiter=",")
        writer.writeheader()

        for file_name in os.listdir(results_dir):
            # Avoid running into issues with hyperparameter opt results.
            if not file_name.startswith("experiment_"):
                continue

            if not file_name.endswith("_config.json"):
                continue

            config_path = f"{results_dir}/{file_name}"
            config = load_json(config_path)

            path_prefix: str = config["meta"]["results_path_prefix"]
            tag_components = path_prefix.split("_")
            tag = "_".join(tag_components[1:-1])

            data_path = f"{results_dir}/{file_name.replace('_config.json', '_results.csv')}"

            experiment_data = pd.read_csv(data_path, delimiter=";")

            if "end" in config["meta"]:
                end = config["meta"]["end"]
            else:
                end = "unknown"

            experiment = {
                "path": file_name,
                "tag": tag,
                "start": config["meta"]["start"],
                "end": end,
                "seed": config["meta"]["seed"],
                "qubit_num": config["qubit_num"],
                "gate_count": config["gate_count"],
                "target": config["target"],
                "mutation_prob": config["ga_params"]["mutation_prob"],
                "crossover_prob": config["ga_params"]["crossover_prob"],
                "selection_strategy": config["ga_params"]["selection_strategy"],
                "cache_rebuild_frequency": config["caching_params"]["cache_rebuild_frequency"],
                "cache_size": config["caching_params"]["cache_size"],
                "merging_rounds": config["caching_params"]["merging_rounds"],
                "actual_generations": len(experiment_data),
                "failed": len(experiment_data) < 5000
            }
            writer.writerow(experiment)


if __name__ == "__main__":
    write_experiment_data_to_csv(results_dir="evaluation/ga_case_study/results/results/", target_path="evaluation/ga_case_study/experiments.csv")
