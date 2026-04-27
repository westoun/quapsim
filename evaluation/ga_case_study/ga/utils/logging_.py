from dataclasses import dataclass
import os
from typing import Tuple

@dataclass
class GaResults:
    best_fitness: Tuple[float]
    population_redundancy: float


def log_epoch_results(generation: int, ga_results: GaResults, target_path: str) -> None:

    add_header = not os.path.exists(target_path)

    with open(target_path, "a") as target_file:

        if add_header:
            header = "generation"

            for i in range(len(ga_results.best_fitness)):
                header += f"; fitness #{i}"

            header += "; population redundancy"

            target_file.write(header + "\n")

        line = f"{generation}"

        for value in ga_results.best_fitness:
            line += f"; {value}"

        line += f"; {ga_results.population_redundancy}"

        target_file.write(line + "\n")
