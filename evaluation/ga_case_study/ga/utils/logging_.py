import os
from typing import Tuple


def log_epoch_results(generation: int, best_fitness: Tuple[float], target_path: str) -> None:

    add_header = not os.path.exists(target_path)

    with open(target_path, "a") as target_file:

        if add_header:
            header = "generation"

            for i in range(len(best_fitness)):
                header += f"; fitness #{i}"

            target_file.write(header + "\n")

        line = f"{generation}"

        for value in best_fitness:
            line += f"; {value}"

        target_file.write(line + "\n")
