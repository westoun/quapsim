
import numpy as np

from dataclasses import dataclass
from quapsim import SimulatorParams


@dataclass
class ExperimentParams:
    qubit_num: int
    gate_count: int
    population_size: int
    mutation_prob: float
    crossover_prob: float
    max_generations: int
    simulator_params: SimulatorParams
    target_unitary: np.ndarray
    selection_strategy: str
    results_path: str
