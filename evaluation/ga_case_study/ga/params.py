
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
    selection_strategy: str
    simulator_params: SimulatorParams
    cache_rebuild_frequency: int
    target_unitary: np.ndarray
    seed: int
    results_path_prefix: str
