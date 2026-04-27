
from copy import deepcopy
import numpy as np
import random
from statistics import mean, stdev
from typing import List, Type, Tuple

from dataclasses import dataclass
from quapsim.gates import IGate
from quapsim import Circuit
from quapsim import QuaPSim
from quapsim.simulator.utils import compute_redundancy


from .utils.random_ import random_circuit
from .utils.logging_ import log_epoch_results, GaResults
from .mutation import ReplaceGateMutation
from .crossover import TwoPointCrossover
from .selection import ISelection, RouletteSelection, \
    TournamentSelection, NSGA2Selection
from .fitness import Fitness


@dataclass
class GaParams:
    qubit_num: int
    gate_count: int
    population_size: int
    mutation_prob: float
    crossover_prob: float
    max_generations: int
    simulator: QuaPSim
    target_unitary: np.ndarray
    selection_strategy: str
    results_path: str


class GeneticAlgorithm:
    params: GaParams
    mutation: ReplaceGateMutation
    crossover: TwoPointCrossover
    simulator: QuaPSim
    fitness: Fitness
    selection: ISelection

    def __init__(self, params: GaParams):
        self.params = params
        self.mutation = ReplaceGateMutation(params.qubit_num)
        self.crossover = TwoPointCrossover()
        self.simulator = params.simulator
        self.fitness = Fitness(
            target_unitary=params.target_unitary
        )

        # init different selection strategies
        if params.selection_strategy == "tournament":
            self.selection = TournamentSelection(
                n=params.population_size, tournament_size=2)
        elif params.selection_strategy == "roulette":
            self.selection = RouletteSelection(
                n=params.population_size
            )
        elif params.selection_strategy == "nsga":
            pass
        else:
            raise NotImplementedError(
                f"No implementation found for selection strategy '{params.selection_strategy}'")

    def run(self):
        # log params

        population = [
            random_circuit(
                qubit_num=self.params.qubit_num,
                gate_count=self.params.gate_count,
            )
            for _ in range(self.params.population_size)
        ]

        elite = []
        for generation in range(1, self.params.max_generations + 1):
            offspring = [deepcopy(circuit) for circuit in population]

            if generation > 1:
                # Shuffle to avoid crossover in the same proximity across
                # generations.
                random.shuffle(offspring)

                for i in range(0, len(offspring) - 1, 2):
                    if random.random() < self.params.crossover_prob:
                        offspring[i], offspring[i +
                                                1] = self.crossover.cross(offspring[i], offspring[i + 1])

                for i, circuit in enumerate(offspring):
                    for j, gate in enumerate(circuit.gates):
                        if random.random() < self.params.mutation_prob:
                            offspring[i].gates[j] = self.mutation.mutate(
                                gate)

            self.simulator.evaluate(offspring)
            offspring.extend(elite)

            fitness_scores: List[Tuple] = self.fitness.score(offspring)

            population, fitness_scores = self.selection.select(
                offspring, fitness_scores)

            elite = [population[0]]

            population_redundancy = compute_redundancy(population)

            ga_results = GaResults(
                best_fitness=fitness_scores[0],
                population_redundancy=population_redundancy
            )
            log_epoch_results(
                generation,
                ga_results=ga_results,
                target_path=self.params.results_path
            )

            # Note: Fitness of best circuit is not necessarily the best
            # fitness for each category, if NSGA-X is used.

            # Log results.

        # Remove log file to avoid disk blowup.
