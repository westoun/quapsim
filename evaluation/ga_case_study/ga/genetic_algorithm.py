
from copy import deepcopy
import numpy as np
import random
from statistics import mean, stdev
from typing import List, Type, Tuple

from dataclasses import dataclass
from quapsim.gates import IGate
from quapsim import Circuit
from quapsim import QuaPSim, SimpleDictCache
from quapsim.simulator.utils import compute_redundancy


from .utils.random_ import random_circuit
from .utils.logging_ import log_epoch_results, GaResults, SimulatorResults, \
    fetch_simulator_results, reset_simulator_log, remove_simulator_log, \
    log_experiment_params, log_end_date
from .mutation import ReplaceGateMutation
from .crossover import TwoPointCrossover
from .selection import ISelection, RouletteSelection, \
    TournamentSelection, NSGA2Selection
from .fitness import Fitness
from .params import ExperimentParams


class GeneticAlgorithm:
    params: ExperimentParams
    mutation: ReplaceGateMutation
    crossover: TwoPointCrossover
    simulator: QuaPSim
    fitness: Fitness
    selection: ISelection

    def __init__(self, params: ExperimentParams):
        self.params = params

        self.mutation = ReplaceGateMutation(params.qubit_num)
        self.crossover = TwoPointCrossover()
        self.fitness = Fitness(
            target_unitary=params.target_unitary
        )

        cache = SimpleDictCache()

        self.simulator = QuaPSim(params.simulator_params, cache=cache)

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
        log_experiment_params(self.params)

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

            self._evaluate(generation, offspring)

            offspring.extend(elite)

            fitness_scores: List[Tuple] = self.fitness.score(offspring)

            population, fitness_scores = self.selection.select(
                offspring, fitness_scores)

            elite = [population[0]]

            population_redundancy = compute_redundancy(population)

            # Note: Fitness of best circuit is not necessarily the best
            # fitness for each category, if NSGA-X is used.
            ga_results = GaResults(
                best_fitness=fitness_scores[0],
                population_redundancy=population_redundancy
            )

            simulator_results = fetch_simulator_results()

            log_epoch_results(
                generation,
                ga_results=ga_results,
                simulator_results=simulator_results,
                target_path_prefix=self.params.results_path_prefix
            )

            reset_simulator_log()

        remove_simulator_log()
        log_end_date(self.params)

    def _evaluate(self, generation: int, population: List[Circuit]) -> None:
        if self.params.simulator_params.cache_size > 0:

            if (generation - 1) % self.params.cache_rebuild_frequency == 0:
                self.simulator.build_cache(population)

            self.simulator.simulate_using_cache(
                population, set_unitary=True
            )

        else:
            self.simulator.simulate_without_cache(
                population, set_unitary=True
            )
