
from copy import deepcopy
from statistics import mean, stdev
import random
from typing import List, Type, Tuple
from tqdm import tqdm

from dataclasses import dataclass
from quapsim.gates import IGate
from quapsim import QuaPSim


from utils.random_ import random_circuit
from mutation import ReplaceGateMutation
from crossover import TwoPointCrossover
from selection import ISelection
from fitness import Fitness

@dataclass
class GaParams:
    qubit_num: int 
    gate_count: int
    population_size: int
    mutation_prob: float 
    crossover_prob: float
    max_generations: int
    simulator: QuaPSim
    selection_strategy: str

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
        self.fitness = Fitness()

        # init different selection strategies


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
        for generation in tqdm(range(1, self.params.max_generations + 1), leave=False, desc="Generation"):
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

            population = self.selection.select(offspring, fitness_scores)

            elite = [population[0]]

            # Note: Fitness of best circuit is not necessarily the best 
            # fitness for each category, if NSGA-X is used.

            # Log results.

        # Remove log file to avoid disk blowup.