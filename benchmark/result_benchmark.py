#!/usr/bin/env python3

import logging
from scipy.spatial import distance
import warnings

from quasim import QuaSim
from quapsim import QuaPSim, SimulatorParams, SimpleDictCache
from .utils import create_random_circuits


def run_result_benchmark(
    quasim: QuaSim, quapsim: QuaPSim, circuit_count=100, gate_count=40, qubit_num=4
):
    logging.info("Starting to run result benchmarking.")

    quapsim_circuits = []
    quasim_circuits = []

    for _ in range(circuit_count):
        quapsim_circuit, quasim_circuit = create_random_circuits(
            gate_count=gate_count, qubit_num=qubit_num
        )
        quapsim_circuits.append(quapsim_circuit)
        quasim_circuits.append(quasim_circuit)

    quapsim.evaluate(quapsim_circuits)
    quasim.evaluate(quasim_circuits)

    for quapsim_circuit, quasim_circuit in zip(quapsim_circuits, quasim_circuits):

        quapsim_probabilities = quapsim_circuit.probabilities.tolist()
        quasim_probabilities = quasim_circuit.probabilities.tolist()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            error = distance.jensenshannon(quapsim_probabilities, quasim_probabilities)

        if error > 0.01:
            print(f"\nEncountered strong divergence ({error}) on")
            print(f"\t{quapsim_circuit}")
            print(f"\t{quasim_circuit}")
            print(quapsim_probabilities)
            print(quasim_probabilities)
            break

    else:
        print(
            f"\nFinished result benchmarking. No significant divergences between quapsim and quasim encountered."
        )
        print(
            f"\t Cache type: {type(quapsim.cache)}, Simulator params: {quapsim.params}"
        )
