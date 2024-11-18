#!/usr/bin/env python3

from datetime import datetime

from quasim import QuaSim
from quapsim import QuaPSim, SimulatorParams, SimpleDictCache
from .utils import create_random_circuits


def run_time_benchmark_quasim_quapsim_no_cache(
    circuit_count=1000, gate_count=40, qubit_num=3
):
    quasim = QuaSim()
    quapsim = QuaPSim(SimulatorParams(processes=1, cache_size=0))

    quapsim_circuits = []
    quasim_circuits = []

    for _ in range(circuit_count):
        quapsim_circuit, quasim_circuit = create_random_circuits(
            gate_count=gate_count, qubit_num=qubit_num
        )
        quapsim_circuits.append(quapsim_circuit)
        quasim_circuits.append(quasim_circuit)

    start = datetime.now()
    quapsim.evaluate(quapsim_circuits)
    end = datetime.now()
    quapsim_duration = end - start

    start = datetime.now()
    quasim.evaluate(quasim_circuits)
    end = datetime.now()
    quasim_duration = end - start

    print(
        f"Finished evaluation benchmarking on {circuit_count} circuits with {gate_count} gates and {qubit_num} qubits each."
    )
    print("Quapsim duration: ", quapsim_duration)
    print("Quasim duration: ", quasim_duration)


def run_time_benchmark_quasim_quapsim_cached(
    circuit_count=1000, gate_count=40, qubit_num=3
):
    quasim = QuaSim()

    cache = SimpleDictCache()
    quapsim = QuaPSim(SimulatorParams(processes=1, cache_size=1000), cache=cache)  #

    quapsim_circuits = []
    quasim_circuits = []

    for _ in range(circuit_count):
        quapsim_circuit, quasim_circuit = create_random_circuits(
            gate_count=gate_count, qubit_num=qubit_num
        )
        quapsim_circuits.append(quapsim_circuit)
        quasim_circuits.append(quasim_circuit)

    start = datetime.now()
    quapsim.evaluate(quapsim_circuits)
    end = datetime.now()
    quapsim_duration = end - start

    start = datetime.now()
    quasim.evaluate(quasim_circuits)
    end = datetime.now()
    quasim_duration = end - start

    print(
        f"Finished evaluation benchmarking on {circuit_count} circuits with {gate_count} gates and {qubit_num} qubits each."
    )
    print("Quapsim duration: ", quapsim_duration)
    print("Quasim duration: ", quasim_duration)


def run_time_benchmark_quapsim_cached_vs_no_cache(
    circuit_count=1000, gate_count=40, qubit_num=3
):
    cache = SimpleDictCache()
    quapsim_cached = QuaPSim(SimulatorParams(processes=1, cache_size=1000), cache=cache)

    quapsim_no_cache = QuaPSim(SimulatorParams(processes=1, cache_size=0))

    quapsim_circuits = []
    for _ in range(circuit_count):
        quapsim_circuit, _ = create_random_circuits(
            gate_count=gate_count, qubit_num=qubit_num
        )
        quapsim_circuits.append(quapsim_circuit)

    start = datetime.now()
    quapsim_cached.evaluate(quapsim_circuits)
    end = datetime.now()
    quapsim_cached_duration = end - start

    # Reset circuits, since quapsim does not evaluate circuits
    # whose states have already been set.
    for circuit in quapsim_circuits:
        circuit.set_state(None)

    start = datetime.now()
    quapsim_no_cache.evaluate(quapsim_circuits)
    end = datetime.now()
    quapsim_no_cache_duration = end - start

    print(
        f"Finished evaluation benchmarking on {circuit_count} circuits with {gate_count} gates and {qubit_num} qubits each."
    )
    print("Quapsim cached duration: ", quapsim_cached_duration)
    print("Quapsim no cache duration: ", quapsim_no_cache_duration)
