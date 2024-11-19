#!/usr/bin/env python3

from datetime import datetime
import logging

from quasim import QuaSim
from quapsim import QuaPSim, SimulatorParams, SimpleDictCache
from .utils import create_random_circuits


def run_time_benchmark_quasim_quapsim(
    quasim=QuaSim, quapsim=QuaPSim, circuit_count=1000, gate_count=40, qubit_num=3
):
    logging.info("Starting to run time benchmarking, quasim against quapsim.")

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
        f"\nFinished evaluation benchmarking on {circuit_count} circuits with {gate_count} gates and {qubit_num} qubits each."
    )
    print(
        f"Quapsim duration: ",
        quapsim_duration,
    )
    print(f"\t Cache type: {type(quapsim.cache)}, Simulator params: {quapsim.params}")
    print("Quasim duration: ", quasim_duration)


def run_time_benchmark_quapsim_quapsim(
    quapsim1: QuaPSim, quapsim2: QuaPSim, circuit_count=1000, gate_count=40, qubit_num=3
):
    logging.info("Starting to run time benchmarking, quapsim against quapsim.")

    quapsim_circuits = []
    for _ in range(circuit_count):
        quapsim_circuit, _ = create_random_circuits(
            gate_count=gate_count, qubit_num=qubit_num
        )
        quapsim_circuits.append(quapsim_circuit)

    start = datetime.now()
    quapsim1.evaluate(quapsim_circuits)
    end = datetime.now()
    quapsim1_duration = end - start

    # Reset circuits, since quapsim does not evaluate circuits
    # whose states have already been set.
    for circuit in quapsim_circuits:
        circuit.set_state(None)

    start = datetime.now()
    quapsim2.evaluate(quapsim_circuits)
    end = datetime.now()
    quapsim2_duration = end - start

    print(
        f"\nFinished evaluation benchmarking on {circuit_count} circuits with {gate_count} gates and {qubit_num} qubits each."
    )
    print(
        f"Quapsim1 duration: ",
        quapsim1_duration,
    )
    print(f"\t Cache type: {type(quapsim1.cache)}, Simulator params: {quapsim1.params}")
    print(
        f"Quapsim2 duration: ",
        quapsim2_duration,
    )
    print(f"\t Cache type: {type(quapsim2.cache)}, Simulator params: {quapsim2.params}")
