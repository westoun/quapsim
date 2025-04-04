
from datetime import datetime
import json
import random

from benchmark.utils import (
    create_random_gate_configs,
    build_quapsim_circuit
)
from quapsim import QuaPSim, SimulatorParams, SimpleDictCache


cache = SimpleDictCache()
params = SimulatorParams(
    processes=1,
    cache_size=0,
    merging_rounds=0,
)
simulator = QuaPSim(params, cache)

MIN_QUBITS = 3
MAX_QUBITS = 8
seed_values = [0, 1, 2, 3]
EVALUATIONS_PER_SETUP = 10000

gate_simulation_durations = {}

for qubit_num in range(MIN_QUBITS, MAX_QUBITS + 1):
    gate_simulation_durations[qubit_num] = {}

    print("")

    for seed_value in seed_values:
        random.seed(seed_value)

        print(f"Starting gate evaluation for {qubit_num} qubits and seed {seed_value}")

        for _ in range(EVALUATIONS_PER_SETUP):

            gate_configs = create_random_gate_configs(
                gate_count=2, qubit_num=qubit_num)
            circuit = build_quapsim_circuit(gate_configs, qubit_num=qubit_num)

            start = datetime.now()

            simulator.simulate_without_cache(
                [circuit], state=None, set_unitary=True)

            end = datetime.now()
            
            duration = end - start
            duration = duration.total_seconds()

            for gate in circuit.gates:
                gate_name = gate.__class__.__name__

                if gate_name in gate_simulation_durations[qubit_num]:
                    gate_simulation_durations[qubit_num][gate_name].append(
                        duration)
                else:
                    gate_simulation_durations[qubit_num][gate_name] = [
                        duration]

with open("experiment_gate_sim_results.json", "w") as target_file:
    json.dump(gate_simulation_durations, target_file)