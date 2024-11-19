#!/usr/bin/env python3


import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s: %(message)s", 
    filename="main.log", filemode="w"
)

from quapsim import QuaPSim, SimulatorParams, Circuit, ICache, SimpleDictCache
from quapsim.gates import H, CX, X, Swap, create_unitary


if __name__ == "__main__":
    circuit = Circuit(2)
    circuit.apply(H(0))
    circuit.apply(CX(0, 1))

    simulator = QuaPSim(SimulatorParams(1, 0))
    simulator.evaluate([circuit])

    print(circuit.state)
    print(circuit.probabilities)
    print(circuit.probability_dict)
