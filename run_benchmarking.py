#!/usr/bin/env python3

import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s: %(message)s", 
    filename="benchmarking.log", filemode="a"
)

from quasim import QuaSim
from quapsim import QuaPSim, SimulatorParams, SimpleDictCache

from benchmark import (
    run_result_benchmark,
    run_time_benchmark_quasim_quapsim,
    run_time_benchmark_quapsim_quapsim,
)


if __name__ == "__main__":
    quasim = QuaSim()
    cache = SimpleDictCache()
    quapsim = QuaPSim(SimulatorParams(processes=1, cache_size=200), cache=cache)
    run_result_benchmark(quasim=quasim, quapsim=quapsim, qubit_num=4)

    quasim = QuaSim()
    quapsim = QuaPSim(SimulatorParams(processes=1, cache_size=0), cache=None)
    run_result_benchmark(quasim=quasim, quapsim=quapsim, qubit_num=4)

    quasim = QuaSim()
    quapsim = QuaPSim(SimulatorParams(processes=1, cache_size=0))
    run_time_benchmark_quasim_quapsim(quasim=quasim, quapsim=quapsim, qubit_num=4)

    quasim = QuaSim()
    cache = SimpleDictCache()
    quapsim = QuaPSim(SimulatorParams(processes=1, cache_size=200), cache=cache)
    run_time_benchmark_quasim_quapsim(quasim=quasim, quapsim=quapsim, qubit_num=4)

    cache = SimpleDictCache()
    quapsim1 = QuaPSim(SimulatorParams(processes=1, cache_size=200), cache=cache)
    quapsim2 = QuaPSim(SimulatorParams(processes=1, cache_size=0))
    run_time_benchmark_quapsim_quapsim(
        quapsim1=quapsim1, quapsim2=quapsim2, qubit_num=4
    )
