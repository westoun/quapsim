#!/usr/bin/env python3

import click
import logging
import random
from uuid import uuid4

from benchmark.utils import (
    create_random_gate_configs,
    build_quapsim_circuit,
    compute_redundancy,
    adjust_redundancy,
)
from quapsim import QuaPSim, SimulatorParams, SimpleDictCache


@click.command()
@click.option(
    "--circuits",
    "-c",
    "circuit_count",
    type=click.INT,
    help="The amount of circuits that are randomly generated.",
)
@click.option(
    "--gates",
    "-g",
    "gate_count",
    type=click.INT,
    help="The amount of gates per circuit.",
)
@click.option(
    "--qubits",
    "-q",
    "qubit_num",
    type=click.INT,
    help="The amount of qubits per circuit.",
)
@click.option(
    "--redundancy",
    "-r",
    type=click.FLOAT,
    default=None,
    help="The desired degree of redundancy within the population.",
)
@click.option(
    "--cache-size",
    "-cs",
    type=click.INT,
    help="The amount of unitaries stored in the cache.",
)
@click.option(
    "--reordering-steps",
    "-rs",
    type=click.INT,
    help="The amount of reordering steps used per circuit.",
)
@click.option(
    "--merging-rounds",
    "-mr",
    type=click.INT,
    help="The amount of merging rounds used to build the cache.",
)
@click.option(
    "--seed",
    "-s",
    type=click.INT,
    default=0,
    help="The seed value used for pythons random module.",
)
@click.option(
    "--tag",
    "-t",
    type=click.STRING,
    default=None,
    help="An optional tag that is logged alongside the experiment config for later identification.",
)
def run_experiment(
    circuit_count,
    gate_count,
    qubit_num,
    redundancy,
    cache_size,
    reordering_steps,
    merging_rounds,
    seed,
    tag,
):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s: %(message)s",
        filename=f"experiment_{tag}_{circuit_count}c_{gate_count}g_{qubit_num}q_{redundancy}r_{str(uuid4())}.log",
        filemode="w",
    )

    logging.info(
        (
            f"Starting experiment with circuit_count={circuit_count}, gate_count={gate_count}, qubit_num={qubit_num}, "
            f"redundancy={redundancy}, cache_size={cache_size}, reordering_steps={reordering_steps}, "
            f"merging_rounds={merging_rounds}, seed={seed}, tag={tag}"
        )
    )

    if seed is not None:
        random.seed(seed)

    cache = SimpleDictCache()
    params = SimulatorParams(
        processes=1,
        cache_size=cache_size,
        reordering_steps=reordering_steps,
        merging_rounds=merging_rounds,
    )
    simulator = QuaPSim(params, cache)

    circuits = []
    for _ in range(circuit_count):
        gate_configs = create_random_gate_configs(
            gate_count=gate_count, qubit_num=qubit_num
        )
        circuit = build_quapsim_circuit(gate_configs, qubit_num=qubit_num)
        circuits.append(circuit)

    logging.info(
        f"Population redundancy before adjustment: {compute_redundancy(circuits)}"
    )

    if redundancy is not None:
        adjust_redundancy(circuits, target=redundancy)

    logging.info(
        f"Population redundancy after adjustment: {compute_redundancy(circuits)}"
    )

    simulator.evaluate(circuits)


if __name__ == "__main__":
    run_experiment()
