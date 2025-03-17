#!/usr/bin/env python3

from dataclasses import dataclass


@dataclass
class SimulatorParams:
    processes: int = 1
    cache_size: int = 0  # For now, measured in # matrices.
    merging_rounds: int = 50


DEFAULT_PARAMS: SimulatorParams = SimulatorParams(
    processes=1, cache_size=0
)
