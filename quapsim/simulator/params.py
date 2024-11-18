#!/usr/bin/env python3

from dataclasses import dataclass


@dataclass
class SimulatorParams:
    processes: int = 1
    cache_size: float = 0


DEFAULT_PARAMS: SimulatorParams = SimulatorParams(processes=1, cache_size=0)
