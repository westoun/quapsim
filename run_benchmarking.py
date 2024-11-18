#!/usr/bin/env python3

from benchmark import run_result_benchmark_non_cached, \
    run_result_benchmark_cached, \
    run_time_benchmark_quasim_quapsim_no_cache, \
    run_time_benchmark_quasim_quapsim_cached, \
    run_time_benchmark_quapsim_cached_vs_no_cache


if __name__ == "__main__":
    run_result_benchmark_non_cached(qubit_num=4)
    run_result_benchmark_cached(qubit_num=4)

    run_time_benchmark_quasim_quapsim_no_cache(qubit_num=4)
    run_time_benchmark_quasim_quapsim_cached(qubit_num=4)
    run_time_benchmark_quapsim_cached_vs_no_cache(qubit_num=4)