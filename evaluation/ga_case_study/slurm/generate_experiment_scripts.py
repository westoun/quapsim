qubit_nums = [6]
gate_counts = [20]
selection_strategies = ["roulette"]
synthesis_targets = ["random"]

rebuild_frequencies = [20]
cache_sizes = [100]
merging_rounds = [20]

seed_num = 1
seed_offset = 30

# Adjust these numbers based on the results from hyperparameter tuning.
mutation_prob = 0.01
crossover_prob = 0.5

tag = "varying_cache_params"

with open("evaluation/ga_case_study/slurm/experiments.sh", "w") as target_file:

    for seed_i in range(seed_num):
        seed = seed_offset + seed_i

        # Experiment parameters
        for synthesis_target in synthesis_targets:
            for qubit_num in qubit_nums:
                for gate_count in gate_counts:

                    # GA parameters
                    for selection_strategy in selection_strategies:
                        
                        # Caching parameters
                        for rebuild_frequency in rebuild_frequencies:
                            for cache_size in cache_sizes:
                                for merging_round_num in merging_rounds:

                                    experiment_cmd = f"sbatch --job-name=ga_{synthesis_target}_{selection_strategy}"
                                    experiment_cmd += f"_{qubit_num}q_{gate_count}g"
                                    experiment_cmd += f"_{rebuild_frequency}rf_{cache_size}cs_{merging_round_num}mr"
                                    
                                    experiment_cmd += f" --export=cache_size={cache_size},merging_rounds={merging_round_num},"
                                    experiment_cmd += f"rebuild_frequency={rebuild_frequency},qubit_num={qubit_num},gate_count={gate_count},"
                                    experiment_cmd += f"selection_strategy=\"{selection_strategy}\",synthesis_target=\"{synthesis_target}\","
                                    experiment_cmd += f"crossover_prob={crossover_prob},mutation_prob={mutation_prob},seed={seed},tag=\"{tag}\""

                                    experiment_cmd += f" run_experiment.sh"

                                    target_file.write(experiment_cmd + "\n")
                    