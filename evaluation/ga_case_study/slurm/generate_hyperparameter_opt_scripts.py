

qubit_nums = [4, 6, 8]
gate_counts = [10, 15, 20, 25, 30]
selection_strategies = ["roulette"]
synthesis_targets = ["random"]
seed_count = 5

with open("evaluation/ga_case_study/slurm/hyperparameter_search.sh", "w") as target_file:

    for synthesis_target in synthesis_targets:
        for selection_strategy in selection_strategies:
            for qubit_num in qubit_nums:
                for gate_count in gate_counts:

                    experiment_cmd = f"sbatch --job-name=hpo_{synthesis_target}_{selection_strategy}"
                    experiment_cmd += f"_{qubit_num}q_{gate_count}g"
                    experiment_cmd += f" --export=qubit_num={qubit_num},gate_count={gate_count}"
                    experiment_cmd += f",selection_strategy=\"{selection_strategy}\",synthesis_target=\"{synthesis_target}\""
                    experiment_cmd += f",seed_count={seed_count}"
                    experiment_cmd += f" run_hyperparameter_opt.sh"

                    target_file.write(experiment_cmd + "\n")
