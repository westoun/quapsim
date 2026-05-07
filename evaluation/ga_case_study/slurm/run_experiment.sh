#!/bin/bash
#SBATCH --job-name=quapsim
#SBATCH --output=logs/slurm.%j.%x.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=4-00:00:00

echo "Activate Python and virtualenv"
cd /home/ws/ws16/quapsim/evaluation/ga_case_study

module load python/3.11.1

echo "Start GA Experiment Run"

# Activate virtual environment
source /home/ws/ws16/quapsim/venv/bin/activate

# Run the Python script
echo python run_ga_experiment.py -cs ${cache_size} -mr ${merging_rounds} -rf ${rebuild_frequency} -qn ${qubit_num} -gc ${gate_count} -ss ${selection_strategy} -st ${synthesis_target} -cp ${crossover_prob} -mp ${mutation_prob} -s ${seed} -t ${tag}
python run_ga_experiment.py -cs ${cache_size} -mr ${merging_rounds} -rf ${rebuild_frequency} -qn ${qubit_num} -gc ${gate_count} -ss ${selection_strategy} -st ${synthesis_target} -cp ${crossover_prob} -mp ${mutation_prob} -s ${seed} -t ${tag}
