#!/bin/bash
#SBATCH --job-name=LassoLars_3_5   # create a name for your job
#SBATCH --nodes=1                # node count
#SBATCH -o ./slurm-output/slurm-%j.out # STDOUT
#SBATCH --ntasks=1               # total number of tasks
#SBATCH --cpus-per-task=70       # cpu-cores per task
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)
#SBATCH --account=syseng
#SBATCH --partition=syseng

module purge
module load miniconda gnu8/8.3.0

eval "$(conda shell.bash hook)"
conda activate markovian
python ./scripts/experiments.py LassoLars 3 0.05