#!/bin/bash
#SBATCH --job-name=HistGradientBoostingRegressor_5_15   # create a name for your job
#SBATCH --nodes=1                # node count
#SBATCH -o ./slurm-output/slurm-%j.out # STDOUT
#SBATCH --ntasks=1               # total number of tasks
#SBATCH --cpus-per-task=70       # cpu-cores per task
#SBATCH --account=syseng
#SBATCH --partition=syseng

/work/syseng/users/fjacevedo/Markovian/env/bin/python /work/syseng/users/fjacevedo/Markovian/scripts/experiments.py HistGradientBoostingRegressor 5 0.15
