import itertools as itr
from pathlib import Path
PATH = Path(__file__).parent.resolve()

regressors = ['ElasticNet', 'Lars', 'Lasso', 'LassoLars', 'ARDRegression',
              'BayesianRidge', 'GaussianProcessRegressor', 'AdaBoostRegressor',
              'KNeighborsRegressor', 'RandomForestRegressor',
              'HistGradientBoostingRegressor', 'GradientBoostingRegressor']
obs = [0.05, 0.10, 0.15]
rs = [3, 5, 7]
for rg, r, p in itr.product(regressors, rs, obs):
    with open(PATH/f"../exps/{rg}_{r}_{int(p*100)}.sh", "w", newline='\n') as file:
        file.write(f"""#!/bin/bash
#SBATCH --job-name={rg}_{r}_{int(p*100)}   # create a name for your job
#SBATCH --nodes=1                # node count
#SBATCH -o ./slurm-output/slurm-%j.out # STDOUT
#SBATCH --ntasks=1               # total number of tasks
#SBATCH --cpus-per-task=70       # cpu-cores per task
#SBATCH --account=syseng
#SBATCH --partition=syseng

/work/syseng/users/fjacevedo/Markovian/env/bin/python /work/syseng/users/fjacevedo/Markovian/scripts/experiments.py {rg} {r} {p}
""")
