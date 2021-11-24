import os
import itertools as itr
from pathlib import Path
PATH = Path(__file__).parent.resolve()

regressors = ['ElasticNet', 'Lars', 'Lasso', 'LassoLars',
              'GaussianProcessRegressor', 'KNeighborsRegressor',
              'RandomForestRegressor']
obs = [0.05, 0.10, 0.15]
rs = [3, 5, 7]
for rg, r, p in itr.product(regressors, rs, obs):
    body = ('#!/bin/bash\n',
            f'#SBATCH --job-name={rg}_{r}_{int(p*100)}   # create a name for your job\n',
            '#SBATCH --nodes=1                # node count\n',
            '#SBATCH -o ./slurm-output/slurm-%j.out # STDOUT\n',
            '#SBATCH --ntasks=1               # total number of tasks\n',
            '#SBATCH --cpus-per-task=70       # cpu-cores per task\n',
            '#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)\n',
            '#SBATCH --account=syseng\n',
            '#SBATCH --partition=syseng\n\n',
            'module purge\n',
            'module load miniconda gnu8/8.3.0\n\n',
            'eval "$(conda shell.bash hook)"\n',
            'conda activate markovian\n',
            f'python ./scripts/experiments.py {rg} {r} {p}')

    with open(PATH.parent/f'exps/{rg}_{r}_{int(p*100)}.sh', 'w', newline='\n') as file:
        for line in body:
            file.write(line)

for file in os.listdir(PATH.parent/'exps'):
    if file.split('.')[-1] == 'sh':
        file_path = PATH.parent/f'exps/{file}'
        os.system(f'chmod u+x {file_path}')
        print(f'{file}: Done!')
