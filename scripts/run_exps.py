import os
from pathlib import Path
PATH = Path(__file__).parent.resolve()

with open(PATH.parent/'master_run.sh', 'w', newline='\n') as bash:
    bash.write('#!/bin/bash\n')
    for file in os.listdir(PATH.parent/'exps'):
        if file.split('.')[1] == 'txt':
            continue
        bash.writelines(f'sbatch ./exps/{file}\n')

file_path = PATH.parent/'master_run.sh'
os.system(f'chmod u+x {file_path}')
