import os
import re
import subprocess
from pathlib import Path
PATH = Path(__file__).parent.resolve()
with open(PATH.parent/'status.csv', 'w') as f:
    for file in os.listdir(PATH.parent/'slurm-output'):
        out = subprocess.run([
            "sacct", f"--job={''.join(re.findall('[0-9]', file))}",
            "--format=JobID, Jobname%50,state,time"],
            stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')
        to_write = out[2].split()
        f.writelines(','.join(to_write)+'\n')
    out = subprocess.run([
        "sacct", "--state=PD",
        "--format=JobID, Jobname%50,state,time"],
        stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')
    for line in out[2:]:
        to_write = line.split()
        f.writelines(','.join(to_write)+'\n')
