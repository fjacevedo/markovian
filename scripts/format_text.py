from pathlib import Path

PATH = Path(__file__).parent.resolve()

FILE = PATH.parent/'prints/RidgeCV_3_15.txt'

with open(FILE, 'r') as file:
    lines = file.readlines()
    splits = list()
    for step in range(0, 850, 10):
        lines[-1] = lines[-1].replace(f'step={step}', f'step={step}\n')
with open(FILE, 'w', newline='\n') as file:
    file.writelines(lines)
