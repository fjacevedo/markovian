import numpy as np
from pathlib import Path
PATH = Path(__file__).parent.resolve()

with open(PATH.parent/"status.csv", "r") as file:
    for line in file:
        print(line.split(","))
