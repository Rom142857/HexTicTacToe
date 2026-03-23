"""Combine multiple position pickle files into one."""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game import Player

files = sys.argv[1:-1]
output = sys.argv[-1]

all_positions = []
for f in files:
    with open(f, "rb") as fh:
        data = pickle.load(fh)
    print(f"  {f}: {len(data)} positions")
    all_positions.extend(data)

print(f"Total: {len(all_positions)} positions")
with open(output, "wb") as fh:
    pickle.dump(all_positions, fh)
print(f"Saved to {output}")
