"""Verify playout output format is compatible with train.py."""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from game import Player

path = os.path.join(os.path.dirname(__file__), "data", "positions_human_playouts.pkl")
with open(path, "rb") as f:
    data = pickle.load(f)

print(f"Loaded {len(data)} positions")
print(f"Tuple length: {len(data[0])}")
p = data[0]
print(f"Sample: board={len(p[0])} stones, player={p[1]}, eval={p[2]}, win={p[3]}, gid={p[4][:8]}...")
wins = sum(1 for p in data if p[3] > 0)
losses = sum(1 for p in data if p[3] < 0)
draws = sum(1 for p in data if p[3] == 0)
print(f"W/L/D: {wins}/{losses}/{draws}")
sizes = [len(p[0]) for p in data]
print(f"Board sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.1f}")
