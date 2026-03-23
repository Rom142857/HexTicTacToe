"""Run a sweep of training configurations and compare results.

Tests different subsample rates, data sources, and symmetry augmentation.
Evaluates each by val accuracy and plays the best against baseline.

Usage: python -m learned_eval.train_sweep
"""

import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import Player, HEX_DIRECTIONS
from learned_eval.pattern_table import build_arrays
from learned_eval.train import (
    extract_features, build_dataset, train, save_results,
    split_by_game, _board_has_win, SCORE_SCALE
)
import learned_eval.train as train_module

_WIN_LENGTH = 6


def augment_symmetry(positions):
    """Double dataset by adding each position from the opponent's perspective.

    For a 5-tuple (board, cp, eval, win, gid):
      - Flip current_player (A<->B)
      - Negate win_score
      - eval is set to 0 (we don't have the opponent's eval)
    """
    augmented = list(positions)
    for p in positions:
        board, cp, eval_score, win_score, gid = p
        opp = Player.B if cp == Player.A else Player.A
        augmented.append((board, opp, -eval_score, -win_score, gid))
    return augmented


def subsample_positions(positions, rate):
    """Keep every Nth position per game, sorted by board size."""
    if rate <= 1:
        return positions
    from collections import defaultdict
    is_5tuple = len(positions[0]) == 5
    gid_idx = 4 if is_5tuple else 3
    by_game = defaultdict(list)
    for p in positions:
        by_game[p[gid_idx]].append(p)
    result = []
    for gid, game_pos in by_game.items():
        game_pos.sort(key=lambda p: len(p[0]))
        result.extend(game_pos[::rate])
    return result


def run_experiment(name, positions, output_dir, epochs=200, lr=0.01, l2=0.005,
                   init_weights=None):
    """Run a single training experiment, return val metrics."""
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {name}")
    print(f"  {len(positions)} positions")
    print(f"{'='*60}")

    is_5tuple = len(positions[0]) == 5
    target_idx = 3 if is_5tuple else 2
    win_idx = 3 if is_5tuple else 2

    # Filter already-won
    before = len(positions)
    positions = [p for p in positions if not _board_has_win(p[0])]
    if len(positions) < before:
        print(f"Removed {before - len(positions)} already-won positions")

    train_pos, val_pos = split_by_game(positions, val_fraction=0.2)

    train_data = build_dataset(train_pos, target_idx=target_idx, win_idx=win_idx,
                                binary_targets=True)
    val_data = build_dataset(val_pos, target_idx=target_idx, win_idx=win_idx,
                              binary_targets=True)

    num_canon = train_module.NUM_CANON
    params = train(train_data, val_data, num_canon,
                   epochs=epochs, lr=lr, l2=l2, init_weights=init_weights)

    # Final val metrics
    from learned_eval.train import _evaluate
    val_idx, val_vals, val_targets, val_weights, val_wins = val_data
    val_mse, val_r2, val_acc = _evaluate(params, val_idx, val_vals, val_targets,
                                          val_weights, val_wins)

    os.makedirs(output_dir, exist_ok=True)
    save_results(params, output_dir)

    return {
        "name": name,
        "n_positions": len(positions),
        "n_train": len(train_pos),
        "n_val": len(val_pos),
        "val_mse": val_mse,
        "val_r2": val_r2,
        "val_acc": val_acc,
        "output_dir": output_dir,
    }


def main():
    # Initialize pattern tables
    window_length = 6
    train_module.WINDOW_LENGTH = window_length
    arrays = build_arrays(window_length)
    train_module.CANON_PATTERNS = arrays[0]
    train_module.CANON_INDEX = arrays[1]
    train_module.CANON_SIGN = arrays[2]
    train_module.NUM_CANON = arrays[3]
    print(f"Window length {window_length}: {arrays[3]} canonical patterns")

    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")

    # Load init weights (baseline ai_tuned)
    init_weights_path = os.path.join(base_dir, "results", "pattern_values.npy")
    init_weights = np.load(init_weights_path) if os.path.exists(init_weights_path) else None

    # Load playout data
    playout_path = os.path.join(data_dir, "positions_human_playouts.pkl")
    if not os.path.exists(playout_path):
        print(f"ERROR: {playout_path} not found. Run playout_positions.py first.")
        return

    with open(playout_path, "rb") as f:
        playout_positions = pickle.load(f)
    print(f"Loaded {len(playout_positions)} playout positions")

    # Load human data (4-tuple: board, cp, win_label, gid)
    human_path = os.path.join(data_dir, "positions_human.pkl")
    with open(human_path, "rb") as f:
        human_positions = pickle.load(f)
    print(f"Loaded {len(human_positions)} human positions")

    # Normalize human win labels to ±1 and convert to 5-tuple
    human_5tuple = []
    for p in human_positions:
        board, cp, raw_win, gid = p
        win = 1.0 if raw_win > 0 else -1.0
        human_5tuple.append((board, cp, 0.0, win, gid))
    human_positions = human_5tuple

    results = []

    # ── Playout experiments: varying subsample ──
    for ss in [1, 2, 4, 8]:
        sampled = subsample_positions(playout_positions, ss)
        name = f"playout_ss{ss}"
        out = os.path.join(base_dir, "sweep", name)
        r = run_experiment(name, sampled, out, init_weights=init_weights)
        results.append(r)

    # ── Playout + symmetry augmentation ──
    for ss in [2, 4]:
        sampled = subsample_positions(playout_positions, ss)
        sampled = augment_symmetry(sampled)
        name = f"playout_ss{ss}_sym"
        out = os.path.join(base_dir, "sweep", name)
        r = run_experiment(name, sampled, out, init_weights=init_weights)
        results.append(r)

    # ── Human game outcomes: varying subsample ──
    for ss in [1, 2, 4, 8]:
        sampled = subsample_positions(human_positions, ss)
        name = f"human_ss{ss}"
        out = os.path.join(base_dir, "sweep", name)
        r = run_experiment(name, sampled, out, init_weights=init_weights)
        results.append(r)

    # ── Human + symmetry augmentation ──
    for ss in [2, 4]:
        sampled = subsample_positions(human_positions, ss)
        sampled = augment_symmetry(sampled)
        name = f"human_ss{ss}_sym"
        out = os.path.join(base_dir, "sweep", name)
        r = run_experiment(name, sampled, out, init_weights=init_weights)
        results.append(r)

    # ── Summary ──
    print(f"\n\n{'='*70}")
    print(f"  SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Name':<25s} {'N_pos':>7s} {'Val MSE':>9s} {'Val R2':>8s} {'Val Acc':>8s}")
    print(f"  {'-'*25} {'-'*7} {'-'*9} {'-'*8} {'-'*8}")

    results.sort(key=lambda r: -r["val_acc"])
    for r in results:
        print(f"  {r['name']:<25s} {r['n_positions']:>7d} {r['val_mse']:>9.4f} "
              f"{r['val_r2']:>8.4f} {r['val_acc']:>7.1%}")

    best = results[0]
    print(f"\n  Best: {best['name']} (val_acc={best['val_acc']:.1%})")
    print(f"  Weights at: {best['output_dir']}")

    # Save summary
    summary_path = os.path.join(base_dir, "sweep", "summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
