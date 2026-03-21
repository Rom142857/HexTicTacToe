"""Train pattern values via linear regression on hand-tuned eval targets.

Extracts 6-cell window features from positions, maps to canonical patterns,
and optimizes values so sum_of_values approximates the hand-tuned evaluator.

Usage: python -m learned_eval.train [--input data/positions.pkl] [--epochs 200] [--lr 0.1]
"""

import json
import math
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import Player, HEX_DIRECTIONS
from learned_eval.pattern_table import (
    WINDOW_LENGTH, CANON_INDEX, CANON_SIGN, NUM_CANON,
    CANON_PATTERNS, pattern_to_int,
)

DIR_VECTORS = list(HEX_DIRECTIONS)


def extract_features(board, current_player):
    """Extract sparse feature vector {canon_index: sum_of_signs} for a position.

    Windows are 8 cells long. Patterns are read from current_player's perspective
    (current_player = 1, opponent = 2).
    """
    opponent = Player.B if current_player == Player.A else Player.A
    features = {}
    seen = set()

    for (q, r) in board:
        for d_idx, (dq, dr) in enumerate(DIR_VECTORS):
            for k in range(WINDOW_LENGTH):
                # Window starting at (q - k*dq, r - k*dr) along direction d_idx
                sq = q - k * dq
                sr = r - k * dr
                wkey = (d_idx, sq, sr)
                if wkey in seen:
                    continue
                seen.add(wkey)

                # Read the 8-cell pattern from current player's perspective
                pat_int = 0
                has_piece = False
                power = 1
                for j in range(WINDOW_LENGTH):
                    cell = board.get((sq + j * dq, sr + j * dr))
                    if cell is None:
                        v = 0
                    elif cell == current_player:
                        v = 1
                        has_piece = True
                    else:
                        v = 2
                        has_piece = True
                    pat_int += v * power
                    power *= 3

                if not has_piece:
                    continue

                ci = CANON_INDEX[pat_int]
                cs = CANON_SIGN[pat_int]
                if cs == 0:
                    continue  # self-symmetric or empty, forced zero

                if ci in features:
                    features[ci] += cs
                else:
                    features[ci] = cs

    return features


def build_dataset(positions):
    """Convert positions to sparse feature matrix and target vector.

    Each position has (board, current_player, eval_score, game_id).
    Target = eval_score, normalized to zero mean / unit variance.
    """
    print(f"Extracting features from {len(positions)} positions...")
    t0 = time.time()

    feat_indices = []
    feat_values = []
    raw_targets = []

    for i, entry in enumerate(positions):
        board, cp, eval_score = entry[0], entry[1], entry[2]

        features = extract_features(board, cp)
        if features:
            idx = np.array(list(features.keys()), dtype=np.int32)
            vals = np.array(list(features.values()), dtype=np.float64)
            feat_indices.append(idx)
            feat_values.append(vals)
            raw_targets.append(eval_score)

        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{len(positions)} ({time.time()-t0:.1f}s)")

    raw_targets = np.array(raw_targets, dtype=np.float64)

    # Normalize targets for stable training
    target_mean = float(np.mean(raw_targets))
    target_std = float(np.std(raw_targets))
    if target_std < 1e-8:
        target_std = 1.0
    targets = (raw_targets - target_mean) / target_std

    weights = np.ones(len(targets), dtype=np.float64)

    print(f"  Done in {time.time()-t0:.1f}s, {len(targets)} usable positions")
    print(f"  Eval targets: mean={target_mean:.1f}, std={target_std:.1f}, "
          f"min={raw_targets.min():.1f}, max={raw_targets.max():.1f}")
    return feat_indices, feat_values, targets, weights, target_mean, target_std


def _evaluate(params, feat_indices, feat_values, targets, weights):
    """Compute MSE loss and R² for a dataset."""
    n = len(targets)
    if n == 0:
        return 0.0, 0.0
    preds = np.zeros(n, dtype=np.float64)
    for i in range(n):
        preds[i] = np.dot(params[feat_indices[i]], feat_values[i])
    residuals = preds - targets
    mse = float(np.mean(weights * residuals ** 2))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((targets - np.mean(targets)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return mse, r2


def train(train_data, val_data, num_params, epochs=200, lr=0.01, l2=0.001):
    """Train pattern values using Adam optimizer on MSE + L2 regularization."""
    tr_idx, tr_vals, tr_targets, tr_weights = train_data[:4]
    params = np.zeros(num_params, dtype=np.float64)
    n = len(tr_targets)

    # Adam state
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    val_idx, val_vals, val_targets, val_weights = val_data[:4]
    has_val = len(val_targets) > 0

    print(f"\nTraining {num_params} parameters on {n} train / {len(val_targets)} val "
          f"for {epochs} epochs (lr={lr}, l2={l2})...")
    t0 = time.time()

    for epoch in range(epochs):
        # Forward pass: linear prediction
        preds = np.zeros(n, dtype=np.float64)
        for i in range(n):
            preds[i] = np.dot(params[tr_idx[i]], tr_vals[i])

        # MSE loss
        residuals = preds - tr_targets
        mse_loss = np.mean(tr_weights * residuals ** 2)
        reg_loss = l2 * np.sum(params ** 2)

        if epoch % 20 == 0 or epoch == epochs - 1:
            ss_res = float(np.sum(residuals ** 2))
            ss_tot = float(np.sum((tr_targets - np.mean(tr_targets)) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            elapsed = time.time() - t0
            if has_val:
                v_loss, v_r2 = _evaluate(params, val_idx, val_vals, val_targets, val_weights)
                print(f"  epoch {epoch:4d}: train mse={mse_loss:.6f} R²={r2:.4f}"
                      f"  |  val mse={v_loss:.6f} R²={v_r2:.4f}  ({elapsed:.1f}s)")
            else:
                print(f"  epoch {epoch:4d}: mse={mse_loss:.6f} R²={r2:.4f}  ({elapsed:.1f}s)")

        # Backward pass: MSE gradient + L2
        grad = np.zeros_like(params)
        scaled_res = 2.0 * tr_weights * residuals / n
        for i in range(n):
            grad[tr_idx[i]] += scaled_res[i] * tr_vals[i]
        grad += 2 * l2 * params

        # Adam update
        t_adam = epoch + 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t_adam)
        v_hat = v / (1 - beta2 ** t_adam)
        params -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # Final metrics
    if has_val:
        v_loss, v_r2 = _evaluate(params, val_idx, val_vals, val_targets, val_weights)
        print(f"\nFinal: train mse={mse_loss:.6f} R²={r2:.4f}  |  val mse={v_loss:.6f} R²={v_r2:.4f}")
    else:
        print(f"\nFinal: mse={mse_loss:.6f} R²={r2:.4f}")
    return params


def save_results(params, output_dir, target_mean=0.0, target_std=1.0):
    """Save the trained lookup table with denormalized values."""
    os.makedirs(output_dir, exist_ok=True)

    # Denormalize params so raw sum gives the hand-tuned eval scale
    # normalized_pred = dot(params, features)
    # raw_pred = normalized_pred * target_std + target_mean
    # So denormalized_params = params * target_std (bias = target_mean added separately)
    denorm_params = params * target_std

    # Save as JSON: {pattern_string: value}
    result = {"_meta": {"bias": target_mean, "std": target_std}}
    for i, pat in enumerate(CANON_PATTERNS):
        pat_str = "".join(str(c) for c in pat)
        result[pat_str] = float(denorm_params[i])

    json_path = os.path.join(output_dir, "pattern_values.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    # Save as numpy array (denormalized)
    npy_path = os.path.join(output_dir, "pattern_values.npy")
    np.save(npy_path, denorm_params)

    # Print top patterns
    sorted_idx = np.argsort(denorm_params)
    print(f"\nTop 20 most valuable patterns (for current player):")
    for i in sorted_idx[-20:][::-1]:
        pat_str = "".join(str(c) for c in CANON_PATTERNS[i])
        readable = pat_str.replace("0", ".").replace("1", "X").replace("2", "O")
        print(f"  {readable:>8s}  {denorm_params[i]:+.1f}")

    print(f"\nTop 20 worst patterns (for current player):")
    for i in sorted_idx[:20]:
        pat_str = "".join(str(c) for c in CANON_PATTERNS[i])
        readable = pat_str.replace("0", ".").replace("1", "X").replace("2", "O")
        print(f"  {readable:>8s}  {denorm_params[i]:+.1f}")

    print(f"\nBias (add to sum): {target_mean:.1f}")
    print(f"Saved to {json_path} and {npy_path}")
    return json_path


def split_by_game(positions, val_fraction=0.2, seed=42):
    """Split positions into train/val by originating game_id.

    Each position has (board, current_player, eval_score, game_id).
    Positions from the same game end up in the same split, avoiding
    leakage from correlated positions within a game.
    """
    rng = np.random.RandomState(seed)

    has_game_ids = len(positions[0]) >= 4
    if has_game_ids:
        game_ids = sorted({entry[3] for entry in positions})
        rng.shuffle(game_ids)
        n_val = max(1, int(len(game_ids) * val_fraction))
        val_games = set(game_ids[:n_val])

        train_pos = []
        val_pos = []
        for entry in positions:
            gid = entry[3]
            if gid in val_games:
                val_pos.append(entry)
            else:
                train_pos.append(entry)

        print(f"Split by game: {len(game_ids)} games -> {len(game_ids) - n_val} train / {n_val} val")
    else:
        indices = np.arange(len(positions))
        rng.shuffle(indices)
        n_val = max(1, int(len(positions) * val_fraction))
        val_set = set(indices[:n_val].tolist())

        train_pos = [positions[i] for i in range(len(positions)) if i not in val_set]
        val_pos = [positions[i] for i in val_set]

        print(f"Split by position: {len(train_pos)} train / {len(val_pos)} val")

    print(f"  {len(train_pos)} train positions, {len(val_pos)} val positions")
    return train_pos, val_pos


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(os.path.dirname(__file__), "data", "positions.pkl"))
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "results"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        positions = pickle.load(f)
    print(f"Loaded {len(positions)} positions from {args.input}")

    train_pos, val_pos = split_by_game(positions, val_fraction=args.val_fraction)

    train_result = build_dataset(train_pos)
    train_data = train_result[:4]
    target_mean, target_std = train_result[4], train_result[5]

    val_result = build_dataset(val_pos)
    val_data = val_result[:4]

    params = train(train_data, val_data, NUM_CANON,
                   epochs=args.epochs, lr=args.lr, l2=args.l2)
    save_results(params, args.output_dir, target_mean, target_std)


if __name__ == "__main__":
    main()
