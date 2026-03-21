"""Train pattern values via logistic regression on self-play positions.

Extracts 8-cell window features from positions, maps to canonical patterns,
and optimizes values so sigmoid(sum_of_values) predicts win probability.

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
    """Convert positions to sparse feature matrix, target vector, and weights.

    Each position has (board, current_player, wins, losses, draws[, game_id]).
    Target = empirical win rate = (wins + 0.5*draws) / total.
    Weight = total observations for that position.
    """
    print(f"Extracting features from {len(positions)} positions...")
    t0 = time.time()

    feat_indices = []
    feat_values = []
    targets = []
    weights = []

    for i, entry in enumerate(positions):
        if len(entry) == 5:
            board, cp, wins, losses, draws = entry
        else:
            board, cp, wins, losses, draws = entry[:5]

        total = wins + losses + draws
        if total == 0:
            continue
        target = (wins + 0.5 * draws) / total

        features = extract_features(board, cp)
        if features:
            idx = np.array(list(features.keys()), dtype=np.int32)
            vals = np.array(list(features.values()), dtype=np.float64)
            feat_indices.append(idx)
            feat_values.append(vals)
            targets.append(target)
            weights.append(total)

        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{len(positions)} ({time.time()-t0:.1f}s)")

    weights = np.array(weights, dtype=np.float64)
    weights /= weights.mean()  # normalize so mean weight = 1

    total_obs = sum(e[2] + e[3] + e[4] for e in positions)
    print(f"  Done in {time.time()-t0:.1f}s, {len(targets)} usable positions from {total_obs} observations")
    return feat_indices, feat_values, np.array(targets, dtype=np.float64), weights


def _evaluate(params, feat_indices, feat_values, targets, weights):
    """Compute loss and accuracy for a dataset."""
    n = len(targets)
    if n == 0:
        return 0.0, 0.0
    evals = np.zeros(n, dtype=np.float64)
    for i in range(n):
        evals[i] = np.dot(params[feat_indices[i]], feat_values[i])
    evals_clipped = np.clip(evals, -30, 30)
    preds = 1.0 / (1.0 + np.exp(-evals_clipped))
    preds_safe = np.clip(preds, 1e-15, 1 - 1e-15)
    per_sample = -(targets * np.log(preds_safe) + (1 - targets) * np.log(1 - preds_safe))
    loss = float(np.mean(weights * per_sample))
    acc = float(np.sum((preds > 0.5) == (targets > 0.5))) / n
    return loss, acc


def train(train_data, val_data, num_params, epochs=200, lr=0.1, l2=0.01):
    """Train pattern values using Adam optimizer on weighted BCE + L2 regularization.

    L2 penalty shrinks rare/noisy pattern values toward zero, preventing
    overfitting on patterns seen in only a few positions.
    """
    tr_idx, tr_vals, tr_targets, tr_weights = train_data
    params = np.zeros(num_params, dtype=np.float64)
    n = len(tr_targets)

    # Adam state
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    val_idx, val_vals, val_targets, val_weights = val_data
    has_val = len(val_targets) > 0

    print(f"\nTraining {num_params} parameters on {n} train / {len(val_targets)} val "
          f"for {epochs} epochs (l2={l2})...")
    t0 = time.time()

    for epoch in range(epochs):
        # Forward pass on train
        evals = np.zeros(n, dtype=np.float64)
        for i in range(n):
            evals[i] = np.dot(params[tr_idx[i]], tr_vals[i])

        evals_clipped = np.clip(evals, -30, 30)
        preds = 1.0 / (1.0 + np.exp(-evals_clipped))

        preds_safe = np.clip(preds, 1e-15, 1 - 1e-15)
        per_sample = -(tr_targets * np.log(preds_safe) + (1 - tr_targets) * np.log(1 - preds_safe))
        bce_loss = np.mean(tr_weights * per_sample)
        reg_loss = l2 * np.sum(params ** 2)
        loss = bce_loss + reg_loss

        correct = np.sum((preds > 0.5) == (tr_targets > 0.5))
        acc = correct / n

        if epoch % 20 == 0 or epoch == epochs - 1:
            elapsed = time.time() - t0
            if has_val:
                v_loss, v_acc = _evaluate(params, val_idx, val_vals, val_targets, val_weights)
                print(f"  epoch {epoch:4d}: train loss={bce_loss:.4f}+reg={reg_loss:.4f} acc={acc:.3f}"
                      f"  |  val loss={v_loss:.4f} acc={v_acc:.3f}  ({elapsed:.1f}s)")
            else:
                print(f"  epoch {epoch:4d}: loss={bce_loss:.4f}+reg={reg_loss:.4f}  acc={acc:.3f}  ({elapsed:.1f}s)")

        # Backward pass: BCE gradient + L2 gradient
        residuals = tr_weights * (preds - tr_targets) / n
        grad = np.zeros_like(params)
        for i in range(n):
            grad[tr_idx[i]] += residuals[i] * tr_vals[i]
        grad += 2 * l2 * params  # L2 gradient

        # Adam update
        t_adam = epoch + 1
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t_adam)
        v_hat = v / (1 - beta2 ** t_adam)
        params -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # Final metrics
    if has_val:
        v_loss, v_acc = _evaluate(params, val_idx, val_vals, val_targets, val_weights)
        print(f"\nFinal: train loss={bce_loss:.4f} acc={acc:.3f}  |  val loss={v_loss:.4f} acc={v_acc:.3f}")
    else:
        print(f"\nFinal: loss={bce_loss:.4f}  acc={acc:.3f}")
    return params


def save_results(params, output_dir):
    """Save the trained lookup table."""
    os.makedirs(output_dir, exist_ok=True)

    # Save as JSON: {pattern_string: value}
    result = {}
    for i, pat in enumerate(CANON_PATTERNS):
        pat_str = "".join(str(c) for c in pat)
        result[pat_str] = float(params[i])

    json_path = os.path.join(output_dir, "pattern_values.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    # Save as numpy array
    npy_path = os.path.join(output_dir, "pattern_values.npy")
    np.save(npy_path, params)

    # Print top patterns
    sorted_idx = np.argsort(params)
    print(f"\nTop 20 most valuable patterns (for current player):")
    for i in sorted_idx[-20:][::-1]:
        pat_str = "".join(str(c) for c in CANON_PATTERNS[i])
        # Readable: . = empty, X = current player, O = opponent
        readable = pat_str.replace("0", ".").replace("1", "X").replace("2", "O")
        print(f"  {readable:>8s}  {params[i]:+.4f}")

    print(f"\nTop 20 worst patterns (for current player):")
    for i in sorted_idx[:20]:
        pat_str = "".join(str(c) for c in CANON_PATTERNS[i])
        readable = pat_str.replace("0", ".").replace("1", "X").replace("2", "O")
        print(f"  {readable:>8s}  {params[i]:+.4f}")

    print(f"\nSaved to {json_path} and {npy_path}")
    return json_path


def split_by_game(positions, val_fraction=0.2, seed=42):
    """Split positions into train/val by originating game_id.

    Positions from the same game end up in the same split, avoiding
    leakage from correlated positions within a game.
    Falls back to a random position split if no game IDs are present.
    """
    has_game_ids = any(len(entry) >= 6 for entry in positions)

    rng = np.random.RandomState(seed)

    if has_game_ids:
        game_ids = sorted({entry[5] for entry in positions if len(entry) >= 6})
        rng.shuffle(game_ids)
        n_val = max(1, int(len(game_ids) * val_fraction))
        val_games = set(game_ids[:n_val])

        train_pos = []
        val_pos = []
        for entry in positions:
            gid = entry[5] if len(entry) >= 6 else 0
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

        print(f"Split by position (no game IDs): {len(train_pos)} train / {len(val_pos)} val")

    print(f"  {len(train_pos)} train positions, {len(val_pos)} val positions")
    return train_pos, val_pos


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(os.path.dirname(__file__), "data", "positions.pkl"))
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "results"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        positions = pickle.load(f)
    print(f"Loaded {len(positions)} positions from {args.input}")

    train_pos, val_pos = split_by_game(positions, val_fraction=args.val_fraction)

    train_data = build_dataset(train_pos)
    val_data = build_dataset(val_pos)

    params = train(train_data, val_data, NUM_CANON,
                   epochs=args.epochs, lr=args.lr, l2=args.l2)
    save_results(params, args.output_dir)


if __name__ == "__main__":
    main()
