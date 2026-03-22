"""Train pattern values via linear regression on position targets.

Extracts 6-cell window features from positions, maps to canonical patterns,
and optimizes values so sum_of_values approximates the target.

Supports two target types:
  --target eval  (default) Train on evaluator scores (entry[2])
  --target win   Train on game outcomes (entry[3] for 5-tuple, entry[2] for 4-tuple)

Usage: python -m learned_eval.train [--input data/positions.pkl] [--target eval] [--epochs 200]
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
from ai import LINE_SCORES
from learned_eval.pattern_table import build_arrays

DIR_VECTORS = list(HEX_DIRECTIONS)
_WIN_LENGTH = 6

# These globals are set by main() based on --window-length
WINDOW_LENGTH = 6
CANON_PATTERNS = None
CANON_INDEX = None
CANON_SIGN = None
NUM_CANON = 0


def _board_has_win(board):
    """Check if any player has 6 in a row on this board."""
    seen = set()
    for (q, r) in board:
        for dq, dr in DIR_VECTORS:
            for k in range(_WIN_LENGTH):
                sq, sr = q - k * dq, r - k * dr
                wkey = (dq, dr, sq, sr)
                if wkey in seen:
                    continue
                seen.add(wkey)
                counts = {}
                for j in range(_WIN_LENGTH):
                    cell = board.get((sq + j * dq, sr + j * dr))
                    if cell is not None:
                        counts[cell] = counts.get(cell, 0) + 1
                for c in counts.values():
                    if c == _WIN_LENGTH:
                        return True
    return False


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


def build_dataset(positions, target_idx=2, win_idx=None, binary_targets=False):
    """Convert positions to sparse feature matrix and target vector.

    For eval targets: tanh compresses continuous scores to (-1, +1).
    For binary win/loss targets (binary_targets=True): targets used as-is (±1).
    """
    print(f"Extracting features from {len(positions)} positions (target=entry[{target_idx}])...")
    t0 = time.time()

    feat_indices = []
    feat_values = []
    raw_targets = []
    win_scores = []

    for i, entry in enumerate(positions):
        board, cp = entry[0], entry[1]
        target = entry[target_idx]

        features = extract_features(board, cp)
        if features:
            idx = np.array(list(features.keys()), dtype=np.int32)
            vals = np.array(list(features.values()), dtype=np.float64)
            feat_indices.append(idx)
            feat_values.append(vals)
            raw_targets.append(target)
            if win_idx is not None:
                win_scores.append(entry[win_idx])

        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{len(positions)} ({time.time()-t0:.1f}s)")

    raw_targets = np.array(raw_targets, dtype=np.float64)
    win_scores = np.array(win_scores, dtype=np.float64) if win_scores else None

    if binary_targets:
        targets = raw_targets  # already ±1, no transformation
    else:
        targets = np.tanh(raw_targets)
    weights = np.ones(len(targets), dtype=np.float64)

    normal = raw_targets[np.abs(raw_targets) < 4999]
    n_forced = len(raw_targets) - len(normal)
    print(f"  Done in {time.time()-t0:.1f}s, {len(targets)} usable positions ({n_forced} forced wins)")
    print(f"  Non-forced scores: mean={normal.mean():.3f}, std={normal.std():.3f}, "
          f"range=[{normal.min():.2f}, {normal.max():.2f}]")
    if win_scores is not None:
        n_decisive = np.sum(win_scores != 0)
        print(f"  Win scores: {int(np.sum(win_scores > 0))}W / {int(np.sum(win_scores < 0))}L / "
              f"{int(np.sum(win_scores == 0))}D ({n_decisive} decisive)")
    return feat_indices, feat_values, targets, weights, win_scores


def _evaluate(params, feat_indices, feat_values, targets, weights,
              win_scores=None):
    """Compute MSE loss in tanh space, R², and game-outcome accuracy."""
    n = len(targets)
    if n == 0:
        return 0.0, 0.0, 0.0
    raw_preds = np.zeros(n, dtype=np.float64)
    for i in range(n):
        raw_preds[i] = np.dot(params[feat_indices[i]], feat_values[i])
    sig_preds = np.tanh(raw_preds)
    residuals = sig_preds - targets
    mse = float(np.mean(weights * residuals ** 2))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((targets - np.mean(targets)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    acc = 0.0
    if win_scores is not None:
        decisive = win_scores != 0
        n_decisive = np.sum(decisive)
        if n_decisive > 0:
            correct = np.sum((raw_preds[decisive] > 0) == (win_scores[decisive] > 0))
            acc = float(correct / n_decisive)

    return mse, r2, acc


SCORE_SCALE = 20_000  # must match generate_positions.SCORE_SCALE


def _init_from_line_scores(num_params):
    """Initialize params to match the hand-tuned LINE_SCORES evaluator.

    Values are in normalized score units (LINE_SCORES / SCORE_SCALE).
    """
    params = np.zeros(num_params, dtype=np.float64)
    for i, pat in enumerate(CANON_PATTERNS):
        my_count = sum(1 for c in pat if c == 1)
        opp_count = sum(1 for c in pat if c == 2)
        if my_count > 0 and opp_count == 0 and my_count < _WIN_LENGTH:
            params[i] = LINE_SCORES[my_count] / SCORE_SCALE
        elif opp_count > 0 and my_count == 0 and opp_count < _WIN_LENGTH:
            params[i] = -LINE_SCORES[opp_count] / SCORE_SCALE
    return params


def train(train_data, val_data, num_params, epochs=200, lr=0.01, l2=0.001):
    """Train pattern values using Adam on tanh-space MSE + L2.

    pred = tanh(dot(params, features)),  target = tanh(score).
    Scores are pre-normalized by SCORE_SCALE so everything is ~O(1).
    """
    tr_idx, tr_vals, tr_targets, tr_weights, tr_wins = train_data
    params = _init_from_line_scores(num_params)
    n = len(tr_targets)

    # Adam state
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    val_idx, val_vals, val_targets, val_weights, val_wins = val_data
    has_val = len(val_targets) > 0

    print(f"\nTraining {num_params} parameters on {n} train / {len(val_targets)} val "
          f"for {epochs} epochs (lr={lr}, l2={l2})...")
    t0 = time.time()

    for epoch in range(epochs):
        # Forward pass
        raw_preds = np.zeros(n, dtype=np.float64)
        for i in range(n):
            raw_preds[i] = np.dot(params[tr_idx[i]], tr_vals[i])
        sig_preds = np.tanh(raw_preds)

        # MSE loss in tanh space
        residuals = sig_preds - tr_targets
        mse_loss = np.mean(tr_weights * residuals ** 2)

        if epoch % 20 == 0 or epoch == epochs - 1:
            ss_res = float(np.sum(residuals ** 2))
            ss_tot = float(np.sum((tr_targets - np.mean(tr_targets)) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            _, _, tr_acc = _evaluate(params, tr_idx, tr_vals, tr_targets, tr_weights, tr_wins)
            elapsed = time.time() - t0
            if has_val:
                v_loss, v_r2, v_acc = _evaluate(params, val_idx, val_vals, val_targets,
                                                val_weights, val_wins)
                print(f"  epoch {epoch:4d}: train mse={mse_loss:.6f} R²={r2:.4f} acc={tr_acc:.3f}"
                      f"  |  val mse={v_loss:.6f} R²={v_r2:.4f} acc={v_acc:.3f}  ({elapsed:.1f}s)")
            else:
                print(f"  epoch {epoch:4d}: mse={mse_loss:.6f} R²={r2:.4f} acc={tr_acc:.3f}  ({elapsed:.1f}s)")

        # Backward pass: d/d_params MSE(tanh(pred), target)
        sig_deriv = 1.0 - sig_preds ** 2
        grad = np.zeros_like(params)
        scaled_res = 2.0 * tr_weights * residuals * sig_deriv / n
        for i in range(n):
            grad[tr_idx[i]] += scaled_res[i] * tr_vals[i]
        grad += 2 * l2 * params

        # Adam update with linear warmup + cosine decay
        t_adam = epoch + 1
        warmup_epochs = 20
        if epoch < warmup_epochs:
            cur_lr = lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
            cur_lr = lr * 0.5 * (1 + math.cos(math.pi * progress))

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t_adam)
        v_hat = v / (1 - beta2 ** t_adam)
        params -= cur_lr * m_hat / (np.sqrt(v_hat) + eps)

    # Final metrics
    _, _, tr_acc = _evaluate(params, tr_idx, tr_vals, tr_targets, tr_weights, tr_wins)
    if has_val:
        v_loss, v_r2, v_acc = _evaluate(params, val_idx, val_vals, val_targets,
                                        val_weights, val_wins)
        print(f"\nFinal: train mse={mse_loss:.6f} R²={r2:.4f} acc={tr_acc:.3f}"
              f"  |  val mse={v_loss:.6f} R²={v_r2:.4f} acc={v_acc:.3f}")
    else:
        print(f"\nFinal: mse={mse_loss:.6f} R²={r2:.4f} acc={tr_acc:.3f}")
    return params


def save_results(params, output_dir):
    """Save the trained lookup table."""
    os.makedirs(output_dir, exist_ok=True)

    denorm_params = params

    # Save as JSON: {pattern_string: value}
    result = {"_meta": {"score_scale": SCORE_SCALE, "window_length": WINDOW_LENGTH}}
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
        print(f"  {readable:>8s}  {denorm_params[i]:+.3f}")

    print(f"\nTop 20 worst patterns (for current player):")
    for i in sorted_idx[:20]:
        pat_str = "".join(str(c) for c in CANON_PATTERNS[i])
        readable = pat_str.replace("0", ".").replace("1", "X").replace("2", "O")
        print(f"  {readable:>8s}  {denorm_params[i]:+.3f}")

    print(f"\nSaved to {json_path} and {npy_path}")
    return json_path


def split_by_game(positions, val_fraction=0.2, seed=42):
    """Split positions into train/val by originating game_id.

    Each position has (board, current_player, eval_score, game_id).
    Positions from the same game end up in the same split, avoiding
    leakage from correlated positions within a game.
    """
    rng = np.random.RandomState(seed)

    # game_id is at index 3 for 4-tuple, index 4 for 5-tuple
    gid_idx = 4 if len(positions[0]) == 5 else 3
    has_game_ids = len(positions[0]) >= 4
    if has_game_ids:
        game_ids = sorted({entry[gid_idx] for entry in positions})
        rng.shuffle(game_ids)
        n_val = max(1, int(len(game_ids) * val_fraction))
        val_games = set(game_ids[:n_val])

        train_pos = []
        val_pos = []
        for entry in positions:
            gid = entry[gid_idx]
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
    parser.add_argument("--target", choices=["eval", "win"], default="eval",
                        help="'eval' = evaluator score (entry[2]), 'win' = game outcome (entry[3] for 5-tuple)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--no-draws", action="store_true",
                        help="Filter out draw positions (win_score == 0) before training")
    parser.add_argument("--subsample", type=int, default=0,
                        help="Keep every Nth position per game (by board size) to reduce correlation")
    parser.add_argument("--window-length", type=int, default=6, choices=[6, 7, 8],
                        help="Pattern window length (default 6)")
    args = parser.parse_args()

    # Initialize pattern tables for the requested window length
    global WINDOW_LENGTH, CANON_PATTERNS, CANON_INDEX, CANON_SIGN, NUM_CANON
    WINDOW_LENGTH = args.window_length
    CANON_PATTERNS, CANON_INDEX, CANON_SIGN, NUM_CANON, _ = build_arrays(WINDOW_LENGTH)
    print(f"Window length {WINDOW_LENGTH}: {NUM_CANON} canonical patterns")

    with open(args.input, "rb") as f:
        positions = pickle.load(f)
    print(f"Loaded {len(positions)} positions from {args.input}")

    # Determine target index
    is_5tuple = len(positions[0]) == 5
    if args.target == "win":
        target_idx = 3 if is_5tuple else 2  # 4-tuple eval IS the game outcome
        if not is_5tuple:
            # 4-tuple win labels may be ±1000 instead of ±1; normalize them
            max_abs = max(abs(p[target_idx]) for p in positions)
            if max_abs > 1.5:
                print(f"Normalizing 4-tuple win labels from ±{max_abs:.0f} to ±1")
                positions = [
                    (p[0], p[1], p[2] / max_abs, p[3]) for p in positions
                ]
        print(f"Training on game outcomes (entry[{target_idx}])")
    else:
        target_idx = 2
        print(f"Training on search scores (entry[2])")

    # Filter out already-won positions (6-in-a-row on the board).
    before = len(positions)
    positions = [p for p in positions if not _board_has_win(p[0])]
    if len(positions) < before:
        print(f"Removed {before - len(positions)} already-won positions, {len(positions)} remaining")

    # Filter out draws if requested
    if args.no_draws and is_5tuple:
        before = len(positions)
        positions = [p for p in positions if p[3] != 0.0]
        print(f"Removed {before - len(positions)} draw positions, {len(positions)} remaining")

    # Subsample within each game to reduce correlation
    if args.subsample > 0:
        gid_idx = 4 if is_5tuple else 3
        from collections import defaultdict
        by_game = defaultdict(list)
        for p in positions:
            by_game[p[gid_idx]].append(p)
        before = len(positions)
        positions = []
        for gid, game_pos in by_game.items():
            game_pos.sort(key=lambda p: len(p[0]))  # sort by board size (proxy for move order)
            positions.extend(game_pos[::args.subsample])
        print(f"Subsampled every {args.subsample}: {before} -> {len(positions)} positions")

    # win_score index for accuracy reporting
    win_idx = 3 if is_5tuple else (2 if args.target == "win" else None)

    train_pos, val_pos = split_by_game(positions, val_fraction=args.val_fraction)

    # Binary targets when training on win labels (±1 used directly, no tanh)
    binary = args.target == "win"
    train_data = build_dataset(train_pos, target_idx=target_idx, win_idx=win_idx, binary_targets=binary)
    val_data = build_dataset(val_pos, target_idx=target_idx, win_idx=win_idx, binary_targets=binary)

    params = train(train_data, val_data, NUM_CANON,
                   epochs=args.epochs, lr=args.lr, l2=args.l2)
    save_results(params, args.output_dir)


if __name__ == "__main__":
    main()
