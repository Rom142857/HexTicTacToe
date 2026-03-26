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
from scipy import sparse
from tqdm import tqdm

_PLAYER_MAP = None  # set lazily for parquet loading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import Player, HEX_DIRECTIONS
# Hand-tuned line scores by piece count (index = num pieces in a row)
LINE_SCORES = [0, 0, 8, 1200, 3000, 50000, 100000]
from learned_eval.pattern_table import build_arrays

DIR_VECTORS = list(HEX_DIRECTIONS)
_WIN_LENGTH = 6

# These globals are set by main() based on --window-length
WINDOW_LENGTH = 6
ENFORCE_PIECE_SWAP = True
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

    Windows are 6 cells long. Patterns are read from current_player's perspective
    (current_player = 1, opponent = 2).
    Returns None if the board has a completed 6-in-a-row (game already over).
    """
    opponent = Player.B if current_player == Player.A else Player.A
    features = {}
    seen = set()

    for (q, r) in board:
        for d_idx, (dq, dr) in enumerate(DIR_VECTORS):
            for k in range(WINDOW_LENGTH):
                sq = q - k * dq
                sr = r - k * dr
                wkey = (d_idx, sq, sr)
                if wkey in seen:
                    continue
                seen.add(wkey)

                pat_int = 0
                has_piece = False
                my_count = 0
                opp_count = 0
                power = 1
                for j in range(WINDOW_LENGTH):
                    cell = board.get((sq + j * dq, sr + j * dr))
                    if cell is None:
                        v = 0
                    elif cell == current_player:
                        v = 1
                        has_piece = True
                        my_count += 1
                    else:
                        v = 2
                        has_piece = True
                        opp_count += 1
                    pat_int += v * power
                    power *= 3

                if my_count == WINDOW_LENGTH or opp_count == WINDOW_LENGTH:
                    return None

                if not has_piece:
                    continue

                ci = CANON_INDEX[pat_int]
                cs = CANON_SIGN[pat_int]
                if cs == 0:
                    continue

                if ci in features:
                    features[ci] += cs
                else:
                    features[ci] = cs

    return features


def _init_worker(wlen, enforce_piece_swap=True):
    """Initialize pattern table globals in each worker process."""
    global WINDOW_LENGTH, CANON_PATTERNS, CANON_INDEX, CANON_SIGN, NUM_CANON
    if CANON_INDEX is None:
        WINDOW_LENGTH = wlen
        CANON_PATTERNS, CANON_INDEX, CANON_SIGN, NUM_CANON, _ = build_arrays(wlen, enforce_piece_swap=enforce_piece_swap)


def _extract_one(args):
    """Worker: extract features for a single position. Returns (row, features, target, win, board_size) or None."""
    row, entry, target_idx, win_idx = args
    board, cp = entry[0], entry[1]
    features = extract_features(board, cp)
    if not features:
        return None
    target = entry[target_idx]
    win = entry[win_idx] if win_idx is not None else None
    return (row, features, target, win, len(board))


def build_dataset(positions, target_idx=2, win_idx=None, binary_targets=False,
                  temporal_alpha=0.0):
    """Convert positions to sparse CSR feature matrix and target vector.

    Uses multiprocessing for feature extraction and returns a scipy.sparse
    CSR matrix for fast vectorized training.
    """
    from multiprocessing import Pool
    t0 = time.time()
    n = len(positions)

    workers = os.cpu_count() or 1
    args = [(i, positions[i], target_idx, win_idx) for i in range(n)]

    # Collect sparse entries
    rows, cols, vals = [], [], []
    raw_targets = []
    win_scores = []
    board_sizes = []

    with Pool(workers, initializer=_init_worker, initargs=(WINDOW_LENGTH, ENFORCE_PIECE_SWAP)) as pool:
        for result in tqdm(pool.imap(_extract_one, args, chunksize=256),
                           total=n, desc="Extracting features", unit="pos"):
            if result is None:
                continue
            orig_row, features, target, win, bsize = result
            dense_row = len(raw_targets)
            raw_targets.append(target)
            board_sizes.append(bsize)
            if win is not None:
                win_scores.append(win)
            for ci, cv in features.items():
                rows.append(dense_row)
                cols.append(ci)
                vals.append(cv)

    n_out = len(raw_targets)
    feat_matrix = sparse.csr_matrix(
        (np.array(vals, dtype=np.float64),
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(n_out, NUM_CANON))

    raw_targets = np.array(raw_targets, dtype=np.float64)
    win_scores = np.array(win_scores, dtype=np.float64) if win_scores else None

    if binary_targets:
        targets = raw_targets
    else:
        targets = np.tanh(raw_targets)

    if temporal_alpha > 0 and board_sizes:
        board_sizes = np.array(board_sizes, dtype=np.float64)
        max_bs = board_sizes.max()
        weights = (board_sizes / max_bs) ** temporal_alpha
        print(f"  Temporal weighting (alpha={temporal_alpha}): "
              f"min_weight={weights.min():.3f}, mean={weights.mean():.3f}")
    else:
        weights = np.ones(n_out, dtype=np.float64)

    normal = raw_targets[np.abs(raw_targets) < 4999]
    n_forced = len(raw_targets) - len(normal)
    print(f"  Done in {time.time()-t0:.1f}s, {n_out} usable positions ({n_forced} forced wins)")
    if len(normal) > 0:
        print(f"  Non-forced scores: mean={normal.mean():.3f}, std={normal.std():.3f}, "
              f"range=[{normal.min():.2f}, {normal.max():.2f}]")
    if win_scores is not None:
        n_decisive = np.sum(win_scores != 0)
        print(f"  Win scores: {int(np.sum(win_scores > 0))}W / {int(np.sum(win_scores < 0))}L / "
              f"{int(np.sum(win_scores == 0))}D ({n_decisive} decisive)")
    return feat_matrix, targets, weights, win_scores


def _evaluate(params, feat_matrix, targets, weights, win_scores=None):
    """Compute MSE loss in tanh space, R², and game-outcome accuracy."""
    n = len(targets)
    if n == 0:
        return 0.0, 0.0, 0.0
    raw_preds = feat_matrix.dot(params)
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


def train(train_data, val_data, num_params, epochs=200, lr=0.01, l2=0.001,
          reg_toward_init=False, init_weights=None, batch_size=4096):
    """Train pattern values using mini-batch Adam on tanh-space MSE + L2.

    pred = tanh(dot(params, features)),  target = tanh(score).
    Uses scipy.sparse CSR matrix for vectorized forward/backward passes.
    Mini-batches provide stochastic regularization and faster updates.

    If init_weights is a numpy array, use it as both initial params and L2 anchor.
    If reg_toward_init=True (and no init_weights), L2 regularizes toward LINE_SCORES init.
    """
    tr_matrix, tr_targets, tr_weights, tr_wins = train_data
    if init_weights is not None:
        params = init_weights.copy()
        init_params = init_weights.copy()
    else:
        params = _init_from_line_scores(num_params)
        init_params = params.copy() if reg_toward_init else None
    n = len(tr_targets)

    # Adam state
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    val_matrix, val_targets, val_weights, val_wins = val_data
    has_val = len(val_targets) > 0

    n_batches = max(1, (n + batch_size - 1) // batch_size)
    print(f"\nTraining {num_params} parameters on {n} train / {len(val_targets)} val "
          f"for {epochs} epochs (lr={lr}, l2={l2}, batch_size={batch_size}, {n_batches} batches/epoch)...")
    t0 = time.time()
    rng = np.random.RandomState(42)
    global_step = 0
    total_steps = epochs * n_batches

    epoch_bar = tqdm(range(epochs), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        # Shuffle indices each epoch
        perm = rng.permutation(n)

        epoch_loss = 0.0
        for batch_start in range(0, n, batch_size):
            batch_idx = perm[batch_start:batch_start + batch_size]
            b_matrix = tr_matrix[batch_idx]
            b_targets = tr_targets[batch_idx]
            b_weights = tr_weights[batch_idx]
            b_n = len(batch_idx)

            # Forward pass
            raw_preds = b_matrix.dot(params)
            sig_preds = np.tanh(raw_preds)
            residuals = sig_preds - b_targets
            epoch_loss += float(np.sum(b_weights * residuals ** 2))

            # Backward pass
            sig_deriv = 1.0 - sig_preds ** 2
            scaled_res = 2.0 * b_weights * residuals * sig_deriv / b_n

            grad = b_matrix.T.dot(scaled_res)
            if init_params is not None:
                grad += 2 * l2 * (params - init_params)
            else:
                grad += 2 * l2 * params

            # Adam update with linear warmup + cosine decay
            global_step += 1
            warmup_steps = 20 * n_batches
            if global_step < warmup_steps:
                cur_lr = lr * global_step / warmup_steps
            else:
                progress = (global_step - warmup_steps) / max(total_steps - warmup_steps, 1)
                cur_lr = lr * 0.5 * (1 + math.cos(math.pi * progress))

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad ** 2
            m_hat = m / (1 - beta1 ** global_step)
            v_hat = v / (1 - beta2 ** global_step)
            params -= cur_lr * m_hat / (np.sqrt(v_hat) + eps)

        mse_loss = epoch_loss / n
        epoch_bar.set_postfix(mse=f"{mse_loss:.5f}", lr=f"{cur_lr:.5f}")

        if epoch % 20 == 0 or epoch == epochs - 1:
            # Full evaluation
            raw_preds_full = tr_matrix.dot(params)
            sig_full = np.tanh(raw_preds_full)
            res_full = sig_full - tr_targets
            ss_res = float(np.sum(res_full ** 2))
            ss_tot = float(np.sum((tr_targets - np.mean(tr_targets)) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            _, _, tr_acc = _evaluate(params, tr_matrix, tr_targets, tr_weights, tr_wins)
            elapsed = time.time() - t0
            if has_val:
                v_loss, v_r2, v_acc = _evaluate(params, val_matrix, val_targets,
                                                val_weights, val_wins)
                epoch_bar.write(
                    f"  epoch {epoch:4d}: train mse={mse_loss:.6f} R²={r2:.4f} acc={tr_acc:.3f}"
                    f"  |  val mse={v_loss:.6f} R²={v_r2:.4f} acc={v_acc:.3f}  ({elapsed:.1f}s)")
            else:
                epoch_bar.write(
                    f"  epoch {epoch:4d}: mse={mse_loss:.6f} R²={r2:.4f} acc={tr_acc:.3f}  ({elapsed:.1f}s)")

    # Final metrics
    _, _, tr_acc = _evaluate(params, tr_matrix, tr_targets, tr_weights, tr_wins)
    if has_val:
        v_loss, v_r2, v_acc = _evaluate(params, val_matrix, val_targets,
                                        val_weights, val_wins)
        print(f"\nFinal: train mse={mse_loss:.6f} R²={r2:.4f} acc={tr_acc:.3f}"
              f"  |  val mse={v_loss:.6f} R²={v_r2:.4f} acc={v_acc:.3f}")
    else:
        print(f"\nFinal: mse={mse_loss:.6f} R²={r2:.4f} acc={tr_acc:.3f}")
    return params


def save_results(params, output_dir):
    """Save the trained lookup table, always expanded to full no-piece-swap patterns.

    If trained with piece-swap symmetry, expands by applying sign rules.
    This way ai_tuned.py never needs to know about symmetry.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Always expand to the full 377-pattern table (no piece-swap)
    from learned_eval.pattern_table import build_arrays as _ba, _pattern_to_int
    full_patterns, full_index, full_sign, full_ncanon, _ = _ba(WINDOW_LENGTH, enforce_piece_swap=False)

    expanded = np.zeros(full_ncanon, dtype=np.float64)
    for i, pat in enumerate(full_patterns):
        pi = _pattern_to_int(pat)
        ci = CANON_INDEX[pi]
        cs = CANON_SIGN[pi]
        if cs != 0 and ci >= 0:
            expanded[i] = cs * params[ci]

    result = {"_meta": {"score_scale": SCORE_SCALE, "window_length": WINDOW_LENGTH,
                        "piece_swap_symmetry": False}}
    for i, pat in enumerate(full_patterns):
        pat_str = "".join(str(c) for c in pat)
        result[pat_str] = float(expanded[i])

    json_path = os.path.join(output_dir, "pattern_values.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    # Save numpy array (expanded)
    npy_path = os.path.join(output_dir, "pattern_values.npy")
    np.save(npy_path, expanded)

    # Print top patterns
    sorted_idx = np.argsort(expanded)
    print(f"\nTop 20 most valuable patterns (for current player):")
    for i in sorted_idx[-20:][::-1]:
        pat_str = "".join(str(c) for c in full_patterns[i])
        readable = pat_str.replace("0", ".").replace("1", "X").replace("2", "O")
        print(f"  {readable:>8s}  {expanded[i]:+.3f}")

    print(f"\nTop 20 worst patterns (for current player):")
    for i in sorted_idx[:20]:
        pat_str = "".join(str(c) for c in full_patterns[i])
        readable = pat_str.replace("0", ".").replace("1", "X").replace("2", "O")
        print(f"  {readable:>8s}  {expanded[i]:+.3f}")

    print(f"\nSaved to {json_path} and {npy_path}")
    return json_path


def split_by_game(positions, val_fraction=0.2, seed=42):
    """Split positions into train/val by originating game_id.

    Each position has (board, current_player, eval_score, game_id).
    Positions from the same game end up in the same split, avoiding
    leakage from correlated positions within a game.
    """
    rng = np.random.RandomState(seed)

    # game_id index: 4-tuple=3, 5-tuple=4, 6-tuple=4, 7-tuple=4
    tlen = len(positions[0])
    gid_idx = 3 if tlen == 4 else 4
    has_game_ids = tlen >= 4
    if has_game_ids:
        game_ids = sorted({entry[gid_idx] for entry in positions}, key=str)
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


def _parquet_row_to_tuple(args):
    """Convert a parquet row (as dict) to a 5-tuple for training."""
    board_json, cp_int, eval_score, win_score, game_id = args
    board = {tuple(map(int, k.split(','))): _PLAYER_MAP[v]
             for k, v in json.loads(board_json).items()}
    return (board, _PLAYER_MAP[cp_int], eval_score, win_score, game_id)


def _init_parquet_worker():
    """Initialize Player map in worker processes."""
    global _PLAYER_MAP
    if _PLAYER_MAP is None:
        from game import Player
        _PLAYER_MAP = {1: Player.A, 2: Player.B}


def load_parquet(path):
    """Load a parquet file and convert to list of 5-tuples.

    Uses multiprocessing for fast conversion of board JSON strings.
    Returns list of (board_dict, Player, eval_score, win_score, game_id).
    """
    import pandas as pd
    from multiprocessing import Pool

    global _PLAYER_MAP
    _PLAYER_MAP = {1: Player.A, 2: Player.B}

    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} rows from {path}")

    args = list(zip(df['board'], df['current_player'], df['eval_score'],
                     df['win_score'], df['game_id']))

    workers = os.cpu_count() or 1
    positions = []
    with Pool(workers, initializer=_init_parquet_worker) as pool:
        for result in tqdm(pool.imap(_parquet_row_to_tuple, args, chunksize=1024),
                           total=len(args), desc="Converting parquet", unit="pos"):
            positions.append(result)

    return positions


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
    parser.add_argument("--no-piece-swap", action="store_true",
                        help="Disable piece-swap symmetry (use 377 params instead of 195)")
    parser.add_argument("--reg-toward-init", action="store_true",
                        help="Regularize toward LINE_SCORES init instead of toward zero")
    parser.add_argument("--init-weights", type=str, default=None,
                        help="Path to .npy file to use as initial params and L2 anchor")
    parser.add_argument("--temporal-weight", type=float, default=0.0,
                        help="Weight positions by game progress: weight = (board_size/max_board_size)^alpha. "
                             "0 = disabled (default), try 1.0 or 2.0")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Mini-batch size for training (default: 4096)")
    parser.add_argument("--max-moves-from-human", type=int, default=0,
                        help="Only keep positions within N moves of the human start (6/7-tuple only, 0=all)")
    parser.add_argument("--max-games", type=int, default=0,
                        help="Limit to positions from at most N games (0=all, deterministic selection)")
    args = parser.parse_args()

    # Initialize pattern tables for the requested window length
    global WINDOW_LENGTH, ENFORCE_PIECE_SWAP, CANON_PATTERNS, CANON_INDEX, CANON_SIGN, NUM_CANON
    WINDOW_LENGTH = args.window_length
    enforce_ps = not args.no_piece_swap
    ENFORCE_PIECE_SWAP = enforce_ps
    CANON_PATTERNS, CANON_INDEX, CANON_SIGN, NUM_CANON, _ = build_arrays(WINDOW_LENGTH, enforce_piece_swap=enforce_ps)
    sym_label = "with" if enforce_ps else "without"
    print(f"Window length {WINDOW_LENGTH}: {NUM_CANON} canonical patterns ({sym_label} piece-swap symmetry)")

    if args.input.endswith(".parquet"):
        positions = load_parquet(args.input)
    else:
        with open(args.input, "rb") as f:
            positions = pickle.load(f)
        print(f"Loaded {len(positions)} positions from {args.input}")

    # Determine target index based on tuple length
    tlen = len(positions[0])
    print(f"Data format: {tlen}-tuple")
    if tlen == 4:
        # (board, cp, eval_score, game_id)
        is_extended = False
        if args.target == "win":
            target_idx = 2  # eval IS the game outcome for 4-tuple
            max_abs = max(abs(p[target_idx]) for p in positions)
            if max_abs > 1.5:
                print(f"Normalizing 4-tuple win labels from ±{max_abs:.0f} to ±1")
                positions = [
                    (p[0], p[1], p[2] / max_abs, p[3]) for p in positions
                ]
        else:
            target_idx = 2
    elif tlen == 5:
        # (board, cp, eval_score, win_score, game_id)
        is_extended = False
        target_idx = 3 if args.target == "win" else 2
    elif tlen in (6, 7):
        # 6-tuple: (board, cp, moves_from_human, win_score, game_id, is_human_start)
        # 7-tuple: (board, cp, moves_from_human, win_score, playout_id, human_game_id, is_human_start)
        is_extended = True
        if args.target == "win":
            target_idx = 3  # win_score
        else:
            # No eval score in extended format; fall back to win
            print("WARNING: No eval score in extended playout data, using win_score")
            target_idx = 3
    else:
        raise ValueError(f"Unknown tuple length {tlen}")

    if args.target == "win":
        print(f"Training on game outcomes (entry[{target_idx}])")
    else:
        print(f"Training on search scores (entry[{target_idx}])")

    # Filter by moves_from_human (only for 6/7-tuple expanded playout data)
    if args.max_moves_from_human > 0 and tlen in (6, 7):
        before = len(positions)
        positions = [p for p in positions if p[2] <= args.max_moves_from_human]
        print(f"Filtered moves_from_human <= {args.max_moves_from_human}: {before} -> {len(positions)} positions")

    # Filter out draws if requested (win_score at target_idx for win target)
    if args.no_draws and args.target == "win":
        before = len(positions)
        positions = [p for p in positions if p[target_idx] != 0.0]
        print(f"Removed {before - len(positions)} draw positions, {len(positions)} remaining")

    # game_id index: 4-tuple=3, 5-tuple=4, 6-tuple=4, 7-tuple=4 (playout_id)
    gid_idx = 3 if tlen == 4 else 4

    # Limit to N games (deterministic selection)
    if args.max_games > 0:
        from collections import defaultdict as _dd
        by_game = _dd(list)
        for p in positions:
            by_game[p[gid_idx]].append(p)
        game_ids = sorted(by_game.keys(), key=str)
        rng = __import__('random').Random(42)
        rng.shuffle(game_ids)
        keep_ids = set(game_ids[:args.max_games])
        before = len(positions)
        positions = [p for p in positions if p[gid_idx] in keep_ids]
        print(f"Limited to {args.max_games} games: {before} -> {len(positions)} positions")

    # Subsample within each game to reduce correlation
    if args.subsample > 0:
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
    if tlen in (5, 6, 7):
        win_idx = 3
    elif args.target == "win":
        win_idx = 2  # 4-tuple: eval score IS win score
    else:
        win_idx = None

    train_pos, val_pos = split_by_game(positions, val_fraction=args.val_fraction)

    # Binary targets when training on win labels (±1 used directly, no tanh)
    binary = args.target == "win"
    tw = args.temporal_weight
    train_data = build_dataset(train_pos, target_idx=target_idx, win_idx=win_idx,
                               binary_targets=binary, temporal_alpha=tw)
    val_data = build_dataset(val_pos, target_idx=target_idx, win_idx=win_idx,
                             binary_targets=binary, temporal_alpha=tw)

    init_w = None
    if args.init_weights:
        init_w = np.load(args.init_weights)
        print(f"Loaded init weights from {args.init_weights} ({len(init_w)} params)")
    num_params = NUM_CANON
    params = train(train_data, val_data, num_params,
                   epochs=args.epochs, lr=args.lr, l2=args.l2,
                   reg_toward_init=args.reg_toward_init,
                   init_weights=init_w, batch_size=args.batch_size)
    save_results(params, args.output_dir)


if __name__ == "__main__":
    main()
