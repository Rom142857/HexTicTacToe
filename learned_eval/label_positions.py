"""Label existing positions with minimax search scores.

Takes positions (e.g. from human games) and evaluates each one with
MinimaxBot to produce search-score labels. Outputs 5-tuples:
  (board, current_player, search_score, win_score, game_id)

Usage: python -m learned_eval.label_positions [--input data/positions_human.pkl]
"""

import os
import pickle
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import HexGame, Player
from ai import MinimaxBot

SCORE_SCALE = 20_000


def _eval_position(args):
    """Evaluate a single position with MinimaxBot."""
    board, cp, win_score, game_id, time_limit, pattern_path = args

    game = HexGame(win_length=6)
    # Replay stones to build game state
    for (q, r), player in board.items():
        game.board[(q, r)] = player
    game.current_player = cp
    game.move_count = len(board)
    # Assume start of turn (2 moves left, or 1 for A's very first turn)
    game.moves_left_in_turn = 1 if (len(board) == 0 and cp == Player.A) else 2

    if not board:
        return board, cp, 0.0, win_score, game_id

    if pattern_path:
        from ai import MinimaxBot as TunedBot
        bot = TunedBot(time_limit=time_limit, pattern_path=pattern_path)
    else:
        bot = MinimaxBot(time_limit=time_limit)
    bot.get_move(game)
    search_score = bot.last_score / SCORE_SCALE

    return board, cp, search_score, win_score, game_id


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(
        os.path.dirname(__file__), "data", "positions_human.pkl"))
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(__file__), "data", "positions_human_labelled.pkl"))
    parser.add_argument("--time-limit", type=float, default=0.05)
    parser.add_argument("--pattern-path", type=str, default=None,
                        help="Path to pattern_values.json for ai_tuned (default: use hand-tuned ai)")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        positions = pickle.load(f)
    print(f"Loaded {len(positions)} positions from {args.input}")

    # Parse input format: 4-tuple (board, cp, win_label, game_id)
    is_4tuple = len(positions[0]) == 4
    pattern_path = args.pattern_path
    if pattern_path:
        print(f"Using ai_tuned with pattern_path={pattern_path}")
    tasks = []
    for p in positions:
        board, cp = p[0], p[1]
        if is_4tuple:
            raw_win = p[2]
            game_id = p[3]
        else:
            raw_win = p[3]
            game_id = p[4]
        # Normalize win labels to ±1
        win_score = 1.0 if raw_win > 0 else -1.0
        tasks.append((board, cp, win_score, game_id, args.time_limit, pattern_path))

    workers = os.cpu_count() or 1
    print(f"Evaluating with MinimaxBot (time_limit={args.time_limit}, workers={workers})...")

    results = []
    t0 = time.time()
    with Pool(workers) as pool:
        for i, result in enumerate(pool.imap(_eval_position, tasks, chunksize=32)):
            results.append(result)
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(tasks) - i - 1) / rate
                print(f"  {i+1}/{len(tasks)} ({rate:.0f} pos/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({len(results)/elapsed:.0f} pos/s)")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tmp = args.output + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(results, f)
    os.replace(tmp, args.output)

    # Stats
    import numpy as np
    evals = np.array([r[2] for r in results])
    wins = np.array([r[3] for r in results])
    print(f"\nSaved {len(results)} positions to {args.output}")
    print(f"  Eval range: [{evals.min():.2f}, {evals.max():.2f}]")
    print(f"  Eval mean: {evals.mean():.4f}, std: {evals.std():.4f}")
    print(f"  Win/Loss: {int((wins > 0).sum())}/{int((wins < 0).sum())}")

    # Correlation between eval and win
    decisive = wins != 0
    correct = ((evals[decisive] > 0) == (wins[decisive] > 0)).sum()
    print(f"  Eval-win agreement: {correct}/{decisive.sum()} ({100*correct/decisive.sum():.1f}%)")


if __name__ == "__main__":
    main()
