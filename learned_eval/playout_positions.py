"""Play out games from saved positions using ai_tuned, recording win/loss.

For each position, ai_tuned plays both sides to completion.
The outcome (win/loss from current_player's perspective) is recorded.

Output: 5-tuple (board, current_player, 0.0, win_score, game_id)
  where win_score = +1 if current_player wins, -1 if loses, 0 if draw.

Usage: python -m learned_eval.playout_positions [--input data/positions_human.pkl]
"""

import os
import pickle
import random
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import HexGame, Player, HEX_DIRECTIONS

SCORE_SCALE = 20_000
MAX_MOVES = 200
_WIN_LENGTH = 6


def _board_has_win(board):
    """Check if any player already has 6 in a row."""
    for (q, r), player in board.items():
        for dq, dr in HEX_DIRECTIONS:
            count = 1
            for i in range(1, _WIN_LENGTH):
                if board.get((q + dq * i, r + dr * i)) == player:
                    count += 1
                else:
                    break
            if count >= _WIN_LENGTH:
                return True
    return False


def _playout_position(args):
    """Play out a single position with ai_tuned on both sides."""
    board, cp, original_win, game_id, time_limit, pattern_path = args

    from ai_tuned import MinimaxBot as TunedBot

    game = HexGame(win_length=6)
    game.board = dict(board)
    game.current_player = cp
    game.move_count = len(board)
    game.moves_left_in_turn = 2

    bot_a = TunedBot(time_limit=time_limit, pattern_path=pattern_path)
    bot_b = TunedBot(time_limit=time_limit, pattern_path=pattern_path)
    bots = {Player.A: bot_a, Player.B: bot_b}

    total_moves = 0
    forfeit_player = None
    while not game.game_over and total_moves < MAX_MOVES:
        player = game.current_player
        bot = bots[player]
        moves = bot.get_move(game)
        moves = list(moves)

        invalid = False
        for q, r in moves:
            if game.game_over or not game.make_move(q, r):
                invalid = True
                break
            total_moves += 1

        if invalid:
            # Same as evaluate.py: invalid move = forfeit for current player
            forfeit_player = player
            break

    if forfeit_player is not None:
        # The player who made the invalid move loses
        winner = Player.B if forfeit_player == Player.A else Player.A
    else:
        winner = game.winner

    if winner == Player.NONE:
        win_score = 0.0
    elif winner == cp:
        win_score = 1.0
    else:
        win_score = -1.0

    return board, cp, 0.0, win_score, game_id


def main():
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(
        os.path.dirname(__file__), "data", "positions_human.pkl"))
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(__file__), "data", "positions_human_playouts.pkl"))
    parser.add_argument("--time-limit", type=float, default=0.03)
    parser.add_argument("--pattern-path", type=str, default=None,
                        help="Pattern values JSON for ai_tuned (default: built-in)")
    parser.add_argument("--subsample", type=int, default=3,
                        help="Keep every Nth position per game (default: 3)")
    parser.add_argument("--max-positions", type=int, default=0,
                        help="Max positions to process (0 = all)")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        positions = pickle.load(f)
    print(f"Loaded {len(positions)} positions from {args.input}")

    is_4tuple = len(positions[0]) == 4

    # Filter out positions where someone already won
    before = len(positions)
    positions = [p for p in positions if not _board_has_win(p[0])]
    if len(positions) < before:
        print(f"Removed {before - len(positions)} already-won positions")

    # Deduplicate by board state (keep first occurrence per unique board)
    seen_boards = set()
    unique = []
    for p in positions:
        board_key = frozenset(p[0].items())
        if board_key not in seen_boards:
            seen_boards.add(board_key)
            unique.append(p)
    if len(unique) < len(positions):
        print(f"Deduplicated: {len(positions)} -> {len(unique)} unique positions")
    positions = unique

    # Subsample within each game to reduce redundancy
    by_game = defaultdict(list)
    for p in positions:
        gid = p[3] if is_4tuple else p[4]
        by_game[gid].append(p)

    if args.subsample > 1:
        sampled = []
        for gid, game_pos in by_game.items():
            game_pos.sort(key=lambda p: len(p[0]))  # sort by board size
            sampled.extend(game_pos[::args.subsample])
    else:
        sampled = positions

    if args.max_positions > 0:
        rng = random.Random(42)
        rng.shuffle(sampled)
        sampled = sampled[:args.max_positions]

    print(f"After subsampling (every {args.subsample}): {len(sampled)} positions "
          f"from {len(by_game)} games")

    # Build task list
    tasks = []
    for p in sampled:
        board, cp = p[0], p[1]
        if is_4tuple:
            raw_win = p[2]
            game_id = p[3]
        else:
            raw_win = p[3]
            game_id = p[4]
        win_score = 1.0 if raw_win > 0 else -1.0
        tasks.append((board, cp, win_score, game_id, args.time_limit, args.pattern_path))

    workers = os.cpu_count() or 1
    print(f"Playing out {len(tasks)} positions with ai_tuned "
          f"(time_limit={args.time_limit}, workers={workers})...")

    results = []
    t0 = time.time()
    wins = 0
    losses = 0
    draws = 0

    with Pool(workers) as pool:
        for result in tqdm(pool.imap(_playout_position, tasks, chunksize=8),
                           total=len(tasks), desc="Playouts", unit="pos"):
            results.append(result)
            ws = result[3]
            if ws > 0:
                wins += 1
            elif ws < 0:
                losses += 1
            else:
                draws += 1

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({len(results)/elapsed:.1f} pos/s)")
    print(f"  Wins: {wins}, Losses: {losses}, Draws: {draws}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tmp = args.output + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(results, f)
    os.replace(tmp, args.output)
    print(f"Saved {len(results)} positions to {args.output}")


if __name__ == "__main__":
    main()
