"""Generate training positions with hand-tuned eval scores.

Plays games via self-play, snapshots positions after each full turn,
and stores the hand-tuned evaluate_position() score for each.

To ensure diversity, one of the bot's two moves per turn is replaced
with a random candidate when the position has already been seen.

Saves incrementally to disk. Resume by running again with the same output path.

Usage: python -m learned_eval.generate_positions [--time-limit 0.05]
"""

import os
import pickle
import random
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import HexGame, Player
from ai import MinimaxBot, get_candidates, evaluate_position

MAX_MOVES = 200
TARGET_POSITIONS = 50_000
SAVE_INTERVAL = 50  # save every N batches


def _board_key(board, current_player):
    """Hashable key for a (board, current_player) pair."""
    return hash((frozenset(board.items()), current_player))


def play_game_collect(args):
    """Play one game, collecting positions with their eval scores.

    At each turn, if the position has already been seen, replace one of
    the bot's two moves with a random candidate to diversify.
    """
    time_limit, game_idx, seen_keys = args
    rng = random.Random(game_idx)
    game = HexGame(win_length=6)
    bot_a = MinimaxBot(time_limit=time_limit)
    bot_b = MinimaxBot(time_limit=time_limit)

    positions = []
    total_stones = 0

    while not game.game_over and total_stones < MAX_MOVES:
        board_snap = dict(game.board)
        cp = game.current_player

        seen = board_snap and _board_key(board_snap, cp) in seen_keys

        # Always get the bot's moves
        bot = bot_a if cp == Player.A else bot_b
        bot_moves = bot.get_move(game)

        if not seen:
            moves = bot_moves
        else:
            # Replace one of the bot's moves with a random candidate
            moves = list(bot_moves)
            if len(moves) == 2:
                candidates = get_candidates(game)
                alt = [c for c in candidates if c not in moves]
                if alt:
                    replace_idx = rng.randint(0, 1)
                    moves[replace_idx] = rng.choice(alt)
            elif len(moves) == 1:
                candidates = get_candidates(game)
                alt = [c for c in candidates if c not in moves]
                if alt:
                    moves[0] = rng.choice(alt)

        for q, r in moves:
            if game.game_over:
                break
            if not game.make_move(q, r):
                break
            total_stones += 1

    tagged = []
    for board_snap, cp, eval_score in positions:
        bk = _board_key(board_snap, cp)
        tagged.append((bk, board_snap, cp, eval_score, game_idx))

    return tagged, total_stones


def _load_existing(path):
    """Load existing positions from disk, returning the unique dict."""
    unique = {}
    if os.path.exists(path):
        with open(path, "rb") as f:
            positions = pickle.load(f)
        for entry in positions:
            board, cp, eval_score, game_id = entry
            bk = _board_key(board, cp)
            unique[bk] = [board, cp, eval_score, game_id]
        print(f"Resumed: loaded {len(unique)} existing positions from {path}")
    return unique


def _save(unique, path):
    """Save current positions to disk."""
    positions = [(v[0], v[1], v[2], v[3]) for v in unique.values()]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(positions, f)
    os.replace(tmp, path)


def main():
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--time-limit", type=float, default=0.05)
    parser.add_argument("--target", type=int, default=TARGET_POSITIONS)
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "data", "positions.pkl"))
    args = parser.parse_args()

    workers = os.cpu_count() or 1
    batch_size = max(workers * 2, 20)

    unique = _load_existing(args.output)
    games_played = 0
    game_idx = int(time.time() * 1000) % 1_000_000
    games_since_save = 0
    total_moves = 0

    pbar = tqdm(total=args.target, initial=len(unique), unit="pos", desc="Unique positions")

    try:
        with Pool(workers) as pool:
            while len(unique) < args.target:
                seen_keys = frozenset(unique.keys())
                task_args = [(args.time_limit, game_idx + i, seen_keys)
                             for i in range(batch_size)]
                game_idx += batch_size

                for game_positions, move_count in pool.imap_unordered(play_game_collect, task_args):
                    prev_unique = len(unique)
                    games_played += 1
                    games_since_save += 1
                    total_moves += move_count
                    for bk, board_snap, cp, eval_score, gid in game_positions:
                        if bk not in unique:
                            unique[bk] = [board_snap, cp, eval_score, gid]

                    new = len(unique) - prev_unique
                    pbar.update(new)
                    pbar.set_postfix(games=games_played, unique=len(unique))

                if games_since_save >= SAVE_INTERVAL * batch_size:
                    _save(unique, args.output)
                    games_since_save = 0

    except KeyboardInterrupt:
        print(f"\n\nInterrupted! Saving {len(unique)} positions...")

    pbar.close()
    _save(unique, args.output)

    evals = [v[2] for v in unique.values()]
    avg_moves = total_moves / games_played if games_played else 0
    print(f"\nCollected {len(unique)} unique positions in {games_played} games")
    print(f"  Avg moves/game: {avg_moves:.1f}")
    print(f"  Eval range: [{min(evals):.0f}, {max(evals):.0f}]")
    print(f"  Eval mean: {sum(evals)/len(evals):.0f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
