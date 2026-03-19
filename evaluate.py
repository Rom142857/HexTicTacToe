"""Evaluate bots by playing them against each other.

Usage:
    python evaluate.py [num_games]
"""

import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from game import HexGame, Player
from bot import RandomBot
from ai import MinimaxBot


# Grace factor: allow 3x the time limit before counting as a violation.
# This filters out occasional OS scheduling jitter.
GRACE_FACTOR = 3.0
# If a bot accumulates this many violations in a single game, forfeit that game.
MAX_VIOLATIONS_PER_GAME = 10


class TimeLimitExceeded(Exception):
    """Raised when a bot consistently exceeds its time limit."""
    def __init__(self, bot, violations):
        self.bot = bot
        self.violations = violations
        super().__init__(f"{bot} exceeded time limit {violations} times")


def play_game(bot_a, bot_b, radius=5, win_length=6, violations=None):
    """Play one game. Returns (winner, depth_counts_a, depth_counts_b).

    depth_counts maps depth -> number of moves at that depth.
    violations is a dict {bot: count} tracked within this game.
    """
    game = HexGame(radius=radius, win_length=win_length)
    bots = {Player.A: bot_a, Player.B: bot_b}
    depths = {Player.A: defaultdict(int), Player.B: defaultdict(int)}

    while not game.game_over:
        player = game.current_player
        bot = bots[player]

        t0 = time.time()
        q, r = bot.get_move(game)
        elapsed = time.time() - t0

        if elapsed > bot.time_limit * GRACE_FACTOR:
            if violations is not None:
                violations[bot] = violations.get(bot, 0) + 1
                if violations[bot] >= MAX_VIOLATIONS_PER_GAME:
                    raise TimeLimitExceeded(bot, violations[bot])

        depths[player][bot.last_depth] += 1

        if not game.make_move(q, r):
            return (Player.B if player == Player.A else Player.A,
                    depths[Player.A], depths[Player.B])

    return game.winner, depths[Player.A], depths[Player.B]


def _play_one(args):
    """Worker function for multiprocessing. Plays a single game."""
    bot_a, bot_b, game_idx, radius, win_length = args
    swapped = game_idx % 2 == 1

    if swapped:
        seat_a, seat_b = bot_b, bot_a
    else:
        seat_a, seat_b = bot_a, bot_b

    violations = {}
    exceeded = False
    try:
        winner, d_a, d_b = play_game(seat_a, seat_b, radius, win_length, violations)
    except TimeLimitExceeded:
        exceeded = True
        winner = Player.NONE
        d_a, d_b = defaultdict(int), defaultdict(int)

    return (
        winner,
        swapped,
        dict(d_a),
        dict(d_b),
        violations.get(seat_a, 0),
        violations.get(seat_b, 0),
        exceeded,
    )


def evaluate(bot_a, bot_b, num_games=100, radius=5, win_length=6):
    """Play num_games between two bots in parallel, swapping sides each game."""
    bot_a_wins = 0
    bot_b_wins = 0
    draws = 0
    games_played = 0
    bot_a_depths = defaultdict(int)
    bot_b_depths = defaultdict(int)
    bot_a_violations = 0
    bot_b_violations = 0
    aborted_games = 0

    workers = os.cpu_count() or 1
    args = [(bot_a, bot_b, i, radius, win_length) for i in range(num_games)]

    t0 = time.time()

    with Pool(workers) as pool:
        for result in pool.imap_unordered(_play_one, args):
            winner, swapped, d_a, d_b, v_a, v_b, exceeded = result

            if exceeded:
                aborted_games += 1

            if swapped:
                # seat A = bot_b, seat B = bot_a
                for d, c in d_a.items():
                    bot_b_depths[d] += c
                for d, c in d_b.items():
                    bot_a_depths[d] += c
                bot_b_violations += v_a
                bot_a_violations += v_b
                if winner == Player.A:
                    bot_b_wins += 1
                elif winner == Player.B:
                    bot_a_wins += 1
                else:
                    draws += 1
            else:
                for d, c in d_a.items():
                    bot_a_depths[d] += c
                for d, c in d_b.items():
                    bot_b_depths[d] += c
                bot_a_violations += v_a
                bot_b_violations += v_b
                if winner == Player.A:
                    bot_a_wins += 1
                elif winner == Player.B:
                    bot_b_wins += 1
                else:
                    draws += 1

            games_played += 1
            if games_played % 10 == 0:
                print(".", end="", flush=True)

    elapsed = time.time() - t0
    total = max(games_played, 1)

    print(f"\n\n{'='*50}")
    print(f"  {bot_a} vs {bot_b}  —  {games_played} games in {elapsed:.1f}s")
    print(f"{'='*50}")
    na, nb = str(bot_a), str(bot_b)
    print(f"  {na:>15s}: {bot_a_wins:3d} wins ({100*bot_a_wins/total:.0f}%)")
    print(f"  {nb:>15s}: {bot_b_wins:3d} wins ({100*bot_b_wins/total:.0f}%)")
    print(f"  {'Draws':>15s}: {draws:3d}      ({100*draws/total:.0f}%)")
    print()

    for name, depths in [(str(bot_a), bot_a_depths), (str(bot_b), bot_b_depths)]:
        if not depths:
            continue
        total_moves = sum(depths.values())
        avg = sum(d * c for d, c in depths.items()) / total_moves
        lo, hi = min(depths), max(depths)
        print(f"  {name} search depth: avg {avg:.1f}, range [{lo}-{hi}]")
        buckets = sorted(depths.items())
        dist = "  ".join(f"d{d}:{c}" for d, c in buckets)
        print(f"    {dist}")

    if bot_a_violations or bot_b_violations or aborted_games:
        print()
        print(f"  TIME VIOLATIONS: {na}={bot_a_violations}, {nb}={bot_b_violations}"
              f"  ({aborted_games} games forfeited)")

    print(f"{'='*50}")

    return bot_a_wins, bot_b_wins, draws


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20

    a = MinimaxBot(time_limit=0.05)
    b = RandomBot(time_limit=0.05)

    evaluate(a, b, num_games=n)
