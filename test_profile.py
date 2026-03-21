"""Profile ai.py or run ai.py vs og_ai.py competition.

Usage:
    python test_profile.py              # profile ai.py (5 self-play games)
    python test_profile.py --competition  # ai.py vs og_ai.py, 30 games
    python test_profile.py --competition -n 50  # ai.py vs og_ai.py, 50 games
    python test_profile.py --games 10   # profile over 10 self-play games
"""

import argparse
import cProfile
import pstats
import io
from game import HexGame, Player


def run_games(num_games=5):
    """Play num_games using MinimaxBot for both sides, time_limit=0.1."""
    from ai import MinimaxBot
    bot = MinimaxBot(time_limit=0.1)
    total_turns = 0
    total_stones = 0
    results = {Player.A: 0, Player.B: 0, Player.NONE: 0}

    for g in range(num_games):
        game = HexGame(win_length=6)
        turn = 0
        while not game.game_over and turn < 30:
            moves = bot.get_move(game)
            for q, r in moves:
                if not game.game_over:
                    game.make_move(q, r)
            turn += 1
        total_turns += turn
        total_stones += game.move_count
        results[game.winner] += 1
        print(f"  Game {g+1}: {turn} turns, {game.move_count} stones, "
              f"winner={'draw' if game.winner == Player.NONE else game.winner.name}")

    print(f"\nTotals: {num_games} games, {total_turns} turns, {total_stones} stones")
    print(f"  A wins: {results[Player.A]}, B wins: {results[Player.B]}, "
          f"draws: {results[Player.NONE]}")


def run_profile(num_games):
    """Profile ai.py over multiple self-play games."""
    print(f"Profiling ai.py over {num_games} self-play games (time_limit=0.1)\n")

    pr = cProfile.Profile()
    pr.enable()
    run_games(num_games)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats("cumulative")
    print("\n=== Top 30 by cumulative time ===")
    ps.print_stats(30)
    print(s.getvalue())

    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2)
    ps2.sort_stats("tottime")
    print("\n=== Top 30 by total (self) time ===")
    ps2.print_stats(30)
    print(s2.getvalue())


def run_competition(num_games):
    """Run ai.py vs og_ai.py using evaluate.py's framework."""
    from evaluate import evaluate, load_bot

    print(f"Competition: ai.py vs og_ai.py, {num_games} games (time_limit=0.1)\n")
    a = load_bot("ai", time_limit=0.1)
    b = load_bot("og_ai", time_limit=0.1)
    evaluate(a, b, num_games=num_games)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile ai.py or run competition")
    parser.add_argument("--competition", action="store_true",
                        help="Run ai.py vs og_ai.py instead of profiling")
    parser.add_argument("-n", "--games", type=int, default=None,
                        help="Number of games (default: 5 for profile, 30 for competition)")
    args = parser.parse_args()

    if args.competition:
        run_competition(args.games or 30)
    else:
        run_profile(args.games or 5)
