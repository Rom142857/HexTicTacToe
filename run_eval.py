"""Wrapper to evaluate ai.py vs og_ai.py.

Usage:
    python run_eval.py                          # defaults: MinimaxBot from both
    python run_eval.py --new-class MyBot        # custom class name in ai.py
    python run_eval.py --games 200              # fewer games
    python run_eval.py --time-limit 0.5         # custom time limit
"""

import argparse
import importlib
import sys


def main():
    parser = argparse.ArgumentParser(description="Evaluate ai.py vs og_ai.py")
    parser.add_argument("--new-module", default="ai", help="Module for new bot (default: ai)")
    parser.add_argument("--new-class", default="MinimaxBot", help="Class name in new module (default: MinimaxBot)")
    parser.add_argument("--old-module", default="og_ai", help="Module for old bot (default: og_ai)")
    parser.add_argument("--old-class", default="MinimaxBot", help="Class name in old module (default: MinimaxBot)")
    parser.add_argument("--games", type=int, default=400, help="Number of games (default: 400)")
    parser.add_argument("--time-limit", type=float, default=1.0, help="Time limit per move in seconds (default: 1.0)")
    args = parser.parse_args()

    new_mod = importlib.import_module(args.new_module)
    old_mod = importlib.import_module(args.old_module)

    NewBot = getattr(new_mod, args.new_class)
    OldBot = getattr(old_mod, args.old_class)

    from evaluate import evaluate
    evaluate(NewBot(time_limit=args.time_limit), OldBot(time_limit=args.time_limit),
             num_games=args.games, time_limit=args.time_limit)


if __name__ == "__main__":
    main()
