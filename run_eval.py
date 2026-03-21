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
    parser.add_argument("--time-limit", type=float, default=0.1, help="Time limit per move in seconds (default: 0.1)")
    parser.add_argument("--no-tqdm", action="store_true", help="Disable progress bar")
    parser.add_argument("--max-moves", type=int, default=None, help="Max moves per game before draw (default: 200)")
    args = parser.parse_args()

    new_mod = importlib.import_module(args.new_module)
    old_mod = importlib.import_module(args.old_module)

    NewBot = getattr(new_mod, args.new_class)
    OldBot = getattr(old_mod, args.old_class)

    import evaluate as eval_mod
    kwargs = dict(num_games=args.games, time_limit=args.time_limit, use_tqdm=not args.no_tqdm)
    if args.max_moves is not None:
        kwargs['max_moves'] = args.max_moves
    eval_mod.evaluate(NewBot(time_limit=args.time_limit), OldBot(time_limit=args.time_limit), **kwargs)


if __name__ == "__main__":
    main()
