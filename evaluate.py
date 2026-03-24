"""Evaluate bots by playing them against each other.

Usage:
    python evaluate.py [num_games]
"""

import os
import pickle
import random
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from tqdm import tqdm
from game import HexGame, Player
from bot import RandomBot
import importlib


# Grace factor: allow 3x the time limit before counting as a violation.
# This filters out occasional OS scheduling jitter.
GRACE_FACTOR = 3.0
# If a bot accumulates this many violations in a single game, forfeit that game.
MAX_VIOLATIONS_PER_GAME = 10
# Maximum total moves before declaring a draw.
MAX_MOVES_PER_GAME = 200


class TimeLimitExceeded(Exception):
    """Raised when a bot consistently exceeds its time limit."""
    def __init__(self, bot, violations):
        self.bot = bot
        self.violations = violations
        super().__init__(f"{bot} exceeded time limit {violations} times")


def _load_positions(path, num_games, seed=42):
    """Load positions from a pickle file, one per unique game, deterministically.

    Selects positions that are spread across different games using a fixed seed.
    Returns a list of (board_dict, current_player) tuples.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Group by game uid (handle 4-tuple and 5-tuple formats)
    by_uid = defaultdict(list)
    for entry in data:
        if len(entry) == 5:
            board, player, _score, _label, uid = entry
        else:
            board, player, _score, uid = entry
        by_uid[uid].append((board, player))

    # Pick one position per game (middle of the game for interesting positions)
    rng = random.Random(seed)
    uids = sorted(by_uid.keys())  # sort for determinism
    rng.shuffle(uids)
    positions = []
    for uid in uids[:num_games]:
        entries = by_uid[uid]
        # Pick from the middle third for non-trivial positions
        lo = len(entries) // 3
        hi = max(lo + 1, 2 * len(entries) // 3)
        idx = rng.randint(lo, hi - 1)
        positions.append(entries[idx])
    return positions


def _setup_game_from_position(board_dict, current_player, win_length=6):
    """Create a HexGame initialized to the given board position."""
    game = HexGame(win_length=win_length)
    game.board = dict(board_dict)
    game.current_player = current_player
    game.move_count = len(board_dict)
    game.moves_left_in_turn = 2
    return game


def play_game(bot_a, bot_b, win_length=6, violations=None, max_moves=None,
              start_position=None):
    """Play one game. Returns (winner, depth_counts_a, depth_counts_b, time_a, time_b).

    depth_counts maps depth -> number of moves at that depth.
    time_a/time_b are (total_seconds, num_moves) tuples.
    violations is a dict {bot: count} tracked within this game.
    """
    if max_moves is None:
        max_moves = MAX_MOVES_PER_GAME
    if start_position is not None:
        board_dict, current_player = start_position
        game = _setup_game_from_position(board_dict, current_player, win_length)
    else:
        game = HexGame(win_length=win_length)
    bots = {Player.A: bot_a, Player.B: bot_b}
    depths = {Player.A: defaultdict(int), Player.B: defaultdict(int)}
    times = {Player.A: [0.0, 0], Player.B: [0.0, 0]}  # [total_secs, num_moves]

    total_moves = 0
    while not game.game_over:
        player = game.current_player
        bot = bots[player]

        t0 = time.time()
        result = bot.get_move(game)
        elapsed = time.time() - t0

        # Normalize to list of moves based on bot's pair_moves flag
        if bot.pair_moves:
            moves = result
            num_moves = len(moves)
        else:
            moves = [result]
            num_moves = 1

        times[player][0] += elapsed
        times[player][1] += num_moves

        # Allow 2x time budget for pair-move bots
        allowed_time = bot.time_limit * num_moves
        if elapsed > allowed_time * GRACE_FACTOR:
            if violations is not None:
                violations[bot] = violations.get(bot, 0) + 1
                if violations[bot] >= MAX_VIOLATIONS_PER_GAME:
                    raise TimeLimitExceeded(bot, violations[bot])

        depths[player][bot.last_depth] += num_moves
        total_moves += num_moves

        if total_moves >= max_moves:
            return (Player.NONE, depths[Player.A], depths[Player.B],
                    tuple(times[Player.A]), tuple(times[Player.B]))

        invalid = False
        for q, r in moves:
            if game.game_over or not game.make_move(q, r):
                invalid = True
                break
        if invalid:
            return (Player.B if player == Player.A else Player.A,
                    depths[Player.A], depths[Player.B],
                    tuple(times[Player.A]), tuple(times[Player.B]))

    return (game.winner, depths[Player.A], depths[Player.B],
            tuple(times[Player.A]), tuple(times[Player.B]))


def _play_one(args):
    """Worker function for multiprocessing. Plays a single game."""
    bot_a, bot_b, game_idx, win_length, max_moves, start_position = args
    swapped = game_idx % 2 == 1

    if swapped:
        seat_a, seat_b = bot_b, bot_a
    else:
        seat_a, seat_b = bot_a, bot_b

    violations = {}
    exceeded = False
    try:
        winner, d_a, d_b, t_a, t_b = play_game(
            seat_a, seat_b, win_length, violations, max_moves,
            start_position=start_position)
    except TimeLimitExceeded:
        exceeded = True
        winner = Player.NONE
        d_a, d_b = defaultdict(int), defaultdict(int)
        t_a, t_b = (0.0, 0), (0.0, 0)

    move_count = t_a[1] + t_b[1]  # total moves in the game

    return (
        winner,
        swapped,
        dict(d_a),
        dict(d_b),
        violations.get(seat_a, 0),
        violations.get(seat_b, 0),
        exceeded,
        t_a,
        t_b,
        move_count,
    )


def evaluate(bot_a, bot_b, num_games=100, win_length=6, time_limit=0.1,
             use_tqdm=True, max_moves=None, positions=None):
    """Play num_games between two bots in parallel, swapping sides each game.

    If positions is provided (list of (board_dict, current_player) tuples),
    each position is played twice (each bot plays as the to-move side),
    and num_games is ignored.
    """
    if max_moves is None:
        max_moves = MAX_MOVES_PER_GAME
    bot_a.time_limit = time_limit
    bot_b.time_limit = time_limit
    bot_a_wins = 0
    bot_b_wins = 0
    draws = 0
    games_played = 0
    bot_a_depths = defaultdict(int)
    bot_b_depths = defaultdict(int)
    bot_a_violations = 0
    bot_b_violations = 0
    aborted_games = 0
    bot_a_time = [0.0, 0]  # [total_secs, num_moves]
    bot_b_time = [0.0, 0]
    game_lengths = []  # total moves per game

    workers = os.cpu_count() or 1
    if positions is not None:
        # Two games per position: game_idx even = normal, odd = swapped
        num_games = len(positions) * 2
        args = []
        for i, pos in enumerate(positions):
            args.append((bot_a, bot_b, i * 2,     win_length, max_moves, pos))
            args.append((bot_a, bot_b, i * 2 + 1, win_length, max_moves, pos))
    else:
        args = [(bot_a, bot_b, i, win_length, max_moves, None) for i in range(num_games)]

    t0 = time.time()

    with Pool(workers) as pool:
        results_iter = pool.imap_unordered(_play_one, args)
        if use_tqdm:
            results_iter = tqdm(results_iter, total=num_games, desc="Games", unit="game")
        for result in results_iter:
            winner, swapped, d_a, d_b, v_a, v_b, exceeded, t_a, t_b, move_count = result

            if exceeded:
                aborted_games += 1
            else:
                game_lengths.append(move_count)

            if swapped:
                # seat A = bot_b, seat B = bot_a
                for d, c in d_a.items():
                    bot_b_depths[d] += c
                for d, c in d_b.items():
                    bot_a_depths[d] += c
                bot_b_violations += v_a
                bot_a_violations += v_b
                bot_b_time[0] += t_a[0]; bot_b_time[1] += t_a[1]
                bot_a_time[0] += t_b[0]; bot_a_time[1] += t_b[1]
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
                bot_a_time[0] += t_a[0]; bot_a_time[1] += t_a[1]
                bot_b_time[0] += t_b[0]; bot_b_time[1] += t_b[1]
                if winner == Player.A:
                    bot_a_wins += 1
                elif winner == Player.B:
                    bot_b_wins += 1
                else:
                    draws += 1

            games_played += 1
            if use_tqdm:
                results_iter.set_postfix(A=bot_a_wins, B=bot_b_wins, D=draws)

    elapsed = time.time() - t0
    total = max(games_played, 1)

    print(f"\n\n{'='*50}")
    print(f"  {bot_a} vs {bot_b}  \u2014  {games_played} games in {elapsed:.1f}s")
    print(f"{'='*50}")
    na, nb = str(bot_a), str(bot_b)
    print(f"  {na:>15s}: {bot_a_wins:3d} wins ({100*bot_a_wins/total:.0f}%)")
    print(f"  {nb:>15s}: {bot_b_wins:3d} wins ({100*bot_b_wins/total:.0f}%)")
    print(f"  {'Draws':>15s}: {draws:3d}      ({100*draws/total:.0f}%)")
    # Total win rate: wins + half draws (standard tournament scoring)
    bot_a_score = (bot_a_wins + 0.5 * draws) / total
    print(f"\n  {na} total win rate: {100*bot_a_score:.1f}%")
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

    for name, bt in [(str(bot_a), bot_a_time), (str(bot_b), bot_b_time)]:
        if bt[1] > 0:
            avg_ms = 1000 * bt[0] / bt[1]
            print(f"  {name} avg move time: {avg_ms:.0f}ms ({bt[1]} moves)")

    if game_lengths:
        avg_len = sum(game_lengths) / len(game_lengths)
        lo_len, hi_len = min(game_lengths), max(game_lengths)
        print(f"\n  Game length: avg {avg_len:.1f} moves, range [{lo_len}-{hi_len}]")
        bin_size = 50
        num_bins = max_moves // bin_size
        bins = [0] * num_bins
        for gl in game_lengths:
            idx = min(gl // bin_size, num_bins - 1)
            bins[idx] += 1
        max_count = max(bins) if max(bins) > 0 else 1
        bar_width = 30
        for i, count in enumerate(bins):
            lo = i * bin_size
            hi = (i + 1) * bin_size - 1 if i < num_bins - 1 else max_moves
            bar = "█" * max(1, round(bar_width * count / max_count)) if count else ""
            print(f"    {lo:3d}-{hi:3d}: {bar} {count}")

    if bot_a_violations or bot_b_violations or aborted_games:
        print()
        print(f"  TIME VIOLATIONS: {na}={bot_a_violations}, {nb}={bot_b_violations}"
              f"  ({aborted_games} games forfeited)")

    print(f"{'='*50}")

    return bot_a_wins, bot_b_wins, draws


def _play_position_one(args):
    """Worker for position evaluation. Same bot on both sides."""
    bot, win_length, max_moves, start_position = args
    _board, to_move = start_position

    violations = {}
    exceeded = False
    try:
        winner, d_a, d_b, t_a, t_b = play_game(
            bot, bot, win_length, violations, max_moves,
            start_position=start_position)
    except TimeLimitExceeded:
        exceeded = True
        winner = Player.NONE
        d_a, d_b = defaultdict(int), defaultdict(int)
        t_a, t_b = (0.0, 0), (0.0, 0)

    move_count = t_a[1] + t_b[1]
    return (winner, to_move, dict(d_a), dict(d_b),
            exceeded, t_a, t_b, move_count)


def evaluate_positions(bot, positions, num_rollouts=20, win_length=6,
                       time_limit=0.1, use_tqdm=True, max_moves=None):
    """Play each position num_rollouts times with the same bot on both sides.

    Reports wins/losses from the perspective of the to-move color.
    """
    if max_moves is None:
        max_moves = MAX_MOVES_PER_GAME
    bot.time_limit = time_limit

    to_move_wins = 0
    to_move_losses = 0
    draws = 0
    aborted = 0
    depths = defaultdict(int)
    total_time = [0.0, 0]
    game_lengths = []

    args = []
    for pos in positions:
        for _ in range(num_rollouts):
            args.append((bot, win_length, max_moves, pos))

    total_games = len(args)
    workers = os.cpu_count() or 1

    t0 = time.time()
    with Pool(workers) as pool:
        results_iter = pool.imap_unordered(_play_position_one, args)
        if use_tqdm:
            results_iter = tqdm(results_iter, total=total_games,
                                desc="Rollouts", unit="game")
        for result in results_iter:
            winner, to_move, d_a, d_b, exceeded, t_a, t_b, move_count = result

            if exceeded:
                aborted += 1
            else:
                game_lengths.append(move_count)

            for d, c in d_a.items():
                depths[d] += c
            for d, c in d_b.items():
                depths[d] += c
            total_time[0] += t_a[0] + t_b[0]
            total_time[1] += t_a[1] + t_b[1]

            if winner == to_move:
                to_move_wins += 1
            elif winner == Player.NONE:
                draws += 1
            else:
                to_move_losses += 1

            if use_tqdm:
                results_iter.set_postfix(W=to_move_wins, L=to_move_losses, D=draws)

    elapsed = time.time() - t0
    total = max(to_move_wins + to_move_losses + draws, 1)

    print(f"\n\n{'='*50}")
    print(f"  {bot} self-play — {len(positions)} positions × {num_rollouts} rollouts"
          f"  ({total_games} games in {elapsed:.1f}s)")
    print(f"{'='*50}")
    print(f"  {'To-move wins':>15s}: {to_move_wins:3d} ({100*to_move_wins/total:.0f}%)")
    print(f"  {'To-move losses':>15s}: {to_move_losses:3d} ({100*to_move_losses/total:.0f}%)")
    print(f"  {'Draws':>15s}: {draws:3d} ({100*draws/total:.0f}%)")
    win_rate = (to_move_wins + 0.5 * draws) / total
    print(f"\n  To-move win rate: {100*win_rate:.1f}%")
    print()

    if depths:
        total_moves = sum(depths.values())
        avg = sum(d * c for d, c in depths.items()) / total_moves
        lo, hi = min(depths), max(depths)
        print(f"  Search depth: avg {avg:.1f}, range [{lo}-{hi}]")
        buckets = sorted(depths.items())
        dist = "  ".join(f"d{d}:{c}" for d, c in buckets)
        print(f"    {dist}")

    if total_time[1] > 0:
        avg_ms = 1000 * total_time[0] / total_time[1]
        print(f"  Avg move time: {avg_ms:.0f}ms ({total_time[1]} moves)")

    if game_lengths:
        avg_len = sum(game_lengths) / len(game_lengths)
        lo_len, hi_len = min(game_lengths), max(game_lengths)
        print(f"\n  Game length: avg {avg_len:.1f} moves, range [{lo_len}-{hi_len}]")

    if aborted:
        print(f"\n  {aborted} games aborted (time limit exceeded)")

    print(f"{'='*50}")

    return to_move_wins, to_move_losses, draws


def generate_random_positions(num_positions, num_random_moves=10, win_length=6, seed=42):
    """Generate diverse starting positions by playing random moves.

    Each position is produced by a RandomBot making num_random_moves moves
    from an empty board.  Returns a list of (board_dict, current_player) tuples.
    """
    rng = random.Random(seed)
    rbot = RandomBot()
    positions = []
    for _ in range(num_positions):
        game = HexGame(win_length=win_length)
        for _ in range(num_random_moves):
            if game.game_over:
                break
            move = rbot.get_move(game)
            game.make_move(*move)
        if not game.game_over:
            positions.append((dict(game.board), game.current_player))
    return positions


def _throughput_one(args):
    """Worker: run get_move on a single position and report stats."""
    bot, win_length, position, time_limit = args
    bot.time_limit = time_limit
    board_dict, current_player = position
    game = HexGame(win_length=win_length)
    game.board = dict(board_dict)
    game.current_player = current_player
    game.move_count = len(board_dict)
    game.moves_left_in_turn = 2

    t0 = time.time()
    bot.get_move(game)
    elapsed = time.time() - t0
    return bot._nodes, bot.last_depth, elapsed


def throughput_bench(bots, positions, time_limit=0.1, win_length=6, use_tqdm=True):
    """Benchmark node throughput for one or more bots on shared positions.

    Each bot evaluates every position once (single process, sequential)
    so timing is not affected by multiprocessing overhead.
    """
    print(f"\n{'='*55}")
    print(f"  Throughput benchmark — {len(positions)} positions, "
          f"{time_limit:.2f}s per move")
    print(f"{'='*55}")

    for bot in bots:
        bot.time_limit = time_limit
        total_nodes = 0
        total_time = 0.0
        depths = defaultdict(int)

        items = enumerate(positions)
        if use_tqdm:
            items = tqdm(items, total=len(positions),
                         desc=str(bot), unit="pos")
        for _i, (board_dict, current_player) in items:
            game = HexGame(win_length=win_length)
            game.board = dict(board_dict)
            game.current_player = current_player
            game.move_count = len(board_dict)
            game.moves_left_in_turn = 2

            t0 = time.time()
            bot.get_move(game)
            elapsed = time.time() - t0

            total_nodes += bot._nodes
            total_time += elapsed
            depths[bot.last_depth] += 1

        nps = total_nodes / total_time if total_time > 0 else 0
        avg_depth = (sum(d * c for d, c in depths.items())
                     / max(sum(depths.values()), 1))
        avg_ms = 1000 * total_time / max(len(positions), 1)

        print(f"\n  {str(bot):>15s}:")
        print(f"    {total_nodes:>12,} nodes in {total_time:.2f}s "
              f"= {nps:,.0f} nodes/sec")
        print(f"    avg depth {avg_depth:.1f}, avg time {avg_ms:.0f}ms/move")
        buckets = sorted(depths.items())
        dist = "  ".join(f"d{d}:{c}" for d, c in buckets)
        print(f"    {dist}")

    print(f"{'='*55}\n")


class NamedBotWrapper:
    """Wraps a MinimaxBot instance with a custom display name."""
    def __init__(self, bot, name):
        object.__setattr__(self, '_bot', bot)
        object.__setattr__(self, '_name', name)
    def __str__(self):
        return self._name
    def __getattr__(self, attr):
        return getattr(object.__getattribute__(self, '_bot'), attr)
    def __setattr__(self, attr, value):
        setattr(object.__getattribute__(self, '_bot'), attr, value)
    def get_move(self, game):
        return object.__getattribute__(self, '_bot').get_move(game)


def load_bot(module_name, time_limit=0.1):
    """Load MinimaxBot from a module name (without .py extension)."""
    mod = importlib.import_module(module_name)
    bot = mod.MinimaxBot(time_limit=time_limit)
    return NamedBotWrapper(bot, module_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate two AI bots against each other.")
    parser.add_argument("bot_a", nargs="?", default="ai",
                        help="Module name for bot A (default: ai)")
    parser.add_argument("bot_b", nargs="?", default="og_ai",
                        help="Module name for bot B (default: og_ai)")
    parser.add_argument("-n", "--num-games", type=int, default=20,
                        help="Number of games to play (default: 20)")
    parser.add_argument("--no-tqdm", action="store_true",
                        help="Disable progress bar")
    parser.add_argument("--pattern-a", type=str, default=None,
                        help="Pattern values JSON for bot A (forces ai_tuned)")
    parser.add_argument("--pattern-b", type=str, default=None,
                        help="Pattern values JSON for bot B (forces ai_tuned)")
    default_positions = os.path.join(
        os.path.dirname(__file__), "learned_eval", "data", "positions_human_labelled.pkl")
    parser.add_argument("--positions", type=str, default=default_positions,
                        help="Pickle file of seed positions (default: human game positions)")
    parser.add_argument("--no-positions", action="store_true",
                        help="Disable position seeding (start from empty board)")
    parser.add_argument("--random-positions", action="store_true",
                        help="Generate starting positions via random playout instead of human games")
    parser.add_argument("--random-moves", type=int, default=10,
                        help="Number of random moves per position (default: 10)")
    parser.add_argument("--self-play", action="store_true",
                        help="Self-play position evaluation (one bot, report to-move win rate)")
    parser.add_argument("--throughput", action="store_true",
                        help="Run throughput benchmark (nodes/sec) instead of games")
    parser.add_argument("--throughput-positions", type=int, default=50,
                        help="Number of positions for throughput bench (default: 50)")
    parsed = parser.parse_args()

    if parsed.pattern_a:
        from ai import MinimaxBot as TunedBot
        a = NamedBotWrapper(TunedBot(pattern_path=parsed.pattern_a), parsed.pattern_a)
    else:
        a = load_bot(parsed.bot_a, time_limit=0.1)
    if parsed.pattern_b:
        from ai import MinimaxBot as TunedBot
        b = NamedBotWrapper(TunedBot(pattern_path=parsed.pattern_b), parsed.pattern_b)
    else:
        b = load_bot(parsed.bot_b, time_limit=0.1)

    if parsed.throughput:
        positions = generate_random_positions(
            parsed.throughput_positions, num_random_moves=parsed.random_moves)
        throughput_bench([a, b], positions, use_tqdm=not parsed.no_tqdm)
        sys.exit(0)

    positions = None
    if parsed.random_positions:
        positions = generate_random_positions(
            parsed.num_games, num_random_moves=parsed.random_moves)
    elif not parsed.no_positions and parsed.positions:
        positions = _load_positions(parsed.positions, num_games=parsed.num_games)

    if parsed.self_play:
        if positions is None:
            print("Error: --self-play requires positions (don't use --no-positions)")
            sys.exit(1)
        evaluate_positions(a, positions, num_rollouts=parsed.num_games,
                           use_tqdm=not parsed.no_tqdm)
    else:
        evaluate(a, b, num_games=parsed.num_games, positions=positions,
                 use_tqdm=not parsed.no_tqdm)
