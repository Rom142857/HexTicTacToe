"""Time profile of ai_cpp.cpp internals via the profiled build (ai_cpp_prof).

Instruments 6 leaf functions (_make, _undo, _find_instant_win,
_find_threat_cells, _move_delta, _filter_turns_by_threats) with
nanosecond-resolution timers, plus call counts and TT / alpha-beta stats.

Usage:
    python setup.py build_ext --inplace   # build ai_cpp_prof first
    python test_profile.py [--moves N] [--time-limit T]
"""

import time
import argparse
from game import HexGame
from ai_cpp_prof import MinimaxBot


TIMED_FUNCS = [
    ("_make",                "make_ns",         "make_calls"),
    ("_undo",                "undo_ns",         "undo_calls"),
    ("_find_instant_win",    "find_win_ns",     "find_win_calls"),
    ("_find_threat_cells",   "find_threats_ns",  "find_threats_calls"),
    ("_move_delta",          "move_delta_ns",   "move_delta_calls"),
    ("_filter_turns_by_thr", "filter_turns_ns", "filter_turns_calls"),
]


def play_profiled_game(time_limit=0.5, num_moves=15):
    game = HexGame(win_length=6)
    bot = MinimaxBot(time_limit=time_limit)
    all_data = []

    for move_num in range(num_moves):
        if game.game_over:
            break

        t0 = time.perf_counter_ns()
        result = bot.get_move(game)
        wall_ns = time.perf_counter_ns() - t0

        prof = bot.get_profile()
        prof["move_num"] = move_num
        prof["wall_ns"] = wall_ns
        prof["nodes"] = bot._nodes
        prof["depth"] = bot.last_depth
        prof["ebf"] = bot.last_ebf
        prof["score"] = bot.last_score
        prof["board_size"] = len(game.board)
        all_data.append(prof)

        moves = result if isinstance(result, list) else [result]
        for q, r in moves:
            if game.game_over:
                break
            game.make_move(q, r)

    return all_data


def print_report(all_data):
    W = 78
    print("=" * W)
    print("  ai_cpp.cpp TIME PROFILE")
    print("=" * W)

    total_wall = sum(d["wall_ns"] for d in all_data)
    total_nodes = sum(d["nodes"] for d in all_data)

    # ── Per-move summary ──
    print(f"\n  PER-MOVE SUMMARY ({len(all_data)} moves)")
    print("  " + "-" * (W - 2))
    print(f"  {'Move':>4} {'Board':>5} {'Depth':>5} {'EBF':>5} "
          f"{'Nodes':>9} {'Time ms':>8} {'kN/s':>8} {'Score':>10}")
    for d in all_data:
        wall_ms = d["wall_ns"] / 1e6
        kn_s = d["nodes"] / (d["wall_ns"] / 1e9) / 1000 if d["wall_ns"] > 0 else 0
        print(f"  {d['move_num']:4d} {d['board_size']:5d} {d['depth']:5d} "
              f"{d['ebf']:5.1f} {d['nodes']:9,d} {wall_ms:8.1f} "
              f"{kn_s:8.1f} {d['score']:10.0f}")

    # ── Time breakdown ──
    print(f"\n  TIME BREAKDOWN (wall = {total_wall/1e6:.1f} ms)")
    print("  " + "-" * (W - 2))
    total_timed = 0
    rows = []
    for label, ns_key, calls_key in TIMED_FUNCS:
        ns = sum(d[ns_key] for d in all_data)
        calls = sum(d[calls_key] for d in all_data)
        total_timed += ns
        rows.append((label, ns, calls))

    overhead = total_wall - total_timed
    print(f"  {'Function':>22} {'Total ms':>10} {'%':>6} "
          f"{'Calls':>12} {'ns/call':>8}")
    for label, ns, calls in rows:
        pct = 100 * ns / total_wall if total_wall > 0 else 0
        avg = ns / calls if calls > 0 else 0
        print(f"  {label:>22} {ns/1e6:10.1f} {pct:5.1f}% "
              f"{calls:12,d} {avg:8.0f}")
    pct_oh = 100 * overhead / total_wall if total_wall > 0 else 0
    print(f"  {'overhead (TT/sort/...)':>22} {overhead/1e6:10.1f} {pct_oh:5.1f}%")
    print(f"  {'TOTAL':>22} {total_wall/1e6:10.1f}")

    # ── Transposition table ──
    print(f"\n  TRANSPOSITION TABLE")
    print("  " + "-" * (W - 2))
    tp = sum(d["tt_probes"] for d in all_data)
    th = sum(d["tt_hits"] for d in all_data)
    te = sum(d["tt_exact_cutoffs"] for d in all_data)
    tb = sum(d["tt_bound_cutoffs"] for d in all_data)
    ts = sum(d["tt_stores"] for d in all_data)
    print(f"  Probes:          {tp:>12,d}")
    if tp:
        print(f"  Hits:            {th:>12,d}  ({100*th/tp:.1f}%)")
    else:
        print(f"  Hits:            {th:>12,d}")
    print(f"  Exact cutoffs:   {te:>12,d}")
    print(f"  Bound cutoffs:   {tb:>12,d}")
    print(f"  Stores:          {ts:>12,d}")

    # ── Alpha-beta ──
    print(f"\n  ALPHA-BETA PRUNING")
    print("  " + "-" * (W - 2))
    mm = sum(d["minimax_calls"] for d in all_data)
    qs = sum(d["qsearch_calls"] for d in all_data)
    ab_cut = sum(d["ab_cutoffs"] for d in all_data)
    ab_int = sum(d["ab_interior"] for d in all_data)
    print(f"  Minimax calls:   {mm:>12,d}")
    print(f"  Quiescence calls:{qs:>12,d}")
    print(f"  Interior nodes:  {ab_int:>12,d}")
    print(f"  AB cutoffs:      {ab_cut:>12,d}")
    if ab_int:
        print(f"  Cutoff rate:     {100*ab_cut/ab_int:>11.1f}%")

    # ── Throughput ──
    print(f"\n  THROUGHPUT")
    print("  " + "-" * (W - 2))
    total_s = total_wall / 1e9
    print(f"  Total nodes:     {total_nodes:>12,d}")
    print(f"  Total time:      {total_s:>12.3f}s")
    if total_s > 0:
        print(f"  Nodes/sec:       {total_nodes/total_s:>12,.0f}")
    make_c = sum(d["make_calls"] for d in all_data)
    undo_c = sum(d["undo_calls"] for d in all_data)
    if total_s > 0:
        print(f"  Make+undo/sec:   {(make_c+undo_c)/total_s:>12,.0f}")

    # ── Per-move time breakdown heatmap ──
    print(f"\n  PER-MOVE BREAKDOWN (% of wall time per function)")
    print("  " + "-" * (W - 2))
    hdr = f"  {'Move':>4}"
    short_labels = ["make", "undo", "fwin", "fthr", "mdlt", "filt", "rest"]
    for sl in short_labels:
        hdr += f" {sl:>6}"
    print(hdr)
    for d in all_data:
        wall = d["wall_ns"]
        if wall == 0:
            continue
        row = f"  {d['move_num']:4d}"
        timed_sum = 0
        for _, ns_key, _ in TIMED_FUNCS:
            pct = 100 * d[ns_key] / wall
            timed_sum += d[ns_key]
            row += f" {pct:5.1f}%"
        rest_pct = 100 * (wall - timed_sum) / wall
        row += f" {rest_pct:5.1f}%"
        print(row)

    print()
    print("=" * W)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time profile of ai_cpp.cpp")
    parser.add_argument("--moves", type=int, default=12)
    parser.add_argument("--time-limit", type=float, default=0.5)
    args = parser.parse_args()

    print(f"Profiling {args.moves} moves with {args.time_limit}s/move...\n")
    data = play_profiled_game(time_limit=args.time_limit, num_moves=args.moves)
    print_report(data)
