"""Find forced-win setups in human game positions for puzzle generation.

Two-phase approach:
  1. Fast pre-filter (quiescence-level): skip positions where either player
     already has an instant win, and skip positions with no near-threats
     (need 2+ unblocked windows with 3+ stones for a dual-threat setup).
  2. Minimax search on remaining candidates.  The quiescence extension at
     the leaves detects the forced win once the setup move creates the
     unblockable dual threat.

The principal variation is extracted directly from the transposition table
after the search — this gives the exact line the search proved, with the
final winning move found via _find_instant_win at the leaf.

Usage:
    python find_forced_wins.py [--time-limit 0.1] [--workers 8]
"""

import argparse
import json
import os
import pickle
import time
from collections import defaultdict
from multiprocessing import Pool

from ai import (
    MinimaxBot,
    _DIR_VECTORS,
    _NEIGHBOR_OFFSETS_2,
    _WIN_LENGTH,
    _WIN_SCORE,
    _WINDOW_OFFSETS,
    LINE_SCORES,
    _zobrist,
    _zobrist_rng,
)
from game import HexGame, Player


# ---------------------------------------------------------------------------
# Bot state initialisation (extracted from MinimaxBot.get_move)
# ---------------------------------------------------------------------------

def _init_bot(bot, game):
    """Initialise bot incremental state for *game* without running a search."""
    bot._player = game.current_player
    bot._nodes = 0
    bot._deadline = time.time() + 3600
    bot._tt = {}
    bot._history = {}
    bot._rc_stack = []

    bot._hash = 0
    for (q, r), p in game.board.items():
        zkey = (q, r, p)
        v = _zobrist.get(zkey)
        if v is None:
            v = _zobrist_rng.getrandbits(64)
            _zobrist[zkey] = v
        bot._hash ^= v

    sz = _WIN_LENGTH + 1
    bot._score_table = [[0] * sz for _ in range(sz)]
    for a in range(sz):
        for b in range(sz):
            if bot._player == Player.A:
                my, opp = a, b
            else:
                my, opp = b, a
            if my > 0 and opp == 0:
                bot._score_table[a][b] = LINE_SCORES[my]
            elif opp > 0 and my == 0:
                bot._score_table[a][b] = -LINE_SCORES[opp]

    bot._wc = {}
    board = game.board
    seen = set()
    for (q, r) in board:
        for d_idx, oq, or_ in _WINDOW_OFFSETS:
            wkey = (d_idx, q - oq, r - or_)
            if wkey in seen:
                continue
            seen.add(wkey)
            dq, dr = _DIR_VECTORS[d_idx]
            sq, sr = wkey[1], wkey[2]
            a_count = b_count = 0
            for j in range(_WIN_LENGTH):
                cp = board.get((sq + j * dq, sr + j * dr))
                if cp == Player.A:
                    a_count += 1
                elif cp == Player.B:
                    b_count += 1
            if a_count > 0 or b_count > 0:
                bot._wc[wkey] = [a_count, b_count]

    bot._eval_score = 0
    bot._hot_a = set()
    bot._hot_b = set()
    st = bot._score_table
    for wkey, counts in bot._wc.items():
        bot._eval_score += st[counts[0]][counts[1]]
        if counts[0] >= 4:
            bot._hot_a.add(wkey)
        if counts[1] >= 4:
            bot._hot_b.add(wkey)

    bot._cand_refcount = {}
    for (q, r) in board:
        for dq, dr in _NEIGHBOR_OFFSETS_2:
            nb = (q + dq, r + dr)
            if nb not in board:
                bot._cand_refcount[nb] = bot._cand_refcount.get(nb, 0) + 1
    bot._cand_set = set(bot._cand_refcount)


# ---------------------------------------------------------------------------
# Pre-filter helpers
# ---------------------------------------------------------------------------

def _has_instant_win(bot, game):
    """True if either player has an instant win in this position."""
    opp = Player.B if game.current_player == Player.A else Player.A
    return (bot._find_instant_win(game, game.current_player) is not None
            or bot._find_instant_win(game, opp) is not None)


def _has_near_threats(bot):
    """True if at least one player has 2+ unblocked windows with 3+ stones."""
    a3 = b3 = 0
    for counts in bot._wc.values():
        if counts[0] >= 3 and counts[1] == 0:
            a3 += 1
        if counts[1] >= 3 and counts[0] == 0:
            b3 += 1
    return a3 >= 2 or b3 >= 2


# ---------------------------------------------------------------------------
# PV extraction from TT
# ---------------------------------------------------------------------------

def _extract_pv(bot, game):
    """Walk the transposition table to extract the principal variation.

    At nodes where the TT has no best_move (e.g. minimiser nodes where all
    moves lose), we generate the defender's best response via threat turns
    and then continue to the next node where the winner finishes with an
    instant win.
    """
    from itertools import combinations

    pv = []
    undo_stack = []
    seen = set()

    while not game.game_over:
        tt_key = bot._tt_key(game)
        if tt_key in seen:
            break
        seen.add(tt_key)

        entry = bot._tt.get(tt_key)

        # Have a TT entry with a concrete best move — use it directly
        if entry is not None and entry[3] is not None:
            depth, score, flag, best_turn = entry
            if abs(score) < _WIN_SCORE:
                break
            player = game.current_player
            undo_info = bot._make_turn(game, best_turn)
            actual = [cell for cell, _, _ in undo_info]
            pv.append({"player": player, "moves": actual})
            undo_stack.append(undo_info)
            continue

        # No best_move — either a depth-0 leaf or all moves lose.
        # Check if current player has an instant win.
        player = game.current_player
        win_turn = bot._find_instant_win(game, player)
        if win_turn:
            undo_info = bot._make_turn(game, win_turn)
            actual = [cell for cell, _, _ in undo_info]
            pv.append({"player": player, "moves": actual})
            undo_stack.append(undo_info)
            break

        # Current player can't win instantly — they're the defender.
        # Check that the opponent DOES have an instant win after any
        # response (confirming we're still on the forced-win line).
        opponent = Player.B if player == Player.A else Player.A
        opp_threats = bot._find_threat_cells(game, opponent)
        my_threats = bot._find_threat_cells(game, player)
        threat_turns = bot._generate_threat_turns(game, my_threats, opp_threats)
        if not threat_turns:
            break

        # Pick the blocking response that makes the opponent work hardest:
        # prefer responses where the opponent needs 2 stones to win (not 1).
        best_response = None
        for turn in threat_turns:
            undo_info = bot._make_turn(game, turn)
            if game.game_over:
                bot._undo_turn(game, undo_info)
                continue
            opp_win = bot._find_instant_win(game, opponent)
            bot._undo_turn(game, undo_info)
            if opp_win is not None:
                best_response = turn
                # Keep searching for a "harder" defense but take first valid
                break

        if best_response is None:
            # No response leaves opponent with instant win — we've left
            # the proven forced-win line, stop here
            break

        undo_info = bot._make_turn(game, best_response)
        actual = [cell for cell, _, _ in undo_info]
        pv.append({"player": player, "moves": actual})
        undo_stack.append(undo_info)

        if game.game_over:
            break

    # Undo everything to restore original state
    for undo_info in reversed(undo_stack):
        bot._undo_turn(game, undo_info)

    return pv


# ---------------------------------------------------------------------------
# Position loading / game setup
# ---------------------------------------------------------------------------

def _load_all_positions(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    positions = []
    for i, entry in enumerate(data):
        if len(entry) == 5:
            board, player, _score, _label, uid = entry
        else:
            board, player, _score, uid = entry
        positions.append((board, player, uid, i))
    return positions


def _setup_game(board_dict, current_player):
    game = HexGame(win_length=6)
    game.board = dict(board_dict)
    game.current_player = current_player
    game.move_count = len(board_dict)
    game.moves_left_in_turn = 2
    return game


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _analyze_one(args):
    """Check one position for a forced-win setup via minimax + quiescence.

    Returns (pos_idx, result_dict_or_None) so we can always track which
    positions have been processed, even when no win is found.
    """
    board_dict, current_player, game_uid, pos_idx, time_limit = args

    game = _setup_game(board_dict, current_player)

    # Fast pre-filter
    filter_bot = MinimaxBot(time_limit=1.0)
    _init_bot(filter_bot, game)
    if _has_instant_win(filter_bot, game):
        return pos_idx, None
    if not _has_near_threats(filter_bot):
        return pos_idx, None

    # Full minimax search
    bot = MinimaxBot(time_limit=time_limit)
    bot.get_move(game)

    if abs(bot.last_score) < _WIN_SCORE:
        return pos_idx, None

    # Extract PV from TT
    pv = _extract_pv(bot, game)

    # Must be a multi-turn setup (not a 1-turn instant win)
    if len(pv) < 2:
        return pos_idx, None

    if bot.last_score > 0:
        winner = game.current_player
    else:
        winner = Player.B if game.current_player == Player.A else Player.A

    def _p(player):
        return "A" if player == Player.A else "B"

    return pos_idx, {
        "board": {f"{q},{r}": _p(p) for (q, r), p in board_dict.items()},
        "to_move": _p(current_player),
        "winner": _p(winner),
        "sequence": [
            {"player": _p(turn["player"]), "moves": turn["moves"]}
            for turn in pv
        ],
        "game_uid": game_uid,
        "num_stones": len(board_dict),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Find forced-win setups in human game positions."
    )
    parser.add_argument(
        "--time-limit", type=float, default=0.1,
        help="Minimax search time per position in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of worker processes (default: cpu_count)",
    )
    parser.add_argument(
        "--input", type=str,
        default=os.path.join(
            os.path.dirname(__file__), "learned_eval", "data", "positions_human.pkl"
        ),
    )
    parser.add_argument(
        "--output", type=str,
        default=os.path.join(os.path.dirname(__file__), "data", "forced_wins.json"),
    )
    args = parser.parse_args()

    workers = args.workers or os.cpu_count() or 1

    print(f"Loading positions from {args.input}...")
    positions = _load_all_positions(args.input)
    uids = set(uid for _, _, uid, _ in positions)
    print(f"  {len(positions)} positions from {len(uids)} games")

    # Resume support: checkpoint stores how many positions have been processed
    checkpoint_path = args.output.rsplit(".", 1)[0] + ".checkpoint"
    old_checkpoint = checkpoint_path + ".json"  # backwards compat
    start_from = 0
    results = []
    if os.path.exists(old_checkpoint):
        # Migrate old list-of-indices format
        with open(old_checkpoint) as f:
            start_from = max(json.load(f)) + 1
        os.rename(old_checkpoint, old_checkpoint + ".bak")
        with open(checkpoint_path, "w") as f:
            f.write(str(start_from))
        print(f"  Migrated old checkpoint, resuming from position {start_from}")
    elif os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            start_from = int(f.read().strip())
        print(f"  Resuming from position {start_from}")
    if os.path.exists(args.output):
        with open(args.output) as f:
            results = json.load(f)
        print(f"  Loaded {len(results)} prior wins")

    work = [
        (b, p, uid, idx, args.time_limit)
        for b, p, uid, idx in positions[start_from:]
    ]
    total_all = len(positions)
    total_new = len(work)
    print(f"  {total_new} positions to scan")

    print(f"\nScanning ({args.time_limit}s minimax, {workers} workers)...")
    t0 = time.time()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_every = 10
    last_saved = len(results)
    max_seq = max((len(r["sequence"]) for r in results), default=0)
    processed = start_from
    from tqdm import tqdm
    with Pool(workers) as pool:
        bar = tqdm(
            pool.imap(_analyze_one, work, chunksize=8),
            total=total_new, desc="Positions", unit="pos",
        )
        for pos_idx, result in bar:
            processed = pos_idx + 1
            if result is not None:
                results.append(result)
                seq_len = len(result["sequence"])
                if seq_len > max_seq:
                    max_seq = seq_len
            # Save periodically
            if len(results) - last_saved >= save_every or bar.n % 1000 == 0:
                with open(args.output, "w") as f:
                    json.dump(results, f)
                with open(checkpoint_path, "w") as f:
                    f.write(str(processed))
                last_saved = len(results)
            if processed > 0:
                win_rate = len(results) / processed
                est = int(win_rate * total_all)
                bar.set_postfix(wins=len(results), est=est, longest=max_seq)

    # Final checkpoint
    with open(checkpoint_path, "w") as f:
        f.write(str(processed))

    elapsed = time.time() - t0

    # Deduplicate by board state (prefer longer sequences)
    seen_boards = set()
    deduped = []
    for r in sorted(results, key=lambda r: len(r["sequence"]), reverse=True):
        board_key = frozenset(sorted(r["board"].items()))
        if board_key not in seen_boards:
            seen_boards.add(board_key)
            deduped.append(r)

    # Within each game, only keep the earliest forced win for each winner.
    # If the same player keeps having a forced win on consecutive positions,
    # that's the same threat escalating — only the first one is interesting.
    # A new forced win is kept if the previous one (for that game+winner)
    # had a gap (i.e. there were positions in between without a forced win).
    by_game = defaultdict(list)
    for r in deduped:
        by_game[r["game_uid"]].append(r)

    unique = []
    for uid, game_results in by_game.items():
        game_results.sort(key=lambda r: r["num_stones"])
        # Track last kept stone count per winner to detect gaps
        last_kept = {}  # winner -> num_stones of last kept result
        for r in game_results:
            w = r["winner"]
            prev = last_kept.get(w)
            if prev is None:
                # First forced win for this winner in this game — keep
                unique.append(r)
                last_kept[w] = r["num_stones"]
            elif r["num_stones"] - prev > 4:
                # Gap of more than 2 turns (4 stones) — likely a new threat
                unique.append(r)
                last_kept[w] = r["num_stones"]
            # else: same escalating threat, skip

    # Sort: fewer stones first (harder puzzles), then longer sequences
    unique.sort(key=lambda r: (r["num_stones"], -len(r["sequence"])))

    # Final save (deduplicated + sorted)
    with open(args.output, "w") as f:
        json.dump(unique, f, indent=2)

    # Stats
    a_wins = sum(1 for r in unique if r.get("winner") == "A")
    b_wins = sum(1 for r in unique if r.get("winner") == "B")
    attacker = sum(1 for r in unique if r["winner"] == r["to_move"])
    defender = len(unique) - attacker
    seq_lens = [len(r["sequence"]) for r in unique]

    print(f"\n{'='*55}")
    print(f"  Forced Win Setup Detection Complete")
    print(f"{'='*55}")
    print(f"  Positions scanned:   {processed}")
    print(f"  Forced wins (raw):   {len(results)}")
    print(f"  After board dedup:   {len(deduped)}")
    print(f"  Unique positions:    {len(unique)}")
    print(f"  Time:                {elapsed:.1f}s ({total/elapsed:.0f} pos/s)")
    print(f"\n  Player A wins: {a_wins}")
    print(f"  Player B wins: {b_wins}")
    print(f"  To-move wins:  {attacker}")
    print(f"  To-move loses: {defender}")
    if seq_lens:
        print(
            f"\n  Sequence len:  "
            f"min={min(seq_lens)}, max={max(seq_lens)}, "
            f"avg={sum(seq_lens)/len(seq_lens):.1f}"
        )
        dist = defaultdict(int)
        for s in seq_lens:
            dist[s] += 1
        print(f"  Sequence distribution:")
        for k in sorted(dist):
            print(f"    {k} turns: {dist[k]}")

    print(f"\n  Saved {len(unique)} puzzles to {args.output}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
