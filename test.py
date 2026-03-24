"""Measure which rank-pairs are SEARCHED vs CHOSEN at every search depth.

Tracks three things per depth:
  - 'generated': all pairs that were in the turn list (after threat filter)
  - 'searched':  pairs that were actually evaluated (not pruned by alpha-beta)
  - 'chosen':    the single best pair selected

This reveals how much work alpha-beta is actually saving.
"""

import os
import sys
import math
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations
from multiprocessing import Pool

from game import HexGame, Player
from ai import (MinimaxBot, _ROOT_CANDIDATE_CAP, _CANDIDATE_CAP,
                _DELTA_WEIGHT, _WIN_SCORE, _MAX_QDEPTH, _INNER_PAIRS,
                _EXACT, _LOWER, _UPPER, TimeUp)

NUM_GAMES = 100
MAX_MOVES = 200
TIME_LIMIT = 0.1


class InstrumentedBot(MinimaxBot):
    """MinimaxBot that records generated/searched/chosen rank-pairs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # category -> depth_label -> { (rank_lo, rank_hi): count }
        # categories: 'generated', 'searched', 'chosen'
        self.rank_data = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self._search_depth = 0

    def _generate_turns(self, game):
        candidates = list(self._cand_set)
        if len(candidates) >= 2:
            is_a = game.current_player == Player.A
            maximizing = game.current_player == self._player
            candidates.sort(
                key=lambda c: self._move_delta(c[0], c[1], is_a),
                reverse=maximizing)
            self._last_ranked_root = list(candidates)
        else:
            self._last_ranked_root = list(candidates)
        return super()._generate_turns(game)

    def _search_root(self, game, turns, depth):
        self._search_depth = depth

        # Record generated and searched at root
        ranked = getattr(self, '_last_ranked_root', None)
        if ranked and len(ranked) >= 2:
            rank_map = {c: i for i, c in enumerate(ranked)}

            # All generated turns
            for turn in turns:
                r1 = rank_map.get(turn[0], -1)
                r2 = rank_map.get(turn[1], -1)
                if r1 >= 0 and r2 >= 0:
                    lo, hi = min(r1, r2), max(r1, r2)
                    self.rank_data['generated']['root'][(lo, hi)] += 1

        result, scores = super()._search_root(game, turns, depth)

        # Root searches all turns (no pruning at root), so searched = generated
        if ranked and len(ranked) >= 2:
            for turn in turns:
                r1 = rank_map.get(turn[0], -1)
                r2 = rank_map.get(turn[1], -1)
                if r1 >= 0 and r2 >= 0:
                    lo, hi = min(r1, r2), max(r1, r2)
                    self.rank_data['searched']['root'][(lo, hi)] += 1

            if result and len(result) == 2:
                r1 = rank_map.get(result[0], -1)
                r2 = rank_map.get(result[1], -1)
                if r1 >= 0 and r2 >= 0:
                    lo, hi = min(r1, r2), max(r1, r2)
                    self.rank_data['chosen']['root'][(lo, hi)] += 1

        return result, scores

    def _minimax(self, game, depth, alpha, beta):
        self._check_time()

        if game.game_over:
            if game.winner == self._player:
                return _WIN_SCORE
            elif game.winner != Player.NONE:
                return -_WIN_SCORE
            return 0

        tt_key = self._tt_key(game)
        tt_entry = self._tt.get(tt_key)
        tt_move = None
        if tt_entry:
            tt_depth, tt_score, tt_flag, tt_move = tt_entry
            if tt_depth >= depth:
                if tt_flag == _EXACT:
                    return tt_score
                elif tt_flag == _LOWER:
                    alpha = max(alpha, tt_score)
                elif tt_flag == _UPPER:
                    beta = min(beta, tt_score)
                if alpha >= beta:
                    return tt_score

        if depth == 0:
            score = self._quiescence(game, alpha, beta, _MAX_QDEPTH)
            self._tt[tt_key] = (0, score, _EXACT, None)
            return score

        win_turn = self._find_instant_win(game, game.current_player)
        if win_turn:
            undo_info = self._make_turn(game, win_turn)
            score = _WIN_SCORE if game.winner == self._player else -_WIN_SCORE
            self._undo_turn(game, undo_info)
            self._tt[tt_key] = (depth, score, _EXACT, win_turn)
            return score

        opponent = Player.B if game.current_player == Player.A else Player.A
        opp_win = self._find_instant_win(game, opponent)
        if opp_win:
            from ai import _DIR_VECTORS, _WIN_LENGTH
            p_idx = 0 if opponent == Player.A else 1
            o_idx = 1 - p_idx
            must_hit = []
            board = game.board
            hot = self._hot_a if opponent == Player.A else self._hot_b
            wc = self._wc
            for wkey in hot:
                counts = wc[wkey]
                if counts[p_idx] >= _WIN_LENGTH - 2 and counts[o_idx] == 0:
                    d_idx, sq, sr = wkey
                    dq, dr = _DIR_VECTORS[d_idx]
                    empties = frozenset(
                        (sq + j * dq, sr + j * dr)
                        for j in range(_WIN_LENGTH)
                        if (sq + j * dq, sr + j * dr) not in board
                    )
                    must_hit.append(empties)
            if len(must_hit) > 1:
                can_block = False
                all_cells = set()
                for s in must_hit:
                    all_cells |= s
                for c1 in all_cells:
                    for c2 in all_cells:
                        if all(c1 in w or c2 in w for w in must_hit):
                            can_block = True
                            break
                    if can_block:
                        break
                if not can_block:
                    score = -_WIN_SCORE if opponent != self._player else _WIN_SCORE
                    self._tt[tt_key] = (depth, score, _EXACT, None)
                    return score

        orig_alpha = alpha
        orig_beta = beta
        maximizing = game.current_player == self._player

        candidates = list(self._cand_set)
        ranked_candidates = None
        if len(candidates) < 2:
            if not candidates:
                score = self._eval_score
                self._tt[tt_key] = (depth, score, _EXACT, None)
                return score
            c = candidates[0]
            turns = [(c, c)]
        else:
            is_a = game.current_player == Player.A
            history = self._history
            move_delta = self._move_delta

            delta_sign = _DELTA_WEIGHT if maximizing else -_DELTA_WEIGHT
            candidates.sort(
                key=lambda c: history.get(c, 0) + move_delta(c[0], c[1], is_a) * delta_sign,
                reverse=True)
            candidates = candidates[:_CANDIDATE_CAP]
            ranked_candidates = list(candidates)

            n = len(candidates)
            turns = [(candidates[i], candidates[j]) for i, j in _INNER_PAIRS
                     if i < n and j < n]
            turns = self._filter_turns_by_threats(game, turns)

        if not turns:
            score = self._eval_score
            self._tt[tt_key] = (depth, score, _EXACT, None)
            return score

        if tt_move is not None:
            try:
                idx = turns.index(tt_move)
                turns[0], turns[idx] = turns[idx], turns[0]
            except ValueError:
                pass

        # --- Record generated pairs ---
        depth_label = f'd{self._search_depth - depth}'
        rank_map = None
        if ranked_candidates and len(ranked_candidates) >= 2:
            rank_map = {c: i for i, c in enumerate(ranked_candidates)}
            for turn in turns:
                r1 = rank_map.get(turn[0], -1)
                r2 = rank_map.get(turn[1], -1)
                if r1 >= 0 and r2 >= 0:
                    lo, hi = min(r1, r2), max(r1, r2)
                    self.rank_data['generated'][depth_label][(lo, hi)] += 1

        best_move = None
        searched_turns = []

        if maximizing:
            value = -math.inf
            for turn in turns:
                searched_turns.append(turn)
                undo_info = self._make_turn(game, turn)
                if game.game_over:
                    child_val = _WIN_SCORE if game.winner == self._player else -_WIN_SCORE
                else:
                    child_val = self._minimax(game, depth - 1, alpha, beta)
                self._undo_turn(game, undo_info)
                if child_val > value:
                    value = child_val
                    best_move = turn
                alpha = max(alpha, value)
                if alpha >= beta:
                    history = self._history
                    history[turn[0]] = history.get(turn[0], 0) + depth * depth
                    history[turn[1]] = history.get(turn[1], 0) + depth * depth
                    break
        else:
            value = math.inf
            for turn in turns:
                searched_turns.append(turn)
                undo_info = self._make_turn(game, turn)
                if game.game_over:
                    child_val = _WIN_SCORE if game.winner == self._player else -_WIN_SCORE
                else:
                    child_val = self._minimax(game, depth - 1, alpha, beta)
                self._undo_turn(game, undo_info)
                if child_val < value:
                    value = child_val
                    best_move = turn
                beta = min(beta, value)
                if alpha >= beta:
                    history = self._history
                    history[turn[0]] = history.get(turn[0], 0) + depth * depth
                    history[turn[1]] = history.get(turn[1], 0) + depth * depth
                    break

        # --- Record searched and chosen ---
        if rank_map is not None:
            for turn in searched_turns:
                r1 = rank_map.get(turn[0], -1)
                r2 = rank_map.get(turn[1], -1)
                if r1 >= 0 and r2 >= 0:
                    lo, hi = min(r1, r2), max(r1, r2)
                    self.rank_data['searched'][depth_label][(lo, hi)] += 1

            if best_move is not None:
                r1 = rank_map.get(best_move[0], -1)
                r2 = rank_map.get(best_move[1], -1)
                if r1 >= 0 and r2 >= 0:
                    lo, hi = min(r1, r2), max(r1, r2)
                    self.rank_data['chosen'][depth_label][(lo, hi)] += 1

        if value <= orig_alpha:
            flag = _UPPER
        elif value >= orig_beta:
            flag = _LOWER
        else:
            flag = _EXACT

        self._tt[tt_key] = (depth, value, flag, best_move)
        return value


def _play_one(args):
    game_idx, time_limit = args
    bot_a = InstrumentedBot(time_limit=time_limit)
    bot_b = InstrumentedBot(time_limit=time_limit)

    game = HexGame()
    bots = {Player.A: bot_a, Player.B: bot_b}
    move_count = 0

    while not game.game_over and move_count < MAX_MOVES:
        player = game.current_player
        bot = bots[player]
        result = bot.get_move(game)
        moves_list = result if bot.pair_moves else [result]
        for q, r in moves_list:
            if game.game_over:
                break
            game.make_move(q, r)
            move_count += 1

    winner = game.winner.name if game.winner != Player.NONE else "DRAW"

    # Merge both bots' rank_data
    merged = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for bot in (bot_a, bot_b):
        for cat, depth_data in bot.rank_data.items():
            for label, counts in depth_data.items():
                for pair, count in counts.items():
                    merged[cat][label][pair] += count

    serializable = {}
    for cat, depth_data in merged.items():
        serializable[cat] = {label: dict(counts) for label, counts in depth_data.items()}
    return serializable, move_count, winner


def main():
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_GAMES
    tl = float(sys.argv[2]) if len(sys.argv) > 2 else TIME_LIMIT

    workers = os.cpu_count() or 1
    args = [(i, tl) for i in range(num_games)]

    print(f"Running {num_games} self-play games, time_limit={tl}s, "
          f"CAP={_ROOT_CANDIDATE_CAP} (root) / {_CANDIDATE_CAP} (inner), "
          f"INNER_PAIRS={len(_INNER_PAIRS)}, {workers} workers")

    t0 = time.time()
    # cat -> label -> { (lo, hi): count }
    all_data = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    game_lengths = []
    winners = defaultdict(int)

    with Pool(workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_play_one, args)):
            data, move_count, winner = result
            for cat, depth_data in data.items():
                for label, counts in depth_data.items():
                    for pair, count in counts.items():
                        all_data[cat][label][tuple(pair)] += count
            game_lengths.append(move_count)
            winners[winner] += 1
            if (i + 1) % 10 == 0 or i + 1 == num_games:
                elapsed = time.time() - t0
                print(f"  {i+1}/{num_games} games ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Results: {dict(winners)}")
    print(f"Avg game length: {sum(game_lengths)/len(game_lengths):.1f} moves")

    # Collect all depth labels across categories
    all_labels = set()
    for cat in all_data:
        all_labels |= set(all_data[cat].keys())
    labels = sorted(all_labels, key=lambda x: -1 if x == 'root' else int(x[1:]))

    # Print summary
    categories = ['generated', 'searched', 'chosen']
    for label in labels:
        parts = []
        for cat in categories:
            total = sum(all_data[cat].get(label, {}).values())
            parts.append(f"{cat}={total}")
        print(f"  {label}: {', '.join(parts)}")
        # Pruning rate
        gen = sum(all_data['generated'].get(label, {}).values())
        srch = sum(all_data['searched'].get(label, {}).values())
        if gen > 0:
            print(f"    pruning rate: {100*(1-srch/gen):.1f}%")

    # --- Determine global max rank ---
    max_rank = 0
    for cat in all_data:
        for label, counts in all_data[cat].items():
            for (r1, r2) in counts:
                max_rank = max(max_rank, r1, r2)
    max_rank = min(max_rank + 1, max(_ROOT_CANDIDATE_CAP, _CANDIDATE_CAP) + 2)

    # --- Plot: 3 columns (generated/searched/chosen) x N depth rows ---
    n_labels = len(labels)
    fig, axes = plt.subplots(n_labels, 3, figsize=(20, 5 * n_labels))
    if n_labels == 1:
        axes = [axes]

    for row, label in enumerate(labels):
        for col, cat in enumerate(categories):
            counts = all_data[cat].get(label, {})
            total = sum(counts.values())

            heatmap = np.zeros((max_rank, max_rank), dtype=float)
            for (r1, r2), count in counts.items():
                if r1 < max_rank and r2 < max_rank:
                    heatmap[r1, r2] += count
                    if r1 != r2:
                        heatmap[r2, r1] += count

            if total > 0:
                heatmap_pct = heatmap / total * 100
            else:
                heatmap_pct = heatmap

            ax = axes[row][col]
            im = ax.imshow(heatmap_pct, cmap='YlOrRd', origin='upper', aspect='equal')
            gen_total = sum(all_data['generated'].get(label, {}).values())
            srch_total = sum(all_data['searched'].get(label, {}).values())
            if cat == 'searched' and gen_total > 0:
                subtitle = f' (pruned {100*(1-srch_total/gen_total):.0f}%)'
            else:
                subtitle = ''
            ax.set_title(f'{label} {cat} — n={total}{subtitle}')
            ax.set_xlabel('Move rank')
            ax.set_ylabel('Move rank')
            ax.set_xticks(range(max_rank))
            ax.set_yticks(range(max_rank))
            plt.colorbar(im, ax=ax, label='%', shrink=0.8)

            for i in range(max_rank):
                for j in range(max_rank):
                    val = heatmap_pct[i, j]
                    if val > 1.0:
                        ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                                fontsize=5,
                                color='black' if val < heatmap_pct.max() * 0.6 else 'white')

    plt.tight_layout()
    plt.savefig('rank_heatmap.png', dpi=150)
    print(f"\nSaved rank_heatmap.png")

    # --- Print searched-but-not-chosen waste per depth ---
    print("\n--- Waste analysis: searched but never chosen ---")
    for label in labels:
        searched = all_data['searched'].get(label, {})
        chosen = all_data['chosen'].get(label, {})
        total_searched = sum(searched.values())
        if total_searched == 0:
            continue

        # For each rank pair, compute search_count - chosen_count
        waste = {}
        for pair, s_count in searched.items():
            c_count = chosen.get(pair, 0)
            waste[pair] = s_count - c_count

        total_waste = sum(waste.values())
        print(f"\n  {label}: {total_waste}/{total_searched} searches were waste "
              f"({100*total_waste/total_searched:.1f}%)")

        # Top wasted pairs
        sorted_waste = sorted(waste.items(), key=lambda x: -x[1])
        print(f"  Top 10 wasted pairs:")
        for (r1, r2), w in sorted_waste[:10]:
            s = searched[(r1, r2)]
            c = chosen.get((r1, r2), 0)
            print(f"    ({r1:2d},{r2:2d}): searched {s:5d}, chosen {c:4d}, "
                  f"waste {w:5d} ({100*w/total_searched:.1f}%)")


if __name__ == '__main__':
    main()
