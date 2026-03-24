import itertools
import math
import random
import time
from collections import defaultdict
from enum import Enum
from operator import itemgetter

from bot import Bot
from game import Player


# ── Minimal Board/Cell shim (replaces the old Board module) ──

class Cell(Enum):
    X = 1
    O = 2


class Board:
    def __init__(self):
        self.moves = []
        self.turn = 0
        self.winner = None

    def reset(self):
        self.moves = []
        self.turn = 0
        self.winner = None

    def current_player(self):
        return Cell.X if self.turn % 2 == 0 else Cell.O

    def play(self, m):
        player = self.current_player()
        for coord in m:
            self.moves.append((player, coord[0], coord[1]))
        self.turn += 1
        return False

DEFAULT_DEPTH = 5
TT_SIZE = 1048576  # 2^20 fixed size for the Transposition Table
POW10 = [10**i for i in range(10)]
WIN_DIRS = [(1, 0), (0, 1), (1, -1)]
NEIGHBOR_OFFSETS = [
    (dq, dr)
    for dq in [-2, -1, 0, 1, 2]
    for dr in [-2, -1, 0, 1, 2]
    if dq != 0 or dr != 0
]


class Engine(Board):
    def __init__(self, weights, name="Hex-Engine v1.0"):
        self.weights = weights
        self.name = name
        self.zobrist_table = defaultdict(lambda: random.getrandbits(64))
        # Pre-fill Zobrist table for the center piece
        self.zobrist_table[(1, 0, 0)] = random.getrandbits(64)

        # 2. Array-Based Transposition Table (No GC Pauses)
        self.tt: list = [None] * TT_SIZE

        super().__init__()
        self._init_incremental_state()

    def reset(self):
        super().reset()
        self._init_incremental_state()

    def play(self, m):
        player = self.current_player()
        pval = player.value

        if len(m) != 2:
            raise ValueError(" You must provide exactly two moves")

        # 1. Let the parent Board class handle ALL validation, history, and turn-counters.
        is_win = super().play(m)

        m1 = (pval, m[0][0], m[0][1])
        m2 = (pval, m[1][0], m[1][1])

        self._add_move(m1)
        self._add_move(m2)

        # 3. Use the Engine's O(1) win check instead of the Board's O(N) one
        if self.x_needs[-1] > 0:
            self.winner = Cell.X
            return True
        if self.o_needs[-1] > 0:
            self.winner = Cell.O
            return True

        return is_win

    def _init_incremental_state(self):
        self.current_hash = 0
        self.empty_cells_set = set()
        self.frontier_counts = defaultdict(int)
        self.history_table = defaultdict(int)

        self.window_counts = defaultdict(lambda: [0, 0])
        self.window_cells = {}
        self.cell_windows = {}
        self.x_needs = [0, 0, 0, 0, 0, 0]
        self.o_needs = [0, 0, 0, 0, 0, 0]
        self.cell_scores = defaultdict(int)

        existing_moves = list(self.moves)
        self.board_state = {}
        for m in existing_moves:
            val = m[0].value if hasattr(m[0], "value") else m[0]
            self._add_move((val, m[1], m[2]))

    def _update_window(self, key, player, delta):
        counts = self.window_counts[key]
        x_c, o_c = counts[0], counts[1]

        try:
            cells = self.window_cells[key]
        except KeyError:
            dq, dr, sq, sr = key
            cells = tuple((sq + i * dq, sr + i * dr) for i in range(6))
            self.window_cells[key] = cells

        # 1. Incremental Cell Scores (O(1) Move Ordering)
        # Remove old contributions from cell_scores
        if x_c > 0 and o_c == 0:
            val = POW10[x_c]
            for cell in cells:
                self.cell_scores[cell] -= val
            self.x_needs[x_c - 1] -= 1
        elif o_c > 0 and x_c == 0:
            val = POW10[o_c]
            for cell in cells:
                self.cell_scores[cell] -= val
            self.o_needs[o_c - 1] -= 1

        if player == 1:
            counts[0] += delta
            x_c += delta
        else:
            counts[1] += delta
            o_c += delta

        # Add new contributions to cell_scores
        if x_c > 0 and o_c == 0:
            val = POW10[x_c]
            for cell in cells:
                self.cell_scores[cell] += val
            self.x_needs[x_c - 1] += 1
        elif o_c > 0 and x_c == 0:
            val = POW10[o_c]
            for cell in cells:
                self.cell_scores[cell] += val
            self.o_needs[o_c - 1] += 1

        if x_c == 0 and o_c == 0:
            del self.window_counts[key]

    def _add_move(self, move):
        player, q, r = move
        self.board_state[(q, r)] = player
        self.current_hash ^= self.zobrist_table[move]

        self.empty_cells_set.discard((q, r))

        for dq, dr in NEIGHBOR_OFFSETS:
            nq, nr = q + dq, r + dr
            self.frontier_counts[(nq, nr)] += 1
            if (nq, nr) not in self.board_state:
                self.empty_cells_set.add((nq, nr))

        try:
            windows = self.cell_windows[(q, r)]
        except KeyError:
            windows = []
            for dq, dr in WIN_DIRS:
                for k in range(6):
                    windows.append((dq, dr, q - k * dq, r - k * dr))
            self.cell_windows[(q, r)] = windows

        for key in windows:
            self._update_window(key, player, 1)

    def _remove_move(self, move):
        player, q, r = move
        del self.board_state[(q, r)]
        self.current_hash ^= self.zobrist_table[move]

        if self.frontier_counts[(q, r)] > 0:
            self.empty_cells_set.add((q, r))

        for dq, dr in NEIGHBOR_OFFSETS:
            nq, nr = q + dq, r + dr
            self.frontier_counts[(nq, nr)] -= 1
            if self.frontier_counts[(nq, nr)] == 0:
                del self.frontier_counts[(nq, nr)]
                self.empty_cells_set.discard((nq, nr))

        try:
            windows = self.cell_windows[(q, r)]
        except KeyError:
            windows = []
            for dq, dr in WIN_DIRS:
                for k in range(6):
                    windows.append((dq, dr, q - k * dq, r - k * dr))
            self.cell_windows[(q, r)] = windows

        for key in windows:
            self._update_window(key, player, -1)

    def evaluate(self):
        if self.x_needs[5] > 0:
            return math.inf
        if self.o_needs[5] > 0:
            return -math.inf

        w1, w2, w3, w4, w5 = self.weights
        x_score = (
            w1 * self.x_needs[0]
            + w2 * self.x_needs[1]
            + w3 * self.x_needs[2]
            + w4 * self.x_needs[3]
            + w5 * self.x_needs[4]
        )
        o_score = (
            w1 * self.o_needs[0]
            + w2 * self.o_needs[1]
            + w3 * self.o_needs[2]
            + w4 * self.o_needs[3]
            + w5 * self.o_needs[4]
        )

        return x_score - o_score

    def alphabeta(self, depth, alpha, beta, maximizing_player):
        alpha_orig = alpha
        beta_orig = beta

        tt_index = self.current_hash % TT_SIZE
        tt_entry = self.tt[tt_index]
        tt_best_move = None

        if (
            tt_entry is not None
            and tt_entry[0] == self.current_hash
            and tt_entry[1] == maximizing_player
        ):
            tt_best_move = tt_entry[5]
            if tt_entry[2] >= depth:
                if tt_entry[3] == 0:
                    return tt_entry[4]
                elif tt_entry[3] == 1:
                    alpha = max(alpha, tt_entry[4])
                elif tt_entry[3] == 2:
                    beta = min(beta, tt_entry[4])
                if alpha >= beta:
                    return tt_entry[4]

        is_win = self.o_needs[5] > 0 if maximizing_player else self.x_needs[5] > 0

        if depth == 0 or is_win:
            if is_win:
                return -math.inf if maximizing_player else math.inf
            return self.evaluate()

        # O(1) Candidate Move Ordering
        # Generate and score candidate moves
        cell_scores_list = [
            (cell, self.cell_scores[cell]) for cell in self.empty_cells_set
        ]
        cell_scores_list.sort(key=itemgetter(1), reverse=True)

        if cell_scores_list and cell_scores_list[0][1] >= 10000:
            critical_cells = []
            for cell, score in cell_scores_list:
                if score < 10000:
                    break
                critical_cells.append(cell)

            if len(critical_cells) == 1:
                top_cells = [item[0] for item in cell_scores_list[:20]]
                candidate_moves = [(critical_cells[0], other) for other in top_cells]
                candidate_moves.pop(0)
            else:
                candidate_moves = list(itertools.combinations(critical_cells[:20], 2))
        else:
            top_cells = [item[0] for item in cell_scores_list[:20]]
            candidate_moves = list(itertools.combinations(top_cells, 2))

        score_dict = dict(cell_scores_list)

        def move_score(c):
            s1, s2 = score_dict[c[0]], score_dict[c[1]]
            base_score = (s1 + 0.01 * s2) if s1 >= s2 else (s2 + 0.01 * s1)
            # base_score = s1 + s2
            hc = c if c[0] < c[1] else (c[1], c[0])
            return base_score + self.history_table[hc]

        candidate_moves.sort(key=move_score, reverse=True)
        if not (cell_scores_list and cell_scores_list[0][1] >= 10000):
            candidate_moves = candidate_moves[: min(15, max(3, depth * 3))]

        if not candidate_moves:
            return self.evaluate()

        best_eval = -math.inf if maximizing_player else math.inf
        player = self.current_player()
        pval = player.value
        best_node_move = None

        if tt_best_move and tt_best_move in candidate_moves:
            candidate_moves.remove(tt_best_move)
            candidate_moves.insert(0, tt_best_move)

        if maximizing_player:
            max_eval = -math.inf
            for idx, (c1, c2) in enumerate(candidate_moves):
                original_turn = self.turn

                m1 = (pval, c1[0], c1[1])
                m2 = (pval, c2[0], c2[1])
                self._add_move(m1)
                self._add_move(m2)
                self.turn += 1

                if idx == 0:
                    ev = self.alphabeta(depth - 1, alpha, beta, False)
                else:
                    ev = self.alphabeta(depth - 1, alpha, alpha + 1, False)
                    if alpha < ev < beta:
                        ev = self.alphabeta(depth - 1, alpha, beta, False)

                self._remove_move(m1)
                self._remove_move(m2)
                self.turn = original_turn

                if ev > max_eval:
                    max_eval = ev
                    best_node_move = (c1, c2)
                best_eval = max_eval
                alpha = max(alpha, ev)
                if beta <= alpha:
                    hc = (c1, c2) if c1 < c2 else (c2, c1)
                    self.history_table[hc] += depth * depth
                    break
        else:
            min_eval = math.inf
            for idx, (c1, c2) in enumerate(candidate_moves):
                original_turn = self.turn

                m1 = (pval, c1[0], c1[1])
                m2 = (pval, c2[0], c2[1])
                self._add_move(m1)
                self._add_move(m2)
                self.turn += 1

                if idx == 0:
                    ev = self.alphabeta(depth - 1, alpha, beta, True)
                else:
                    ev = self.alphabeta(depth - 1, beta - 1, beta, True)
                    if alpha < ev < beta:
                        ev = self.alphabeta(depth - 1, alpha, beta, True)

                self._remove_move(m1)
                self._remove_move(m2)
                self.turn = original_turn

                if ev < min_eval:
                    min_eval = ev
                    best_node_move = (c1, c2)
                best_eval = min_eval
                beta = min(beta, ev)
                if beta <= alpha:
                    hc = (c1, c2) if c1 < c2 else (c2, c1)
                    self.history_table[hc] += depth * depth
                    break

        tt_flag = 0
        if best_eval <= alpha_orig:
            tt_flag = 2
        elif best_eval >= beta_orig:
            tt_flag = 1

        existing_tt = self.tt[tt_index]

        if (
            existing_tt is None
            or existing_tt[0] != self.current_hash
            or depth >= existing_tt[2]
        ):
            self.tt[tt_index] = (
                self.current_hash,
                maximizing_player,
                depth,
                tt_flag,
                best_eval,
                best_node_move,
            )

        return best_eval

    def get_best_move(self, depth=DEFAULT_DEPTH):
        self.history_table = defaultdict(int)
        player = self.current_player()
        pval = player.value
        is_maximizing = pval == 1

        best_move = None
        best_overall_move = None

        # 3. Iterative Deepening
        for d in range(1, depth + 1):
            alpha = -math.inf
            beta = math.inf
            best_eval = -math.inf if is_maximizing else math.inf

            # Candidate Move Ordering
            cell_scores_list = [
                (cell, self.cell_scores[cell]) for cell in self.empty_cells_set
            ]
            cell_scores_list.sort(key=itemgetter(1), reverse=True)

            if cell_scores_list and cell_scores_list[0][1] >= 10000:
                critical_cells = [
                    item[0] for item in cell_scores_list if item[1] >= 10000
                ]
                if len(critical_cells) == 1:
                    top_cells = [item[0] for item in cell_scores_list[:25]]
                    candidate_moves = [
                        (critical_cells[0], other)
                        for other in top_cells
                        if other != critical_cells[0]
                    ]
                else:
                    candidate_moves = list(
                        itertools.combinations(critical_cells[:15], 2)
                    )
            else:
                top_cells = [item[0] for item in cell_scores_list[:25]]
                candidate_moves = list(itertools.combinations(top_cells, 2))

            score_dict = dict(cell_scores_list)

            def move_score(c):
                s1, s2 = score_dict[c[0]], score_dict[c[1]]
                base_score = (s1 + 0.01 * s2) if s1 >= s2 else (s2 + 0.01 * s1)
                hc = c if c[0] < c[1] else (c[1], c[0])
                return base_score + self.history_table[hc]

            candidate_moves.sort(key=move_score, reverse=True)
            if not (cell_scores_list and cell_scores_list[0][1] >= 10000):
                candidate_moves = candidate_moves[: min(30, max(15, d * 5))]

            # Inject the best move from the previous iteration to the front
            if best_overall_move in candidate_moves:
                candidate_moves.remove(best_overall_move)
                candidate_moves.insert(0, best_overall_move)

            if not candidate_moves:
                return None

            for idx, (c1, c2) in enumerate(candidate_moves):
                original_turn = self.turn

                m1 = (pval, c1[0], c1[1])
                m2 = (pval, c2[0], c2[1])
                self._add_move(m1)
                self._add_move(m2)
                self.turn += 1

                if idx == 0:
                    ev = self.alphabeta(d - 1, alpha, beta, not is_maximizing)
                else:
                    if is_maximizing:
                        ev = self.alphabeta(d - 1, alpha, alpha + 1, False)
                        if ev > alpha:
                            ev = self.alphabeta(d - 1, alpha, beta, False)
                    else:
                        ev = self.alphabeta(d - 1, beta - 1, beta, True)
                        if ev < beta:
                            ev = self.alphabeta(d - 1, alpha, beta, True)

                self._remove_move(m1)
                self._remove_move(m2)
                self.turn = original_turn

                if is_maximizing:
                    if ev > best_eval or best_move is None:
                        best_eval = ev
                        best_move = ((c1[0], c1[1]), (c2[0], c2[1]))
                    alpha = max(alpha, ev)
                else:
                    if ev < best_eval or best_move is None:
                        best_eval = ev
                        best_move = ((c1[0], c1[1]), (c2[0], c2[1]))
                    beta = min(beta, ev)

                if beta <= alpha:
                    break

            if best_move:
                best_overall_move = best_move

        return best_move


# ── evaluate.py-compatible wrapper ──

_DEFAULT_WEIGHTS = (1, 90, 100, 10000, 9000)


class MinimaxBot(Bot):
    pair_moves = True

    def __init__(self, time_limit=0.05):
        super().__init__(time_limit)
        self._nodes = 0

    def get_move(self, game):
        if not game.board:
            self.last_depth = 0
            return [(0, 0)]

        engine = Engine(_DEFAULT_WEIGHTS)

        # Populate engine board state from HexGame
        for (q, r), player in game.board.items():
            cell = Cell.X if player == Player.A else Cell.O
            engine.moves.append((cell, q, r))

        # Set turn parity so engine.current_player() matches game.current_player
        expected = Cell.X if game.current_player == Player.A else Cell.O
        if engine.current_player() != expected:
            engine.turn = 1

        engine._init_incremental_state()

        result = engine.get_best_move()
        self.last_depth = DEFAULT_DEPTH

        if result is None:
            return [(0, 0)]

        return [result[0], result[1]]

    def __str__(self):
        return "ai_neon"