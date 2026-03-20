"""Minimax bot with iterative deepening, heuristic eval, and transposition table.

Uses alpha-beta pruning with Zobrist hashing for a transposition table.
The TT avoids re-evaluating positions reached via different move orders
(especially common with 2-stones-per-turn) and provides move ordering
from previous iterations.
"""

import math
import random
import time
from bot import Bot
from game import Player, HEX_DIRECTIONS


class TimeUp(Exception):
    pass


def hex_distance(dq, dr):
    ds = -dq - dr
    return max(abs(dq), abs(dr), abs(ds))


# Scores for contiguous groups of length N (index = count)
# Longer lines are exponentially more valuable
LINE_SCORES = [0, 1, 10, 100, 1000, 10000, 100000]


# Zobrist hash table — random 64-bit values for each (cell, player) pair
_zobrist_rng = random.Random(42)
_zobrist = {}
for _q in range(-5, 6):
    for _r in range(-5, 6):
        if abs(-_q - _r) <= 5:
            for _p in (Player.A, Player.B):
                _zobrist[(_q, _r, _p)] = _zobrist_rng.getrandbits(64)

# Pre-compute all 6-cell windows on the hex board and which windows each cell belongs to.
# This allows O(1) incremental evaluation: when a stone is placed, only the ~18 windows
# containing that cell need updating, instead of rescanning the entire board.
def _precompute_windows(radius=5, win_length=6):
    board_cells = set()
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            if abs(-q - r) <= radius:
                board_cells.add((q, r))

    cell_to_windows = {cell: [] for cell in board_cells}
    all_windows = []

    for dq, dr in HEX_DIRECTIONS:
        visited = set()
        for cell in sorted(board_cells):
            if cell in visited:
                continue
            q, r = cell
            while (q - dq, r - dr) in board_cells:
                q -= dq
                r -= dr
            line = []
            cq, cr = q, r
            while (cq, cr) in board_cells:
                visited.add((cq, cr))
                line.append((cq, cr))
                cq += dq
                cr += dr
            for i in range(len(line) - win_length + 1):
                w_idx = len(all_windows)
                all_windows.append(tuple(line[i:i + win_length]))
                for c in line[i:i + win_length]:
                    cell_to_windows[c].append(w_idx)

    # Tuples for faster iteration
    for cell in cell_to_windows:
        cell_to_windows[cell] = tuple(cell_to_windows[cell])

    return all_windows, cell_to_windows

_ALL_WINDOWS, _CELL_TO_WINDOWS = _precompute_windows()
_NUM_WINDOWS = len(_ALL_WINDOWS)

# Pre-compute the 18 neighbor offsets within hex-distance 2 (excluding self)
_NEIGHBOR_OFFSETS_2 = tuple(
    (dq, dr)
    for dq in range(-2, 3)
    for dr in range(-2, 3)
    if hex_distance(dq, dr) <= 2 and (dq, dr) != (0, 0)
)


# TT entry flags
_EXACT = 0
_LOWER = 1  # true value >= stored (beta cutoff)
_UPPER = 2  # true value <= stored (failed low)


def evaluate_position(game, player):
    """Score the position from player's perspective.

    For each line direction, scan every possible 6-cell window and count
    how many belong to each player. A window with stones from both players
    is dead (score 0). Otherwise score based on count.
    """
    opponent = Player.B if player == Player.A else Player.A
    score = 0

    # For each direction, walk all lines through the board
    for dq, dr in HEX_DIRECTIONS:
        # Find all starting cells: cells with no predecessor in this direction
        visited = set()
        for cell in game.board:
            if cell in visited:
                continue
            # Walk backward to find the start of this line
            q, r = cell
            while (q - dq, r - dr) in game.board:
                q -= dq
                r -= dr
            # Now walk forward, collecting the full line
            line = []
            cq, cr = q, r
            while (cq, cr) in game.board:
                visited.add((cq, cr))
                line.append(game.board[(cq, cr)])
                cq += dq
                cr += dr
            # Score all windows of length 6 in this line
            for i in range(len(line) - 5):
                window = line[i:i+6]
                my_count = window.count(player)
                opp_count = window.count(opponent)
                if my_count > 0 and opp_count == 0:
                    score += LINE_SCORES[my_count]
                elif opp_count > 0 and my_count == 0:
                    score -= LINE_SCORES[opp_count]

    return score


def get_candidates(game):
    """Return empty cells within hex-distance 2 of any occupied cell."""
    occupied = [pos for pos, p in game.board.items() if p != Player.NONE]
    if not occupied:
        return [(0, 0)]

    candidates = set()
    for q, r in occupied:
        for dq in range(-2, 3):
            for dr in range(-2, 3):
                if hex_distance(dq, dr) <= 2:
                    nq, nr = q + dq, r + dr
                    if (nq, nr) in game.board and game.board[(nq, nr)] == Player.NONE:
                        candidates.add((nq, nr))
    return list(candidates)


class MinimaxBot(Bot):
    """Iterative-deepening minimax with alpha-beta pruning, TT, and heuristic eval."""

    def __init__(self, time_limit=0.05):
        super().__init__(time_limit)
        self._deadline = 0
        self._nodes = 0
        self._tt = {}
        self._hash = 0

    def get_move(self, game):
        self._deadline = time.time() + self.time_limit
        self._player = game.current_player
        self._nodes = 0
        self.last_depth = 0
        self._tt.clear()

        # Compute initial Zobrist hash from board state
        self._hash = 0
        for pos, p in game.board.items():
            if p != Player.NONE:
                self._hash ^= _zobrist[(pos[0], pos[1], p)]

        # Build score lookup table for current player perspective.
        # _score_table[a_count][b_count] = contribution of a window with
        # that many A/B stones, from self._player's point of view.
        self._score_table = [[0] * 7 for _ in range(7)]
        for a in range(7):
            for b in range(7):
                if self._player == Player.A:
                    my, opp = a, b
                else:
                    my, opp = b, a
                if my > 0 and opp == 0:
                    self._score_table[a][b] = LINE_SCORES[my]
                elif opp > 0 and my == 0:
                    self._score_table[a][b] = -LINE_SCORES[opp]

        # Initialize incremental eval: count stones per window
        self._w_a = [0] * _NUM_WINDOWS
        self._w_b = [0] * _NUM_WINDOWS
        for w_idx, cells in enumerate(_ALL_WINDOWS):
            for cell in cells:
                p = game.board[cell]
                if p == Player.A:
                    self._w_a[w_idx] += 1
                elif p == Player.B:
                    self._w_b[w_idx] += 1

        # Compute initial eval score from window counts
        self._eval_score = 0
        st = self._score_table
        for w_idx in range(_NUM_WINDOWS):
            self._eval_score += st[self._w_a[w_idx]][self._w_b[w_idx]]

        # Initialize incremental candidate set with reference counting.
        # _cand_refcount[cell] = number of occupied cells within distance 2.
        # _cand_set = empty cells with refcount > 0.
        self._cand_refcount = {pos: 0 for pos in game.board}
        for pos, p in game.board.items():
            if p != Player.NONE:
                for dq, dr in _NEIGHBOR_OFFSETS_2:
                    nb = (pos[0] + dq, pos[1] + dr)
                    if nb in self._cand_refcount:
                        self._cand_refcount[nb] += 1
        self._cand_set = set()
        for pos, p in game.board.items():
            if p == Player.NONE and self._cand_refcount[pos] > 0:
                self._cand_set.add(pos)

        if not self._cand_set:
            return (0, 0)
        candidates = list(self._cand_set)
        if len(candidates) == 1:
            return candidates[0]

        random.shuffle(candidates)
        best_move = candidates[0]
        maximizing = game.current_player == self._player

        saved_board = dict(game.board)
        saved_state = game.save_state()
        saved_move_count = game.move_count
        saved_hash = self._hash
        saved_eval = self._eval_score
        saved_wa = self._w_a[:]
        saved_wb = self._w_b[:]
        saved_cand_set = set(self._cand_set)
        saved_cand_rc = dict(self._cand_refcount)

        for depth in range(1, 200):
            try:
                best_move, scores = self._search_root(game, candidates, depth)
                self.last_depth = depth
                # Reorder candidates for next iteration: best-scoring first
                candidates.sort(key=lambda m: scores[m], reverse=maximizing)
            except TimeUp:
                game.board = saved_board
                game.move_count = saved_move_count
                (game.current_player, game.moves_left_in_turn,
                 game.winner, game.winning_cells, game.game_over) = saved_state
                self._hash = saved_hash
                self._eval_score = saved_eval
                self._w_a = saved_wa
                self._w_b = saved_wb
                self._cand_set = saved_cand_set
                self._cand_refcount = saved_cand_rc
                break

        return best_move

    def _check_time(self):
        self._nodes += 1
        if self._nodes % 128 == 0 and time.time() >= self._deadline:
            raise TimeUp

    def _make(self, game, q, r):
        """Make move and update Zobrist hash, incremental eval, and candidates."""
        player = game.current_player
        self._hash ^= _zobrist[(q, r, player)]
        st = self._score_table
        if player == Player.A:
            for w_idx in _CELL_TO_WINDOWS[(q, r)]:
                a = self._w_a[w_idx]
                b = self._w_b[w_idx]
                self._eval_score += st[a + 1][b] - st[a][b]
                self._w_a[w_idx] = a + 1
        else:
            for w_idx in _CELL_TO_WINDOWS[(q, r)]:
                a = self._w_a[w_idx]
                b = self._w_b[w_idx]
                self._eval_score += st[a][b + 1] - st[a][b]
                self._w_b[w_idx] = b + 1
        # Update candidates: (q, r) is now occupied
        self._cand_set.discard((q, r))
        rc = self._cand_refcount
        board = game.board
        for dq, dr in _NEIGHBOR_OFFSETS_2:
            nb = (q + dq, r + dr)
            if nb in rc:
                rc[nb] += 1
                if board[nb] == Player.NONE:
                    self._cand_set.add(nb)
        game.make_move(q, r)

    def _undo(self, game, q, r, state, player):
        """Undo move and restore Zobrist hash, incremental eval, and candidates."""
        game.undo_move(q, r, state)
        self._hash ^= _zobrist[(q, r, player)]
        st = self._score_table
        if player == Player.A:
            for w_idx in _CELL_TO_WINDOWS[(q, r)]:
                a = self._w_a[w_idx]
                b = self._w_b[w_idx]
                self._eval_score += st[a - 1][b] - st[a][b]
                self._w_a[w_idx] = a - 1
        else:
            for w_idx in _CELL_TO_WINDOWS[(q, r)]:
                a = self._w_a[w_idx]
                b = self._w_b[w_idx]
                self._eval_score += st[a][b - 1] - st[a][b]
                self._w_b[w_idx] = b - 1
        # Undo candidates: (q, r) is empty again
        rc = self._cand_refcount
        for dq, dr in _NEIGHBOR_OFFSETS_2:
            nb = (q + dq, r + dr)
            if nb in rc:
                rc[nb] -= 1
                if rc[nb] == 0:
                    self._cand_set.discard(nb)
        if rc[(q, r)] > 0:
            self._cand_set.add((q, r))

    def _tt_key(self, game):
        return (self._hash, game.current_player, game.moves_left_in_turn)

    def _search_root(self, game, candidates, depth):
        """Search all root moves. Returns (best_move, {move: score}) tuple."""
        maximizing = game.current_player == self._player
        best_move = candidates[0]
        alpha = -math.inf
        beta = math.inf

        scores = {}
        for q, r in candidates:
            self._check_time()
            player = game.current_player
            state = game.save_state()
            self._make(game, q, r)
            score = self._minimax(game, depth - 1, alpha, beta)
            self._undo(game, q, r, state, player)
            scores[(q, r)] = score

            if maximizing and score > alpha:
                alpha = score
                best_move = (q, r)
            elif not maximizing and score < beta:
                beta = score
                best_move = (q, r)

        best_score = alpha if maximizing else beta
        # Store root result in TT
        self._tt[self._tt_key(game)] = (depth, best_score, _EXACT, best_move)
        return best_move, scores

    def _minimax(self, game, depth, alpha, beta):
        self._check_time()

        if game.game_over:
            if game.winner == self._player:
                return 100000000
            elif game.winner != Player.NONE:
                return -100000000
            return 0

        # TT lookup
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
            score = self._eval_score
            self._tt[tt_key] = (0, score, _EXACT, None)
            return score

        orig_alpha = alpha
        orig_beta = beta
        candidates = list(self._cand_set)

        # Move ordering: TT best move first
        if tt_move:
            candidates = [tt_move] + [m for m in candidates if m != tt_move]

        maximizing = game.current_player == self._player
        best_move = None

        if maximizing:
            value = -math.inf
            for q, r in candidates:
                player = game.current_player
                state = game.save_state()
                self._make(game, q, r)
                child_val = self._minimax(game, depth - 1, alpha, beta)
                self._undo(game, q, r, state, player)
                if child_val > value:
                    value = child_val
                    best_move = (q, r)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = math.inf
            for q, r in candidates:
                player = game.current_player
                state = game.save_state()
                self._make(game, q, r)
                child_val = self._minimax(game, depth - 1, alpha, beta)
                self._undo(game, q, r, state, player)
                if child_val < value:
                    value = child_val
                    best_move = (q, r)
                beta = min(beta, value)
                if alpha >= beta:
                    break

        # Determine TT flag
        if value <= orig_alpha:
            flag = _UPPER
        elif value >= orig_beta:
            flag = _LOWER
        else:
            flag = _EXACT

        self._tt[tt_key] = (depth, value, flag, best_move)
        return value
