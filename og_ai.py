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
LINE_SCORES = [0, 1, 10, 200, 1000, 10000, 100000]
# Defensive multipliers per count — higher counts need more urgent blocking
_DEF_MULT = [0, 0.8, 0.8, 1.2, 1.5, 3.0, 1.0]


# Zobrist hash table — random 64-bit values for each (cell, player) pair
_zobrist_rng = random.Random(42)
_zobrist = {}
for _q in range(-5, 6):
    for _r in range(-5, 6):
        if abs(-_q - _r) <= 5:
            for _p in (Player.A, Player.B):
                _zobrist[(_q, _r, _p)] = _zobrist_rng.getrandbits(64)

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
                    s = LINE_SCORES[my_count]
                    # Late game: boost high-count offensive windows
                    if my_count >= 4 and game.move_count > 10:
                        s = int(s * 1.5)
                    score += s
                elif opp_count > 0 and my_count == 0:
                    s = LINE_SCORES[opp_count]
                    if opp_count >= 4 and game.move_count > 10:
                        s = int(s * 1.5)
                    score -= int(s * _DEF_MULT[opp_count])

    return score


# Precomputed distance-2 offsets (18 cells, avoids hex_distance calls)
_D2_OFFSETS = [(dq, dr) for dq in range(-2, 3) for dr in range(-2, 3)
               if hex_distance(dq, dr) <= 2 and (dq, dr) != (0, 0)]


def get_candidates(game):
    """Return empty cells within hex-distance 2 of any occupied cell."""
    occupied = [pos for pos, p in game.board.items() if p != Player.NONE]
    if not occupied:
        return [(0, 0)]

    candidates = set()
    for q, r in occupied:
        for dq, dr in _D2_OFFSETS:
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

        candidates = get_candidates(game)
        if len(candidates) == 1:
            return candidates[0]

        random.shuffle(candidates)
        best_move = candidates[0]

        saved_board = dict(game.board)
        saved_state = game.save_state()
        saved_move_count = game.move_count
        saved_hash = self._hash

        for depth in range(1, 200):
            try:
                best_move = self._search_root(game, candidates, depth)
                self.last_depth = depth
            except TimeUp:
                game.board = saved_board
                game.move_count = saved_move_count
                (game.current_player, game.moves_left_in_turn,
                 game.winner, game.winning_cells, game.game_over) = saved_state
                self._hash = saved_hash
                break

        return best_move

    def _check_time(self):
        self._nodes += 1
        if self._nodes % 512 == 0 and time.time() >= self._deadline:
            raise TimeUp

    def _make(self, game, q, r):
        """Make move and update Zobrist hash."""
        player = game.current_player
        self._hash ^= _zobrist[(q, r, player)]
        game.make_move(q, r)

    def _undo(self, game, q, r, state, player):
        """Undo move and restore Zobrist hash."""
        game.undo_move(q, r, state)
        self._hash ^= _zobrist[(q, r, player)]

    def _tt_key(self, game):
        return (self._hash, game.current_player, game.moves_left_in_turn)

    def _search_root(self, game, candidates, depth):
        maximizing = game.current_player == self._player
        best_move = candidates[0]
        alpha = -math.inf
        beta = math.inf

        # Move ordering: TT best move first
        tt_entry = self._tt.get(self._tt_key(game))
        if tt_entry and tt_entry[3]:
            tt_move = tt_entry[3]
            ordered = [tt_move] + [m for m in candidates if m != tt_move]
        else:
            ordered = candidates

        for q, r in ordered:
            self._check_time()
            player = game.current_player
            state = game.save_state()
            self._make(game, q, r)
            score = self._minimax(game, depth - 1, alpha, beta)
            self._undo(game, q, r, state, player)

            if maximizing and score > alpha:
                alpha = score
                best_move = (q, r)
            elif not maximizing and score < beta:
                beta = score
                best_move = (q, r)

        # Store root result in TT
        best_score = alpha if maximizing else beta
        self._tt[self._tt_key(game)] = (depth, best_score, _EXACT, best_move)
        return best_move

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
            score = evaluate_position(game, self._player)
            self._tt[tt_key] = (0, score, _EXACT, None)
            return score

        orig_alpha = alpha
        orig_beta = beta
        candidates = get_candidates(game)

        # Move ordering: TT move first
        if tt_move:
            ordered = [tt_move] + [m for m in candidates if m != tt_move]
        else:
            ordered = candidates

        maximizing = game.current_player == self._player
        best_move = None

        if maximizing:
            value = -math.inf
            for q, r in ordered:
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
            for q, r in ordered:
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
