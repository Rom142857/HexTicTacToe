"""Negamax bot with iterative deepening, heuristic eval, and transposition table.

Uses alpha-beta pruning with Zobrist hashing for a transposition table.
Negamax simplifies minimax by always scoring from the current player's
perspective and negating at each level.
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

# TT entry flags
_EXACT = 0
_LOWER = 1  # true value >= stored (beta cutoff)
_UPPER = 2  # true value <= stored (failed low)


def evaluate_position(game, player):
    """Score the position from player's perspective."""
    opponent = Player.B if player == Player.A else Player.A
    score = 0

    for dq, dr in HEX_DIRECTIONS:
        visited = set()
        for cell in game.board:
            if cell in visited:
                continue
            q, r = cell
            while (q - dq, r - dr) in game.board:
                q -= dq
                r -= dr
            line = []
            cq, cr = q, r
            while (cq, cr) in game.board:
                visited.add((cq, cr))
                line.append(game.board[(cq, cr)])
                cq += dq
                cr += dr
            for i in range(len(line) - 5):
                window = line[i:i+6]
                my_count = window.count(player)
                opp_count = window.count(opponent)
                if my_count > 0 and opp_count == 0:
                    score += LINE_SCORES[my_count]
                elif opp_count > 0 and my_count == 0:
                    score -= int(LINE_SCORES[opp_count] * 1.2)

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
    """Iterative-deepening negamax with alpha-beta pruning, TT, and heuristic eval."""

    def __init__(self, time_limit=0.05):
        super().__init__(time_limit)
        self._deadline = 0
        self._nodes = 0
        self._tt = {}
        self._hash = 0

    def get_move(self, game):
        self._deadline = time.time() + self.time_limit
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
        if self._nodes % 128 == 0 and time.time() >= self._deadline:
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

        cur_player = game.current_player
        for q, r in ordered:
            self._check_time()
            player = game.current_player
            state = game.save_state()
            self._make(game, q, r)
            # If the next player is the same (2nd stone of turn), don't negate
            if game.current_player == cur_player:
                score = self._negamax(game, depth - 1, alpha, beta)
            else:
                score = -self._negamax(game, depth - 1, -beta, -alpha)
            self._undo(game, q, r, state, player)

            if score > alpha:
                alpha = score
                best_move = (q, r)

        self._tt[self._tt_key(game)] = (depth, alpha, _EXACT, best_move)
        return best_move

    def _negamax(self, game, depth, alpha, beta):
        self._check_time()

        if game.game_over:
            if game.winner != Player.NONE:
                # Winner is current_player (make_move doesn't switch on win)
                return 100000000
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
            score = evaluate_position(game, game.current_player)
            self._tt[tt_key] = (0, score, _EXACT, None)
            return score

        orig_alpha = alpha
        candidates = get_candidates(game)

        if tt_move:
            ordered = [tt_move] + [m for m in candidates if m != tt_move]
        else:
            ordered = candidates

        cur_player = game.current_player
        value = -math.inf
        best_move = None

        for q, r in ordered:
            player = game.current_player
            state = game.save_state()
            self._make(game, q, r)
            # If the next player is the same (2nd stone of turn), don't negate
            if game.current_player == cur_player:
                child_val = self._negamax(game, depth - 1, alpha, beta)
            else:
                child_val = -self._negamax(game, depth - 1, -beta, -alpha)
            self._undo(game, q, r, state, player)
            if child_val > value:
                value = child_val
                best_move = (q, r)
            alpha = max(alpha, value)
            if alpha >= beta:
                break

        # Determine TT flag
        if value <= orig_alpha:
            flag = _UPPER
        elif value >= beta:
            flag = _LOWER
        else:
            flag = _EXACT

        self._tt[tt_key] = (depth, value, flag, best_move)
        return value
