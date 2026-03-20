"""Minimax bot with iterative deepening, heuristic eval, and move ordering.

Uses alpha-beta pruning with a heuristic evaluation function that scores
positions based on contiguous line segments along the three hex axes.
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
    """Iterative-deepening minimax with alpha-beta pruning and heuristic eval."""

    def __init__(self, time_limit=0.05):
        super().__init__(time_limit)
        self._deadline = 0
        self._nodes = 0

    def get_move(self, game):
        self._deadline = time.time() + self.time_limit
        self._player = game.current_player
        self._nodes = 0
        self.last_depth = 0

        candidates = get_candidates(game)
        if len(candidates) == 1:
            return candidates[0]

        random.shuffle(candidates)
        best_move = candidates[0]

        saved_board = dict(game.board)
        saved_state = game.save_state()
        saved_move_count = game.move_count

        for depth in range(1, 200):
            try:
                best_move = self._search_root(game, candidates, depth)
                self.last_depth = depth
            except TimeUp:
                game.board = saved_board
                game.move_count = saved_move_count
                (game.current_player, game.moves_left_in_turn,
                 game.winner, game.winning_cells, game.game_over) = saved_state
                break

        return best_move

    def _check_time(self):
        self._nodes += 1
        if self._nodes % 128 == 0 and time.time() >= self._deadline:
            raise TimeUp

    def _search_root(self, game, candidates, depth):
        maximizing = game.current_player == self._player
        best_move = candidates[0]
        best_score = -math.inf if maximizing else math.inf

        for q, r in candidates:
            self._check_time()
            state = game.save_state()
            game.make_move(q, r)
            score = self._minimax(game, depth - 1, -math.inf, math.inf)
            game.undo_move(q, r, state)

            if maximizing and score > best_score:
                best_score = score
                best_move = (q, r)
            elif not maximizing and score < best_score:
                best_score = score
                best_move = (q, r)

        return best_move

    def _minimax(self, game, depth, alpha, beta):
        self._check_time()

        if game.game_over:
            if game.winner == self._player:
                return 100000000
            elif game.winner != Player.NONE:
                return -100000000
            return 0

        if depth == 0:
            return evaluate_position(game, self._player)

        candidates = get_candidates(game)
        maximizing = game.current_player == self._player

        if maximizing:
            value = -math.inf
            for q, r in candidates:
                state = game.save_state()
                game.make_move(q, r)
                value = max(value, self._minimax(game, depth - 1, alpha, beta))
                game.undo_move(q, r, state)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = math.inf
            for q, r in candidates:
                state = game.save_state()
                game.make_move(q, r)
                value = min(value, self._minimax(game, depth - 1, alpha, beta))
                game.undo_move(q, r, state)
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value
