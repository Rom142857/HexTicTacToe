"""Minimax bot with heuristic evaluation, move ordering, and tight candidate generation.

Uses iterative deepening alpha-beta with:
- Line-based heuristic evaluation (scores contiguous groups along hex axes)
- Candidate moves limited to distance 1 from occupied cells
- Move ordering by heuristic score for better pruning
"""

import math
import time
from bot import Bot
from game import Player, HEX_DIRECTIONS


class TimeUp(Exception):
    pass


# Precompute all lines of length 6 on the board for heuristic evaluation
def _precompute_lines(radius, win_length):
    """Generate all possible winning lines (sequences of win_length cells)."""
    cells = set()
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            if abs(-q - r) <= radius:
                cells.add((q, r))

    lines = []
    seen = set()
    for q, r in cells:
        for dq, dr in HEX_DIRECTIONS:
            line = []
            for i in range(win_length):
                pos = (q + dq * i, r + dr * i)
                if pos in cells:
                    line.append(pos)
                else:
                    break
            if len(line) == win_length:
                key = (line[0], (dq, dr))
                if key not in seen:
                    seen.add(key)
                    lines.append(tuple(line))
    return lines


LINES = _precompute_lines(5, 6)

# For each cell, which lines pass through it (for incremental eval)
CELL_TO_LINES = {}
for i, line in enumerate(LINES):
    for pos in line:
        if pos not in CELL_TO_LINES:
            CELL_TO_LINES[pos] = []
        CELL_TO_LINES[pos].append(i)

# Precompute neighbors (distance 1) for each cell
ALL_CELLS = set()
for q in range(-5, 6):
    for r in range(-5, 6):
        if abs(-q - r) <= 5:
            ALL_CELLS.add((q, r))

HEX_NEIGHBORS = {}
_neighbor_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
for cell in ALL_CELLS:
    q, r = cell
    HEX_NEIGHBORS[cell] = [
        (q + dq, r + dr) for dq, dr in _neighbor_dirs
        if (q + dq, r + dr) in ALL_CELLS
    ]


# Score table for line patterns: (my_count, opp_count) -> score
# Only lines with no opponent stones are useful for attack; vice versa for defense
LINE_SCORES = {}
for my in range(7):
    for opp in range(7):
        if my > 0 and opp > 0:
            LINE_SCORES[(my, opp)] = 0  # contested line, no value
        elif my > 0:
            # Exponential scoring for our stones in an open line
            LINE_SCORES[(my, opp)] = [0, 1, 4, 16, 64, 256, 10000][my]
        elif opp > 0:
            # Slightly higher weight for blocking opponent threats
            LINE_SCORES[(my, opp)] = -[0, 1, 4, 16, 64, 256, 10000][opp]
        else:
            LINE_SCORES[(my, opp)] = 0


def evaluate_board(game, player):
    """Heuristic evaluation of the board from player's perspective."""
    if game.winner == player:
        return 100000
    elif game.winner != Player.NONE:
        return -100000

    opp = Player.B if player == Player.A else Player.A
    score = 0

    for line in LINES:
        my_count = 0
        opp_count = 0
        for pos in line:
            cell = game.board[pos]
            if cell == player:
                my_count += 1
            elif cell == opp:
                opp_count += 1
        score += LINE_SCORES[(my_count, opp_count)]

    return score


def get_candidates(game):
    """Return empty cells adjacent to any occupied cell, ordered by heuristic."""
    occupied = []
    for pos, p in game.board.items():
        if p != Player.NONE:
            occupied.append(pos)

    if not occupied:
        return [(0, 0)]

    candidates = set()
    for pos in occupied:
        for neighbor in HEX_NEIGHBORS[pos]:
            if game.board[neighbor] == Player.NONE:
                candidates.add(neighbor)

    if not candidates:
        # Fallback: any empty cell
        return [pos for pos, p in game.board.items() if p == Player.NONE][:10]

    return list(candidates)


def score_move(game, pos, player):
    """Quick heuristic score for move ordering."""
    q, r = pos
    score = 0
    opp = Player.B if player == Player.A else Player.A

    # Check lines through this cell
    if pos in CELL_TO_LINES:
        for li in CELL_TO_LINES[pos]:
            line = LINES[li]
            my_count = 0
            opp_count = 0
            for p in line:
                cell = game.board[p]
                if cell == player:
                    my_count += 1
                elif cell == opp:
                    opp_count += 1
            # Placing here adds to our count
            if opp_count == 0:
                score += [0, 1, 4, 16, 64, 256, 10000][my_count]
            # Blocking opponent
            if my_count == 0:
                score += [0, 1, 4, 16, 64, 256, 10000][opp_count]

    # Centrality bonus (prefer center)
    s = -q - r
    dist = max(abs(q), abs(r), abs(s))
    score += (5 - dist)

    return score


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

        # Order candidates by heuristic score
        candidates.sort(key=lambda p: score_move(game, p, self._player), reverse=True)

        # Cap candidates to limit branching factor
        if len(candidates) > 20:
            candidates = candidates[:20]

        best_move = candidates[0]

        for depth in range(1, 200):
            try:
                move = self._search_root(game, candidates, depth)
                best_move = move
                self.last_depth = depth
            except TimeUp:
                break

        return best_move

    def _check_time(self):
        self._nodes += 1
        if self._nodes % 128 == 0 and time.time() >= self._deadline:
            raise TimeUp

    def _search_root(self, game, candidates, depth):
        best_move = candidates[0]
        best_score = -math.inf

        for pos in candidates:
            self._check_time()
            q, r = pos
            state = game.save_state()
            game.make_move(q, r)
            score = -self._negamax(game, depth - 1, -math.inf, math.inf)
            game.undo_move(q, r, state)

            if score > best_score:
                best_score = score
                best_move = pos

        return best_move

    def _negamax(self, game, depth, alpha, beta):
        self._check_time()

        if game.game_over:
            if game.winner == game.current_player:
                return 100000
            elif game.winner != Player.NONE:
                return -100000
            return 0

        if depth == 0:
            # Evaluate from current player's perspective
            score = evaluate_board(game, game.current_player)
            return score

        candidates = get_candidates(game)
        # Move ordering
        candidates.sort(
            key=lambda p: score_move(game, p, game.current_player), reverse=True
        )
        if len(candidates) > 15:
            candidates = candidates[:15]

        value = -math.inf
        for pos in candidates:
            q, r = pos
            state = game.save_state()
            game.make_move(q, r)
            child_val = -self._negamax(game, depth - 1, -beta, -alpha)
            game.undo_move(q, r, state)
            if child_val > value:
                value = child_val
            if value > alpha:
                alpha = value
            if alpha >= beta:
                break

        return value
