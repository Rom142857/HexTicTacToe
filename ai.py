"""Minimax bot with iterative deepening, heuristic eval, and transposition table.

Designed for infinite hex grid — no board size limits.
Uses alpha-beta pruning with Zobrist hashing for a transposition table.
Window-based evaluation tracks scoring incrementally via dict-keyed windows
that are created lazily as stones spread across the grid.
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
LINE_SCORES = [0, 1, 10, 200, 1000, 10000, 100000]
# Defensive multipliers per count
_DEF_MULT = [0, 0.8, 0.8, 1.2, 1.5, 3.0, 1.0]

_WIN_LENGTH = 6

# Zobrist hash table — lazily generated random 64-bit values per (cell, player)
_zobrist_rng = random.Random(42)
_zobrist = {}

# Window offset patterns: for each direction, the offsets (k*dq, k*dr)
# so that cell (q, r) belongs to window starting at (q - k*dq, r - k*dr).
# Each cell belongs to 3 * _WIN_LENGTH = 18 windows.
_WINDOW_OFFSETS = tuple(
    (d_idx, k * dq, k * dr)
    for d_idx, (dq, dr) in enumerate(HEX_DIRECTIONS)
    for k in range(_WIN_LENGTH)
)

# Direction vectors indexed by dir_index
_DIR_VECTORS = tuple(HEX_DIRECTIONS)

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
    """Score the position from player's perspective on infinite board.

    Finds all unique win_length-cell windows that contain at least one
    occupied cell, then scores each window based on stone counts.
    """
    opponent = Player.B if player == Player.A else Player.A
    score = 0
    board = game.board
    wl = game.win_length

    seen = set()
    for (q, r) in board:
        for d_idx, (dq, dr) in enumerate(HEX_DIRECTIONS):
            for k in range(wl):
                wkey = (d_idx, q - k * dq, r - k * dr)
                if wkey in seen:
                    continue
                seen.add(wkey)
                sq, sr = wkey[1], wkey[2]
                my_count = 0
                opp_count = 0
                for j in range(wl):
                    p = board.get((sq + j * dq, sr + j * dr))
                    if p == player:
                        my_count += 1
                    elif p is not None:
                        opp_count += 1
                if my_count > 0 and opp_count == 0:
                    s = LINE_SCORES[my_count]
                    if my_count >= 4 and game.move_count > 6:
                        s = int(s * 1.5)
                    score += s
                elif opp_count > 0 and my_count == 0:
                    s = LINE_SCORES[opp_count]
                    if opp_count >= 4 and game.move_count > 6:
                        s = int(s * 1.5)
                    score -= int(s * _DEF_MULT[opp_count])

    return score


def get_candidates(game):
    """Return empty cells within hex-distance 2 of any occupied cell."""
    occupied = list(game.board)
    if not occupied:
        return [(0, 0)]

    candidates = set()
    board = game.board
    for q, r in occupied:
        for dq, dr in _NEIGHBOR_OFFSETS_2:
            nb = (q + dq, r + dr)
            if nb not in board:
                candidates.add(nb)
    return list(candidates)


class MinimaxBot(Bot):
    """Iterative-deepening minimax with alpha-beta pruning, TT, and heuristic eval."""

    def __init__(self, time_limit=0.05):
        super().__init__(time_limit)
        self._deadline = 0
        self._nodes = 0
        self._tt = {}
        self._hash = 0
        self._rc_stack = []
        self._history = {}

    def get_move(self, game):
        self._deadline = time.time() + self.time_limit
        self._player = game.current_player
        self._nodes = 0
        self.last_depth = 0
        if len(self._tt) > 1_000_000:
            self._tt.clear()

        # Compute initial Zobrist hash from board state (lazy generation)
        self._hash = 0
        for (q, r), p in game.board.items():
            zkey = (q, r, p)
            v = _zobrist.get(zkey)
            if v is None:
                v = _zobrist_rng.getrandbits(64)
                _zobrist[zkey] = v
            self._hash ^= v

        # Build score lookup table for current player perspective.
        # _score_table[a_count][b_count] = contribution of a window with
        # that many A/B stones, from self._player's point of view.
        sz = _WIN_LENGTH + 1
        self._score_table = [[0] * sz for _ in range(sz)]
        for a in range(sz):
            for b in range(sz):
                if self._player == Player.A:
                    my, opp = a, b
                else:
                    my, opp = b, a
                if my > 0 and opp == 0:
                    self._score_table[a][b] = LINE_SCORES[my]
                elif opp > 0 and my == 0:
                    self._score_table[a][b] = -LINE_SCORES[opp]

        # Initialize incremental eval: window counts stored as dict
        # keyed by (dir_idx, start_q, start_r) -> [a_count, b_count]
        self._wc = {}
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
                a_count = 0
                b_count = 0
                for j in range(_WIN_LENGTH):
                    cp = board.get((sq + j * dq, sr + j * dr))
                    if cp == Player.A:
                        a_count += 1
                    elif cp == Player.B:
                        b_count += 1
                if a_count > 0 or b_count > 0:
                    self._wc[wkey] = [a_count, b_count]

        # Compute initial eval score from window counts
        self._eval_score = 0
        st = self._score_table
        for counts in self._wc.values():
            self._eval_score += st[counts[0]][counts[1]]

        # Initialize incremental candidate set with reference counting.
        # _cand_refcount[cell] = number of occupied cells within distance 2.
        # _cand_set = empty cells with refcount > 0.
        self._cand_refcount = {}
        for (q, r) in board:
            for dq, dr in _NEIGHBOR_OFFSETS_2:
                nb = (q + dq, r + dr)
                if nb not in board:
                    self._cand_refcount[nb] = self._cand_refcount.get(nb, 0) + 1
        self._cand_set = set(self._cand_refcount)

        if not self._cand_set:
            return (0, 0)
        candidates = list(self._cand_set)
        if len(candidates) == 1:
            return candidates[0]

        random.shuffle(candidates)
        best_move = candidates[0]
        maximizing = game.current_player == self._player

        saved_board = dict(game.board)
        saved_state = (game.current_player, game.moves_left_in_turn,
                       game.winner, game.game_over)
        saved_move_count = game.move_count
        saved_hash = self._hash
        saved_eval = self._eval_score
        saved_wc = {k: v[:] for k, v in self._wc.items()}
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
                 game.winner, game.game_over) = saved_state
                self._hash = saved_hash
                self._eval_score = saved_eval
                self._wc = saved_wc
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
        # Update Zobrist hash (lazy generation)
        zkey = (q, r, player)
        v = _zobrist.get(zkey)
        if v is None:
            v = _zobrist_rng.getrandbits(64)
            _zobrist[zkey] = v
        self._hash ^= v

        # Update incremental eval via window counts + detect win
        st = self._score_table
        wc = self._wc
        won = False
        if player == Player.A:
            for d_idx, oq, or_ in _WINDOW_OFFSETS:
                wkey = (d_idx, q - oq, r - or_)
                counts = wc.get(wkey)
                if counts is None:
                    counts = [0, 0]
                    wc[wkey] = counts
                a, b = counts[0], counts[1]
                self._eval_score += st[a + 1][b] - st[a][b]
                counts[0] = a + 1
                if a + 1 == _WIN_LENGTH and b == 0:
                    won = True
        else:
            for d_idx, oq, or_ in _WINDOW_OFFSETS:
                wkey = (d_idx, q - oq, r - or_)
                counts = wc.get(wkey)
                if counts is None:
                    counts = [0, 0]
                    wc[wkey] = counts
                a, b = counts[0], counts[1]
                self._eval_score += st[a][b + 1] - st[a][b]
                counts[1] = b + 1
                if b + 1 == _WIN_LENGTH and a == 0:
                    won = True

        # Update candidates: (q, r) is now occupied
        self._cand_set.discard((q, r))
        rc = self._cand_refcount
        self._rc_stack.append(rc.pop((q, r), 0))
        board = game.board
        for dq, dr in _NEIGHBOR_OFFSETS_2:
            nb = (q + dq, r + dr)
            rc[nb] = rc.get(nb, 0) + 1
            if nb not in board:
                self._cand_set.add(nb)

        # Place stone and manage game state (bypasses game.make_move/_check_win)
        game.board[(q, r)] = player
        game.move_count += 1
        if won:
            game.winner = player
            game.game_over = True
        else:
            game.moves_left_in_turn -= 1
            if game.moves_left_in_turn <= 0:
                if player == Player.A:
                    game.current_player = Player.B
                else:
                    game.current_player = Player.A
                game.moves_left_in_turn = 2

    def _undo(self, game, q, r, state, player):
        """Undo move and restore Zobrist hash, incremental eval, and candidates."""
        del game.board[(q, r)]
        game.move_count -= 1
        game.current_player, game.moves_left_in_turn, game.winner, game.game_over = state

        # Undo Zobrist hash
        zkey = (q, r, player)
        self._hash ^= _zobrist[zkey]  # guaranteed to exist from _make

        # Undo incremental eval via window counts
        st = self._score_table
        wc = self._wc
        if player == Player.A:
            for d_idx, oq, or_ in _WINDOW_OFFSETS:
                wkey = (d_idx, q - oq, r - or_)
                counts = wc[wkey]
                a, b = counts[0], counts[1]
                self._eval_score += st[a - 1][b] - st[a][b]
                counts[0] = a - 1
        else:
            for d_idx, oq, or_ in _WINDOW_OFFSETS:
                wkey = (d_idx, q - oq, r - or_)
                counts = wc[wkey]
                a, b = counts[0], counts[1]
                self._eval_score += st[a][b - 1] - st[a][b]
                counts[1] = b - 1

        # Undo candidates: (q, r) is empty again
        rc = self._cand_refcount
        for dq, dr in _NEIGHBOR_OFFSETS_2:
            nb = (q + dq, r + dr)
            c = rc[nb] - 1
            if c == 0:
                del rc[nb]
                self._cand_set.discard(nb)
            else:
                rc[nb] = c
        # Restore (q, r) candidate with saved refcount (avoids second 18-neighbor loop)
        saved_rc = self._rc_stack.pop()
        if saved_rc > 0:
            rc[(q, r)] = saved_rc
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
            state = (player, game.moves_left_in_turn, game.winner, game.game_over)
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

        # Move ordering: history heuristic, then TT move to front
        history = self._history
        candidates.sort(key=lambda m: history.get(m, 0), reverse=True)
        if tt_move in self._cand_set:
            idx = candidates.index(tt_move)
            candidates[0], candidates[idx] = candidates[idx], candidates[0]

        maximizing = game.current_player == self._player
        best_move = None

        if maximizing:
            value = -math.inf
            for i, (q, r) in enumerate(candidates):
                player = game.current_player
                state = (player, game.moves_left_in_turn, game.winner, game.game_over)
                self._make(game, q, r)
                # LMR: reduce depth for late moves at sufficient depth
                if i >= 3 and depth >= 3:
                    child_val = self._minimax(game, depth - 2, alpha, beta)
                    if child_val > alpha:
                        child_val = self._minimax(game, depth - 1, alpha, beta)
                else:
                    child_val = self._minimax(game, depth - 1, alpha, beta)
                self._undo(game, q, r, state, player)
                if child_val > value:
                    value = child_val
                    best_move = (q, r)
                alpha = max(alpha, value)
                if alpha >= beta:
                    history[(q, r)] = history.get((q, r), 0) + depth * depth
                    break
        else:
            value = math.inf
            for i, (q, r) in enumerate(candidates):
                player = game.current_player
                state = (player, game.moves_left_in_turn, game.winner, game.game_over)
                self._make(game, q, r)
                # LMR: reduce depth for late moves at sufficient depth
                if i >= 3 and depth >= 3:
                    child_val = self._minimax(game, depth - 2, alpha, beta)
                    if child_val < beta:
                        child_val = self._minimax(game, depth - 1, alpha, beta)
                else:
                    child_val = self._minimax(game, depth - 1, alpha, beta)
                self._undo(game, q, r, state, player)
                if child_val < value:
                    value = child_val
                    best_move = (q, r)
                beta = min(beta, value)
                if alpha >= beta:
                    history[(q, r)] = history.get((q, r), 0) + depth * depth
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
