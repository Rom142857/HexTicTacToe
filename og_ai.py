"""Minimax bot with iterative deepening, heuristic eval, and transposition table.

Designed for infinite hex grid — no board size limits.
Uses alpha-beta pruning with Zobrist hashing for a transposition table.
Window-based evaluation tracks scoring incrementally via dict-keyed windows
that are created lazily as stones spread across the grid.

Each minimax ply operates on full turns (2 stones), not individual stones.
"""

import math
import random
import time
from itertools import combinations
from bot import Bot
from game import Player, HEX_DIRECTIONS

# ── Hyperparameters ──────────────────────────────────────────────────
LINE_SCORES = [0, 0, 10, 200, 2000, 50000, 100000]  # eval score per stone count in a window
_DEF_MULT = [0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]      # defensive multiplier per opponent count (no scaling)
_CANDIDATE_CAP = 15          # max single-cell candidates in minimax
_ROOT_CANDIDATE_CAP = 15     # max single-cell candidates at root
_NEIGHBOR_DIST = 1           # hex distance for candidate generation
_DELTA_WEIGHT = 0.001        # weight of eval delta vs history in move ordering


class TimeUp(Exception):
    pass


def hex_distance(dq, dr):
    ds = -dq - dr
    return max(abs(dq), abs(dr), abs(ds))



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

# Pre-compute neighbor offsets within _NEIGHBOR_DIST (excluding self)
_NEIGHBOR_OFFSETS_2 = tuple(
    (dq, dr)
    for dq in range(-_NEIGHBOR_DIST, _NEIGHBOR_DIST + 1)
    for dr in range(-_NEIGHBOR_DIST, _NEIGHBOR_DIST + 1)
    if hex_distance(dq, dr) <= _NEIGHBOR_DIST and (dq, dr) != (0, 0)
)

# TT entry flags
_EXACT = 0
_LOWER = 1  # true value >= stored (beta cutoff)
_UPPER = 2  # true value <= stored (failed low)

_WIN_SCORE = 100000000


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
                    score += LINE_SCORES[my_count]
                elif opp_count > 0 and my_count == 0:
                    score -= int(LINE_SCORES[opp_count] * _DEF_MULT[opp_count])

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
    """Iterative-deepening minimax with alpha-beta pruning, TT, and heuristic eval.

    Each minimax ply = one full turn (2 stones). The first move of the game
    is hardcoded to (0,0) since the board is infinite and symmetric.
    """

    pair_moves = True  # returns both moves of a double turn

    def __init__(self, time_limit=0.05):
        super().__init__(time_limit)
        self._deadline = 0
        self._nodes = 0
        self._tt = {}
        self._hash = 0
        self._rc_stack = []
        self._history = {}
        self.last_ebf = 0

    def get_move(self, game):
        # First move is arbitrary on infinite board
        if not game.board:
            return [(0, 0)]

        self._deadline = time.time() + self.time_limit * 2
        self._player = game.current_player
        self._nodes = 0
        self.last_depth = 0
        self.last_ebf = 0
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
                    self._score_table[a][b] = -int(LINE_SCORES[opp] * _DEF_MULT[opp])

        # Initialize incremental eval: window counts
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
        self._cand_refcount = {}
        for (q, r) in board:
            for dq, dr in _NEIGHBOR_OFFSETS_2:
                nb = (q + dq, r + dr)
                if nb not in board:
                    self._cand_refcount[nb] = self._cand_refcount.get(nb, 0) + 1
        self._cand_set = set(self._cand_refcount)

        if not self._cand_set:
            return [(0, 0)]

        # Generate and sort initial turn candidates
        maximizing = game.current_player == self._player
        is_a = game.current_player == Player.A
        turns = self._generate_turns(game)
        if not turns:
            return [(0, 0)]

        best_move = list(turns[0])

        # Save full state for rollback on TimeUp
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
                nodes_before = self._nodes
                result, scores = self._search_root(game, turns, depth)
                best_move = list(result)
                self.last_depth = depth
                nodes_this_depth = self._nodes - nodes_before
                if nodes_this_depth > 1:
                    self.last_ebf = round(nodes_this_depth ** (1.0 / depth), 1)
                # Reorder turns by score for next iteration
                turns.sort(key=lambda t: scores.get(t, 0), reverse=maximizing)
                # Stop if forced win/loss found
                if abs(scores.get(result, 0)) >= _WIN_SCORE:
                    break
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

    def _move_delta(self, q, r, is_a):
        """Eval delta from placing at (q,r) — read-only, no state mutation."""
        wc = self._wc
        st = self._score_table
        delta = 0
        if is_a:
            new_w = st[1][0]
            for d_idx, oq, or_ in _WINDOW_OFFSETS:
                counts = wc.get((d_idx, q - oq, r - or_))
                if counts is not None:
                    delta += st[counts[0] + 1][counts[1]] - st[counts[0]][counts[1]]
                else:
                    delta += new_w
        else:
            new_w = st[0][1]
            for d_idx, oq, or_ in _WINDOW_OFFSETS:
                counts = wc.get((d_idx, q - oq, r - or_))
                if counts is not None:
                    delta += st[counts[0]][counts[1] + 1] - st[counts[0]][counts[1]]
                else:
                    delta += new_w
        return delta

    def _find_instant_win(self, game, player):
        """Find a winning turn if player can complete a window in one turn (2 stones).

        Checks for windows with 4 or 5 stones and 0 opponent — filling
        the remaining 2 or 1 empties in a single turn wins immediately.
        Returns a (cell1, cell2) turn tuple, or None.
        """
        p_idx = 0 if player == Player.A else 1
        o_idx = 1 - p_idx
        board = game.board
        for wkey, counts in self._wc.items():
            if counts[p_idx] >= _WIN_LENGTH - 2 and counts[o_idx] == 0:
                empties = counts[p_idx] == _WIN_LENGTH - 2  # expect 2 empties
                d_idx, sq, sr = wkey
                dq, dr = _DIR_VECTORS[d_idx]
                cells = []
                for j in range(_WIN_LENGTH):
                    cell = (sq + j * dq, sr + j * dr)
                    if cell not in board:
                        cells.append(cell)
                if len(cells) == 1:
                    # 5 of 6: pair with any candidate
                    other = next((c for c in self._cand_set if c != cells[0]), cells[0])
                    return (min(cells[0], other), max(cells[0], other))
                elif len(cells) == 2:
                    # 4 of 6: play both empties
                    return (min(cells[0], cells[1]), max(cells[0], cells[1]))
        return None

    def _find_threat_cells(self, game, player):
        """Return set of empty cells in windows where player has 4 and opponent has 0."""
        threat_cells = set()
        p_idx = 0 if player == Player.A else 1
        o_idx = 1 - p_idx
        board = game.board
        for wkey, counts in self._wc.items():
            if counts[p_idx] >= 4 and counts[o_idx] == 0:
                d_idx, sq, sr = wkey
                dq, dr = _DIR_VECTORS[d_idx]
                for j in range(_WIN_LENGTH):
                    cell = (sq + j * dq, sr + j * dr)
                    if cell not in board:
                        threat_cells.add(cell)
        return threat_cells

    def _filter_turns_by_threats(self, game, turns):
        """Filter turns to forced moves when threats of four exist."""
        current = game.current_player
        opponent = Player.B if current == Player.A else Player.A

        my_threats = self._find_threat_cells(game, current)
        if my_threats:
            winning = [t for t in turns
                       if t[0] in my_threats or t[1] in my_threats]
            if winning:
                return winning

        opp_threats = self._find_threat_cells(game, opponent)
        if opp_threats:
            blocking = [t for t in turns
                        if t[0] in opp_threats or t[1] in opp_threats]
            if blocking:
                return blocking

        return turns

    def _make_turn(self, game, turn):
        """Apply a full turn (2 stones). Returns undo info list."""
        m1, m2 = turn
        p1 = game.current_player
        s1 = (p1, game.moves_left_in_turn, game.winner, game.game_over)
        self._make(game, m1[0], m1[1])
        if game.game_over:
            # First stone won — no second stone
            return [(m1, s1, p1)]
        p2 = game.current_player
        s2 = (p2, game.moves_left_in_turn, game.winner, game.game_over)
        self._make(game, m2[0], m2[1])
        return [(m1, s1, p1), (m2, s2, p2)]

    def _undo_turn(self, game, undo_info):
        """Undo a full turn in reverse order."""
        for cell, state, player in reversed(undo_info):
            self._undo(game, cell[0], cell[1], state, player)

    def _generate_turns(self, game):
        """Generate C(N,2) turn pairs, ordered by sorting singles first."""
        # Instant win: current player can complete a window in one turn
        win_turn = self._find_instant_win(game, game.current_player)
        if win_turn:
            return [win_turn]

        candidates = list(self._cand_set)
        if len(candidates) < 2:
            if candidates:
                return [(candidates[0], candidates[0])]
            return []

        is_a = game.current_player == Player.A
        move_delta = self._move_delta
        maximizing = game.current_player == self._player

        # Sort singles by delta — pairs from combinations inherit good ordering
        candidates.sort(key=lambda c: move_delta(c[0], c[1], is_a), reverse=maximizing)
        candidates = candidates[:_ROOT_CANDIDATE_CAP]

        turns = list(combinations(candidates, 2))
        return self._filter_turns_by_threats(game, turns)

    def _search_root(self, game, turns, depth):
        """Search all root turns, narrowing alpha-beta window as we go."""
        maximizing = game.current_player == self._player
        best_turn = turns[0]
        alpha = -math.inf
        beta = math.inf

        scores = {}
        for turn in turns:
            self._check_time()
            undo_info = self._make_turn(game, turn)
            if game.game_over:
                score = _WIN_SCORE if game.winner == self._player else -_WIN_SCORE
            else:
                score = self._minimax(game, depth - 1, alpha, beta)
            self._undo_turn(game, undo_info)
            scores[turn] = score

            if maximizing and score > alpha:
                alpha = score
                best_turn = turn
            elif not maximizing and score < beta:
                beta = score
                best_turn = turn

        best_score = alpha if maximizing else beta
        self._tt[self._tt_key(game)] = (depth, best_score, _EXACT, best_turn)
        return best_turn, scores

    def _minimax(self, game, depth, alpha, beta):
        self._check_time()

        if game.game_over:
            if game.winner == self._player:
                return _WIN_SCORE
            elif game.winner != Player.NONE:
                return -_WIN_SCORE
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

        # Instant win: skip full turn generation
        win_turn = self._find_instant_win(game, game.current_player)
        if win_turn:
            undo_info = self._make_turn(game, win_turn)
            score = _WIN_SCORE if game.winner == self._player else -_WIN_SCORE
            self._undo_turn(game, undo_info)
            self._tt[tt_key] = (depth, score, _EXACT, win_turn)
            return score

        # Instant loss: opponent wins next turn and we can't block all threats
        opponent = Player.B if game.current_player == Player.A else Player.A
        opp_win = self._find_instant_win(game, opponent)
        if opp_win:
            # Opponent has at least one winning window — check if we can block.
            # Collect all opponent threat windows (4+ of 6) to find distinct
            # windows that each need at least one cell blocked.
            opp_threats = self._find_threat_cells(game, opponent)
            # We get 2 moves — count how many independent windows need blocking.
            # Each window with 4-of-6 needs one of its 2 empties blocked;
            # each window with 5-of-6 needs its 1 empty blocked.
            # If the threat cells can't all be covered by 2 moves, we lose.
            p_idx = 0 if opponent == Player.A else 1
            o_idx = 1 - p_idx
            # Collect sets of empties per threat window — must hit at least one per window
            must_hit = []
            board = game.board
            for wkey, counts in self._wc.items():
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
                # Check if any pair of cells hits all windows
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

        # Generate and order turns
        candidates = list(self._cand_set)
        if len(candidates) < 2:
            # Edge case: fewer than 2 candidates
            if not candidates:
                score = self._eval_score
                self._tt[tt_key] = (depth, score, _EXACT, None)
                return score
            # Single candidate — make it as a lone move (shouldn't normally happen)
            c = candidates[0]
            turns = [(c, c)]
        else:
            is_a = game.current_player == Player.A
            history = self._history
            move_delta = self._move_delta

            # Sort singles by history + delta, then pairs inherit good ordering
            delta_sign = _DELTA_WEIGHT if maximizing else -_DELTA_WEIGHT
            candidates.sort(
                key=lambda c: history.get(c, 0) + move_delta(c[0], c[1], is_a) * delta_sign,
                reverse=True)
            candidates = candidates[:_CANDIDATE_CAP]

            turns = list(combinations(candidates, 2))
            turns = self._filter_turns_by_threats(game, turns)

        # TT move to front
        if tt_move is not None:
            try:
                idx = turns.index(tt_move)
                turns[0], turns[idx] = turns[idx], turns[0]
            except ValueError:
                pass

        best_move = None

        if maximizing:
            value = -math.inf
            for turn in turns:
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
                    history[turn[0]] = history.get(turn[0], 0) + depth * depth
                    history[turn[1]] = history.get(turn[1], 0) + depth * depth
                    break
        else:
            value = math.inf
            for turn in turns:
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
                    history[turn[0]] = history.get(turn[0], 0) + depth * depth
                    history[turn[1]] = history.get(turn[1], 0) + depth * depth
                    break

        if value <= orig_alpha:
            flag = _UPPER
        elif value >= orig_beta:
            flag = _LOWER
        else:
            flag = _EXACT

        self._tt[tt_key] = (depth, value, flag, best_move)
        return value
