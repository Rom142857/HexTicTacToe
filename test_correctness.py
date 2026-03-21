"""Correctness test: verify ai.py and og_ai.py produce identical moves at fixed depth 2.

Runs a suite of random game positions through both bots and checks that
they return the same moves. Caps iterative deepening at depth 2 with a
huge time limit to avoid timing-related non-determinism.
"""

import random
import sys
import time
from game import HexGame, Player

import ai as ai_new
import og_ai as ai_og


class DepthLimitedNew(ai_new.MinimaxBot):
    """ai.py MinimaxBot capped at a fixed depth."""

    def __init__(self, max_depth=2):
        super().__init__(time_limit=9999)
        self._max_depth = max_depth

    def get_move(self, game):
        if not game.board:
            return [(0, 0)]

        self._deadline = time.time() + 9999
        self._player = game.current_player
        self._nodes = 0
        self.last_depth = 0
        self.last_ebf = 0
        self._tt.clear()
        self._history.clear()

        # Zobrist hash
        self._hash = 0
        for (q, r), p in game.board.items():
            zkey = (q, r, p)
            v = ai_new._zobrist.get(zkey)
            if v is None:
                v = ai_new._zobrist_rng.getrandbits(64)
                ai_new._zobrist[zkey] = v
            self._hash ^= v

        # Score table
        sz = ai_new._WIN_LENGTH + 1
        self._score_table = [[0] * sz for _ in range(sz)]
        for a in range(sz):
            for b in range(sz):
                if self._player == Player.A:
                    my, opp = a, b
                else:
                    my, opp = b, a
                if my > 0 and opp == 0:
                    self._score_table[a][b] = ai_new.LINE_SCORES[my]
                elif opp > 0 and my == 0:
                    self._score_table[a][b] = -int(ai_new.LINE_SCORES[opp] * ai_new._DEF_MULT[opp])

        # Window counts
        self._wc = {}
        board = game.board
        seen = set()
        for (q, r) in board:
            for d_idx, oq, or_ in ai_new._WINDOW_OFFSETS:
                wkey = (d_idx, q - oq, r - or_)
                if wkey in seen:
                    continue
                seen.add(wkey)
                dq, dr = ai_new._DIR_VECTORS[d_idx]
                sq, sr = wkey[1], wkey[2]
                a_count = b_count = 0
                for j in range(ai_new._WIN_LENGTH):
                    cp = board.get((sq + j * dq, sr + j * dr))
                    if cp == Player.A:
                        a_count += 1
                    elif cp == Player.B:
                        b_count += 1
                if a_count > 0 or b_count > 0:
                    self._wc[wkey] = [a_count, b_count]

        self._eval_score = 0
        self._hot_a = set()
        self._hot_b = set()
        st = self._score_table
        for wkey, counts in self._wc.items():
            self._eval_score += st[counts[0]][counts[1]]
            if counts[0] >= 4:
                self._hot_a.add(wkey)
            if counts[1] >= 4:
                self._hot_b.add(wkey)

        # Candidates
        self._cand_refcount = {}
        for (q, r) in board:
            for dq, dr in ai_new._NEIGHBOR_OFFSETS_2:
                nb = (q + dq, r + dr)
                if nb not in board:
                    self._cand_refcount[nb] = self._cand_refcount.get(nb, 0) + 1
        self._cand_set = set(self._cand_refcount)

        if not self._cand_set:
            return [(0, 0)]

        turns = self._generate_turns(game)
        if not turns:
            return [(0, 0)]

        best_move = list(turns[0])
        maximizing = game.current_player == self._player

        for depth in range(1, self._max_depth + 1):
            result, scores = self._search_root(game, turns, depth)
            best_move = list(result)
            self.last_depth = depth
            turns.sort(key=lambda t: scores.get(t, 0), reverse=maximizing)

        return best_move


class DepthLimitedOG(ai_og.MinimaxBot):
    """og_ai.py MinimaxBot capped at a fixed depth."""

    def __init__(self, max_depth=2):
        super().__init__(time_limit=9999)
        self._max_depth = max_depth

    def get_move(self, game):
        if not game.board:
            return [(0, 0)]

        self._deadline = time.time() + 9999
        self._player = game.current_player
        self._nodes = 0
        self.last_depth = 0
        self.last_ebf = 0
        self._tt.clear()
        self._history.clear()

        # Zobrist hash
        self._hash = 0
        for (q, r), p in game.board.items():
            zkey = (q, r, p)
            v = ai_og._zobrist.get(zkey)
            if v is None:
                v = ai_og._zobrist_rng.getrandbits(64)
                ai_og._zobrist[zkey] = v
            self._hash ^= v

        # Score table
        sz = ai_og._WIN_LENGTH + 1
        self._score_table = [[0] * sz for _ in range(sz)]
        for a in range(sz):
            for b in range(sz):
                if self._player == Player.A:
                    my, opp = a, b
                else:
                    my, opp = b, a
                if my > 0 and opp == 0:
                    self._score_table[a][b] = ai_og.LINE_SCORES[my]
                elif opp > 0 and my == 0:
                    self._score_table[a][b] = -int(ai_og.LINE_SCORES[opp] * ai_og._DEF_MULT[opp])

        # Window counts
        self._wc = {}
        board = game.board
        seen = set()
        for (q, r) in board:
            for d_idx, oq, or_ in ai_og._WINDOW_OFFSETS:
                wkey = (d_idx, q - oq, r - or_)
                if wkey in seen:
                    continue
                seen.add(wkey)
                dq, dr = ai_og._DIR_VECTORS[d_idx]
                sq, sr = wkey[1], wkey[2]
                a_count = b_count = 0
                for j in range(ai_og._WIN_LENGTH):
                    cp = board.get((sq + j * dq, sr + j * dr))
                    if cp == Player.A:
                        a_count += 1
                    elif cp == Player.B:
                        b_count += 1
                if a_count > 0 or b_count > 0:
                    self._wc[wkey] = [a_count, b_count]

        self._eval_score = 0
        self._hot_a = set()
        self._hot_b = set()
        st = self._score_table
        for wkey, counts in self._wc.items():
            self._eval_score += st[counts[0]][counts[1]]
            if counts[0] >= 4:
                self._hot_a.add(wkey)
            if counts[1] >= 4:
                self._hot_b.add(wkey)

        # Candidates
        self._cand_refcount = {}
        for (q, r) in board:
            for dq, dr in ai_og._NEIGHBOR_OFFSETS_2:
                nb = (q + dq, r + dr)
                if nb not in board:
                    self._cand_refcount[nb] = self._cand_refcount.get(nb, 0) + 1
        self._cand_set = set(self._cand_refcount)

        if not self._cand_set:
            return [(0, 0)]

        turns = self._generate_turns(game)
        if not turns:
            return [(0, 0)]

        best_move = list(turns[0])
        maximizing = game.current_player == self._player

        for depth in range(1, self._max_depth + 1):
            result, scores = self._search_root(game, turns, depth)
            best_move = list(result)
            self.last_depth = depth
            turns.sort(key=lambda t: scores.get(t, 0), reverse=maximizing)

        return best_move


def make_random_position(rng, num_full_turns):
    """Build a random game position by playing num_full_turns random turns."""
    game = HexGame(win_length=6)
    game.make_move(0, 0)  # Player A's first move

    for _ in range(num_full_turns):
        if game.game_over:
            break
        for _ in range(2):
            if game.game_over:
                break
            candidates = set()
            for q, r in list(game.board):
                for dq in range(-2, 3):
                    for dr in range(-2, 3):
                        ds = -dq - dr
                        if max(abs(dq), abs(dr), abs(ds)) <= 1 and (dq, dr) != (0, 0):
                            nb = (q + dq, r + dr)
                            if nb not in game.board:
                                candidates.add(nb)
            if not candidates:
                break
            q, r = rng.choice(sorted(candidates))
            game.make_move(q, r)

    return game


def copy_game(game):
    """Create an independent copy of a game state."""
    g = HexGame(win_length=game.win_length)
    g.board = dict(game.board)
    g.current_player = game.current_player
    g.moves_left_in_turn = game.moves_left_in_turn
    g.move_count = game.move_count
    g.winner = game.winner
    g.winning_cells = list(game.winning_cells)
    g.game_over = game.game_over
    return g


def normalize_move(move):
    """Normalize a move list for comparison (sort the pair)."""
    if len(move) == 1:
        return tuple(move)
    return (min(move[0], move[1]), max(move[0], move[1]))


def run_tests(num_games=20, seed=12345):
    """Run correctness tests comparing ai.py vs og_ai.py at fixed depth 2."""
    rng = random.Random(seed)

    passed = 0
    failed = 0

    for i in range(num_games):
        num_turns = rng.randint(1, 6)
        game = make_random_position(rng, num_turns)

        if game.game_over:
            continue

        game1 = copy_game(game)
        game2 = copy_game(game)

        # Reset Zobrist tables so both bots get identical random values
        ai_new._zobrist.clear()
        ai_new._zobrist_rng = random.Random(42)
        ai_og._zobrist.clear()
        ai_og._zobrist_rng = random.Random(42)

        bot_new = DepthLimitedNew(max_depth=2)
        bot_og = DepthLimitedOG(max_depth=2)

        move_new = bot_new.get_move(game1)
        move_og = bot_og.get_move(game2)

        norm_new = normalize_move(move_new)
        norm_og = normalize_move(move_og)

        if norm_new == norm_og:
            passed += 1
            print(f"  Test {i+1}: PASS  (stones={game.move_count}, "
                  f"player={game.current_player.name}, move={norm_new})")
        else:
            failed += 1
            print(f"  Test {i+1}: FAIL  (stones={game.move_count}, "
                  f"player={game.current_player.name})")
            print(f"    ai.py:    {norm_new}")
            print(f"    og_ai.py: {norm_og}")
            print(f"    Board: {dict(game.board)}")

    print(f"\nResults: {passed} passed, {failed} failed out of {passed + failed} tests")
    return failed == 0


if __name__ == "__main__":
    print("Correctness test: ai.py vs og_ai.py at fixed depth 2")
    print("=" * 60)
    success = run_tests()
    sys.exit(0 if success else 1)
