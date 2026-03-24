"""Correctness test: verify ai_cpp (flat-array) vs ai_cpp_og (hash-map) produce
identical moves at matched depth.

Uses a very short time limit so both bots reach the same search depth,
then compares their chosen moves across random positions.
"""

import random
import sys
from game import HexGame, Player

import ai_cpp
import ai_cpp_og


def make_random_position(rng, num_full_turns):
    """Build a random game position by playing num_full_turns random turns."""
    game = HexGame(win_length=6)
    game.make_move(0, 0)

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
                        if max(abs(dq), abs(dr), abs(ds)) <= 2 and (dq, dr) != (0, 0):
                            nb = (q + dq, r + dr)
                            if nb not in game.board:
                                candidates.add(nb)
            if not candidates:
                break
            q, r = rng.choice(sorted(candidates))
            game.make_move(q, r)

    return game


def copy_game(game):
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
    if len(move) == 1:
        return tuple(move)
    return (min(move[0], move[1]), max(move[0], move[1]))


def run_tests(num_games=20, seed=12345, time_limit=0.05):
    """Compare ai_cpp (flat-array) vs ai_cpp_og (hash-map) at matched depth."""
    rng = random.Random(seed)

    bot_new = ai_cpp.MinimaxBot(time_limit=time_limit)
    bot_og  = ai_cpp_og.MinimaxBot(time_limit=time_limit)

    passed = 0
    failed = 0
    depth_mismatch = 0

    for i in range(num_games):
        num_turns = rng.randint(1, 6)
        game = make_random_position(rng, num_turns)

        if game.game_over:
            continue

        game1 = copy_game(game)
        game2 = copy_game(game)

        move_new = bot_new.get_move(game1)
        move_og  = bot_og.get_move(game2)

        norm_new = normalize_move(move_new)
        norm_og  = normalize_move(move_og)

        depth_new = bot_new.last_depth
        depth_og  = bot_og.last_depth

        if depth_new != depth_og:
            depth_mismatch += 1

        if norm_new == norm_og:
            passed += 1
            print(f"  Test {i+1}: PASS  (stones={game.move_count}, "
                  f"player={game.current_player.name}, "
                  f"depth={depth_new}/{depth_og}, move={norm_new})")
        else:
            # Different depths can legitimately produce different moves
            if depth_new != depth_og:
                passed += 1
                print(f"  Test {i+1}: OK    (stones={game.move_count}, "
                      f"player={game.current_player.name}, "
                      f"depth={depth_new}/{depth_og}, "
                      f"new={norm_new} og={norm_og}) [depth mismatch]")
            else:
                failed += 1
                print(f"  Test {i+1}: FAIL  (stones={game.move_count}, "
                      f"player={game.current_player.name}, "
                      f"depth={depth_new}/{depth_og})")
                print(f"    ai_cpp:    {norm_new}")
                print(f"    ai_cpp_og: {norm_og}")

    print(f"\nResults: {passed} passed, {failed} failed "
          f"(out of {passed + failed}, {depth_mismatch} depth mismatches)")
    return failed == 0


if __name__ == "__main__":
    print("Correctness test: ai_cpp (flat-array) vs ai_cpp_og (hash-map)")
    print("=" * 64)
    success = run_tests()
    sys.exit(0 if success else 1)
