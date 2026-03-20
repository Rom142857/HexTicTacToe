"""Test incremental eval during actual minimax search by asserting at every leaf."""
from game import HexGame, Player, HEX_DIRECTIONS
from ai import MinimaxBot, ALL_WINDOWS, CELL_WINDOWS, LINE_SCORES
import random

def full_eval(game, player):
    """Independent full eval for comparison."""
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
                    score -= LINE_SCORES[opp_count]
    return score

mismatches = [0]

def check_search(game, bot, depth, max_children=5):
    """Recursively make/undo like minimax, checking eval at every leaf."""
    if game.game_over or depth == 0:
        expected = full_eval(game, bot._player)
        if bot._eval_score != expected:
            mismatches[0] += 1
            if mismatches[0] <= 3:
                print(f"MISMATCH at depth={depth}: incr={bot._eval_score} full={expected}")
        return

    empty = [p for p, v in game.board.items() if v == Player.NONE]
    random.shuffle(empty)
    for q, r in empty[:max_children]:
        if not game.is_valid_move(q, r):
            continue
        player = game.current_player
        is_my = (player == bot._player)
        state = game.save_state()
        bot._update_eval(q, r, is_my)
        game.make_move(q, r)
        check_search(game, bot, depth - 1, max_children)
        game.undo_move(q, r, state)
        bot._undo_eval(q, r, is_my)
        if mismatches[0] >= 3:
            return

rng = random.Random(99)
random.seed(99)

for trial in range(50):
    g = HexGame()
    bot = MinimaxBot()
    bot._player = Player.A
    bot._init_eval(g)

    # Place a few stones first
    for _ in range(rng.randint(0, 8)):
        empty = [p for p, v in g.board.items() if v == Player.NONE]
        if not empty or g.game_over:
            break
        q, r = rng.choice(empty)
        g.make_move(q, r)
    if g.game_over:
        continue

    bot._init_eval(g)
    check_search(g, bot, depth=4, max_children=4)

    # After full search tree, eval should be back to initial
    expected = full_eval(g, Player.A)
    if bot._eval_score != expected:
        mismatches[0] += 1
        print(f"POST-SEARCH MISMATCH trial={trial}: incr={bot._eval_score} full={expected}")

    if mismatches[0] >= 3:
        break

print(f"Done. {mismatches[0]} mismatches across 50 trials with depth-4 search trees")
