"""Pygame interface for Hexagonal Tic-Tac-Toe (infinite grid).

Run this file to play against the AI (you are Player A, AI is Player B).
Controls: Click to place, N to toggle move numbers, E to toggle edit mode, R to restart, Q to quit.
"""

import argparse
import importlib
import os
import pickle
import sys
import math
import time
import pygame
from game import HexGame, Player

# --- Layout ---
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 800
MAX_HEX_SIZE = 28
VISIBLE_DIST = 8  # show empty cells up to this distance from occupied stones

# --- Colors ---
BG_COLOR = (24, 24, 32)
EMPTY_FILL = (48, 48, 58)
GRID_LINE = (72, 72, 85)
PLAYER_A_COLOR = (220, 62, 62)
PLAYER_B_COLOR = (62, 120, 220)
PLAYER_A_HOVER = (120, 40, 40)
WIN_BORDER = (255, 215, 0)
AI_LAST_MOVE = (255, 255, 255)
TEXT_COLOR = (220, 220, 230)
SUBTLE_TEXT = (130, 130, 150)

AI_MOVE_DELAY = 300  # ms between AI stone placements


def _hex_distance(dq, dr):
    return max(abs(dq), abs(dr), abs(dq + dr))


# Precomputed offsets for visible cell generation
_VISIBLE_OFFSETS = tuple(
    (dq, dr)
    for dq in range(-VISIBLE_DIST, VISIBLE_DIST + 1)
    for dr in range(-VISIBLE_DIST, VISIBLE_DIST + 1)
    if _hex_distance(dq, dr) <= VISIBLE_DIST
)


def hex_corners(cx, cy, size):
    """Return 6 corner points for a pointy-top hexagon centered at (cx, cy)."""
    return [
        (cx + size * math.cos(math.radians(60 * i + 30)),
         cy + size * math.sin(math.radians(60 * i + 30)))
        for i in range(6)
    ]


def hex_to_pixel(q, r, size, ox, oy):
    x = size * math.sqrt(3) * (q + r * 0.5) + ox
    y = size * 1.5 * r + oy
    return x, y


def pixel_to_hex(mx, my, size, ox, oy):
    """Convert pixel position to nearest axial hex coordinate."""
    px = (mx - ox) / size
    py = (my - oy) / size
    r_frac = 2.0 / 3 * py
    q_frac = px / math.sqrt(3) - r_frac / 2
    s_frac = -q_frac - r_frac

    rq, rr, rs = round(q_frac), round(r_frac), round(s_frac)
    dq = abs(rq - q_frac)
    dr = abs(rr - r_frac)
    ds = abs(rs - s_frac)

    if dq > dr and dq > ds:
        rq = -rr - rs
    elif dr > ds:
        rr = -rq - rs

    return int(rq), int(rr)


def get_visible_cells(game):
    """Cells to render: occupied + empties within VISIBLE_DIST of occupied."""
    board = game.board
    if not board:
        return {(oq, or_) for oq, or_ in _VISIBLE_OFFSETS}
    cells = set()
    for q, r in board:
        for oq, or_ in _VISIBLE_OFFSETS:
            cells.add((q + oq, r + or_))
    return cells


def compute_view(visible_cells):
    """Compute (hex_size, ox, oy) to fit and center visible cells in the window."""
    if not visible_cells:
        return MAX_HEX_SIZE, WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2

    S3 = math.sqrt(3)
    uxs = [S3 * (q + r * 0.5) for q, r in visible_cells]
    uys = [1.5 * r for q, r in visible_cells]

    min_ux, max_ux = min(uxs), max(uxs)
    min_uy, max_uy = min(uys), max(uys)

    # Extent in unit coords + padding for hex edges
    ext_x = max_ux - min_ux + S3
    ext_y = max_uy - min_uy + 2

    # Available area (leave room for status text top/bottom)
    avail_x = WINDOW_WIDTH - 60
    avail_y = WINDOW_HEIGHT - 140

    size = MAX_HEX_SIZE
    if ext_x > 0:
        size = min(size, avail_x / ext_x)
    if ext_y > 0:
        size = min(size, avail_y / ext_y)
    size = max(8.0, size)

    center_ux = (min_ux + max_ux) / 2
    center_uy = (min_uy + max_uy) / 2
    ox = WINDOW_WIDTH / 2 - center_ux * size
    oy = WINDOW_HEIGHT / 2 - center_uy * size + 20

    return size, ox, oy


EDIT_MODE_COLOR = (255, 180, 40)


def draw_board(screen, game, visible_cells, hover_hex, hex_size, ox, oy, fonts,
               human_player=Player.A, ai_stats=None, last_ai_moves=(),
               edit_mode=False, edit_hover_btn=None,
               show_numbers=False, move_numbers=None,
               save_msg=None, autoplay=False):
    font_big, font_med, font_sm = fonts
    screen.fill(BG_COLOR)

    board = game.board
    human_color = PLAYER_A_COLOR if human_player == Player.A else PLAYER_B_COLOR
    ai_color = PLAYER_B_COLOR if human_player == Player.A else PLAYER_A_COLOR

    # Hex cells
    for (q, r) in visible_cells:
        cx, cy = hex_to_pixel(q, r, hex_size, ox, oy)
        corners = hex_corners(cx, cy, hex_size)

        player = board.get((q, r))
        if player == Player.A:
            fill = PLAYER_A_COLOR
        elif player == Player.B:
            fill = PLAYER_B_COLOR
        elif hover_hex == (q, r) and not game.game_over:
            if edit_mode:
                # Show preview color based on which mouse button
                if edit_hover_btn == 3:
                    fill = tuple(c // 2 for c in PLAYER_B_COLOR)
                else:
                    fill = tuple(c // 2 for c in PLAYER_A_COLOR)
            else:
                fill = PLAYER_A_HOVER
        else:
            fill = EMPTY_FILL

        pygame.draw.polygon(screen, fill, corners)
        pygame.draw.polygon(screen, GRID_LINE, corners, 2)

    # AI last move highlight
    for (q, r) in last_ai_moves:
        cx, cy = hex_to_pixel(q, r, hex_size, ox, oy)
        corners = hex_corners(cx, cy, hex_size)
        pygame.draw.polygon(screen, AI_LAST_MOVE, corners, 3)

    # Winning cells highlight (drawn on top)
    for (q, r) in game.winning_cells:
        cx, cy = hex_to_pixel(q, r, hex_size, ox, oy)
        corners = hex_corners(cx, cy, hex_size)
        pygame.draw.polygon(screen, WIN_BORDER, corners, 3)

    # Move numbers overlay
    if show_numbers and move_numbers:
        num_font = pygame.font.SysFont("Arial", max(10, int(hex_size * 0.6)), bold=True)
        for (q, r), turn_num in move_numbers.items():
            if (q, r) in game.board:
                cx, cy = hex_to_pixel(q, r, hex_size, ox, oy)
                label = num_font.render(str(turn_num), True, (255, 255, 255))
                shadow = num_font.render(str(turn_num), True, (0, 0, 0))
                rect = label.get_rect(center=(cx, cy))
                screen.blit(shadow, shadow.get_rect(center=(cx + 1, cy + 1)))
                screen.blit(label, rect)

    # Status text
    if edit_mode:
        status = font_big.render("EDIT MODE", True, EDIT_MODE_COLOR)
        screen.blit(status, status.get_rect(centerx=WINDOW_WIDTH // 2, y=20))
        edit_hint = font_med.render(
            "Left click = Red  |  Right click = Blue  |  Click same to remove",
            True, SUBTLE_TEXT,
        )
        screen.blit(edit_hint, edit_hint.get_rect(centerx=WINDOW_WIDTH // 2, y=55))
    elif game.winner != Player.NONE:
        name = "You win!" if game.winner == human_player else "AI wins!"
        color = human_color if game.winner == human_player else ai_color
        status = font_big.render(name, True, color)
        screen.blit(status, status.get_rect(centerx=WINDOW_WIDTH // 2, y=20))
    elif game.game_over:
        status = font_big.render("Draw!", True, TEXT_COLOR)
        screen.blit(status, status.get_rect(centerx=WINDOW_WIDTH // 2, y=20))
    elif autoplay:
        color = PLAYER_A_COLOR if game.current_player == Player.A else PLAYER_B_COLOR
        status = font_big.render("AI vs AI", True, color)
        screen.blit(status, status.get_rect(centerx=WINDOW_WIDTH // 2, y=20))
    elif game.current_player != human_player:
        status = font_big.render("AI is thinking...", True, ai_color)
        screen.blit(status, status.get_rect(centerx=WINDOW_WIDTH // 2, y=20))
    else:
        status = font_big.render("Your turn", True, human_color)
        screen.blit(status, status.get_rect(centerx=WINDOW_WIDTH // 2, y=20))
        moves_surf = font_med.render(
            f"Moves left: {game.moves_left_in_turn}", True, SUBTLE_TEXT
        )
        screen.blit(moves_surf, moves_surf.get_rect(centerx=WINDOW_WIDTH // 2, y=55))

    if ai_stats:
        depth, nodes, score = ai_stats
        ai_info = font_sm.render(f"AI: depth {depth}, {nodes:,} nodes, eval {score:+,}",
                                 True, SUBTLE_TEXT)
        screen.blit(ai_info, ai_info.get_rect(centerx=WINDOW_WIDTH // 2, y=WINDOW_HEIGHT - 50))

    if edit_mode:
        instr = font_sm.render("E = exit edit  |  S = save  |  R = restart  |  Q = quit", True, SUBTLE_TEXT)
    else:
        instr = font_sm.render(
            "SPACE = swap  |  A = autoplay  |  N = numbers  |  S = save  |  E = edit  |  R = restart  |  Q = quit",
            True, SUBTLE_TEXT,
        )
    screen.blit(instr, instr.get_rect(centerx=WINDOW_WIDTH // 2, y=WINDOW_HEIGHT - 30))

    if save_msg:
        msg_surf = font_med.render(save_msg, True, (100, 255, 100))
        screen.blit(msg_surf, msg_surf.get_rect(centerx=WINDOW_WIDTH // 2, y=WINDOW_HEIGHT - 70))

    pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(description="Play Hex Tic-Tac-Toe against an AI bot.")
    parser.add_argument("bot", nargs="?", default="og_ai",
                        help="AI module name (default: og_ai)")
    parser.add_argument("--time-limit", type=float, default=0.5,
                        help="AI time limit per move in seconds (default: 0.5)")
    args = parser.parse_args()

    bot_mod = importlib.import_module(args.bot)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Hex Tic-Tac-Toe \u2014 6 in a Row")
    clock = pygame.time.Clock()

    fonts = (
        pygame.font.SysFont("Arial", 28, bold=True),
        pygame.font.SysFont("Arial", 20),
        pygame.font.SysFont("Arial", 16),
    )

    game = HexGame(win_length=6)
    ai = bot_mod.MinimaxBot(time_limit=args.time_limit)

    human_player = Player.A
    hover_hex = None
    last_ai_time = 0
    ai_stats = None
    last_ai_moves = ()
    edit_mode = False
    edit_hover_btn = 1  # track last mouse button for hover preview
    show_numbers = False
    move_numbers = {}  # (q, r) -> turn_number
    turn_number = 0
    save_msg = None
    save_msg_time = 0
    autoplay = False

    while True:
        now = pygame.time.get_ticks()

        visible_cells = get_visible_cells(game)
        hex_size, ox, oy = compute_view(visible_cells)

        # Expire save message after 2 seconds
        if save_msg and now - save_msg_time > 2000:
            save_msg = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEMOTION:
                q, r = pixel_to_hex(*event.pos, hex_size, ox, oy)
                if edit_mode:
                    if (q, r) in visible_cells:
                        hover_hex = (q, r)
                    else:
                        hover_hex = None
                elif game.current_player == human_player and not game.game_over:
                    if (q, r) in visible_cells and game.is_valid_move(q, r):
                        hover_hex = (q, r)
                    else:
                        hover_hex = None

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if edit_mode:
                    q, r = pixel_to_hex(*event.pos, hex_size, ox, oy)
                    if event.button == 1:  # Left click = Red (Player A)
                        if game.board.get((q, r)) == Player.A:
                            del game.board[(q, r)]
                        else:
                            game.board[(q, r)] = Player.A
                    elif event.button == 3:  # Right click = Blue (Player B)
                        if game.board.get((q, r)) == Player.B:
                            del game.board[(q, r)]
                        else:
                            game.board[(q, r)] = Player.B
                    edit_hover_btn = event.button
                elif event.button == 1:
                    if game.current_player == human_player and not game.game_over:
                        q, r = pixel_to_hex(*event.pos, hex_size, ox, oy)
                        if (q, r) in visible_cells and game.is_valid_move(q, r):
                            new_turn = game.moves_left_in_turn == 2 or game.move_count == 0
                            if game.make_move(q, r):
                                if new_turn:
                                    turn_number += 1
                                move_numbers[(q, r)] = turn_number
                                hover_hex = None
                            last_ai_time = now

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_e:
                    edit_mode = not edit_mode
                    if not edit_mode:
                        # Exiting edit mode: Red always goes next, AI plays
                        game.current_player = Player.A
                        game.moves_left_in_turn = 1 if not game.board else 2
                        game.move_count = len(game.board)
                        game.winner = Player.NONE
                        game.winning_cells = []
                        game.game_over = False
                        human_player = Player.B  # AI is now Red, will move next
                        last_ai_time = now
                        ai_stats = None
                        last_ai_moves = ()
                        move_numbers = {}
                        turn_number = 0
                    hover_hex = None
                elif event.key == pygame.K_n:
                    show_numbers = not show_numbers
                elif event.key == pygame.K_s:
                    # Save current position to pickle file
                    save_path = os.path.join(os.path.dirname(__file__), "saved_positions.pkl")
                    positions = []
                    if os.path.exists(save_path):
                        with open(save_path, "rb") as f:
                            positions = pickle.load(f)
                    uid = f"play_{int(time.time())}_{len(positions)}"
                    save_player = Player.A if edit_mode else game.current_player
                    positions.append((dict(game.board), save_player, 0, uid))
                    with open(save_path, "wb") as f:
                        pickle.dump(positions, f)
                    save_msg = f"Position saved ({len(positions)} total)"
                    save_msg_time = now
                elif event.key == pygame.K_r:
                    game.reset()
                    human_player = Player.A
                    hover_hex = None
                    ai_stats = None
                    last_ai_moves = ()
                    edit_mode = False
                    move_numbers = {}
                    turn_number = 0
                    autoplay = False
                elif event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_a and not edit_mode and not game.game_over:
                    autoplay = not autoplay
                    last_ai_time = now
                elif event.key == pygame.K_SPACE and not game.game_over and not edit_mode:
                    # Swap sides
                    human_player = Player.B if human_player == Player.A else Player.A
                    last_ai_time = now
                    hover_hex = None

        # AI turn
        if ((autoplay or game.current_player != human_player) and not game.game_over
                and now - last_ai_time >= AI_MOVE_DELAY):
            draw_board(screen, game, visible_cells, None, hex_size, ox, oy, fonts,
                       human_player, ai_stats, last_ai_moves,
                       edit_mode=edit_mode, edit_hover_btn=edit_hover_btn,
                       show_numbers=show_numbers, move_numbers=move_numbers,
                       save_msg=save_msg, autoplay=autoplay)
            result = ai.get_move(game)
            last_ai_moves = tuple(tuple(m) for m in result) if ai.pair_moves else (tuple(result),)
            turn_number += 1
            if ai.pair_moves:
                for q, r in result:
                    if not game.game_over:
                        game.make_move(q, r)
                        move_numbers[(q, r)] = turn_number
            else:
                game.make_move(*result)
                move_numbers[tuple(result)] = turn_number
            ai_stats = (ai.last_depth, ai._nodes, ai.last_score)
            last_ai_time = pygame.time.get_ticks()

        draw_board(screen, game, visible_cells, hover_hex, hex_size, ox, oy, fonts,
                   human_player, ai_stats, last_ai_moves,
                   edit_mode=edit_mode, edit_hover_btn=edit_hover_btn,
                   show_numbers=show_numbers, move_numbers=move_numbers,
                   save_msg=save_msg, autoplay=autoplay)
        clock.tick(60)


if __name__ == "__main__":
    main()
