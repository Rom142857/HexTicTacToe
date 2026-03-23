"""Pygame puzzle viewer for forced-win positions.

Loads puzzles from data/forced_wins.json and presents them one at a time.
You play as the attacker — find the winning setup move(s).

Controls:
    Click       Place a stone (only correct moves accepted)
    G           Give up — show the full solution
    N / Right   Skip to next puzzle
    P / Left    Previous puzzle
    R           Reset current puzzle
    Q           Quit
"""

import json
import math
import os
import sys

import pygame
from game import HexGame, Player

# --- Layout ---
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 800
MAX_HEX_SIZE = 28
VISIBLE_DIST = 8

# --- Colors ---
BG_COLOR = (24, 24, 32)
EMPTY_FILL = (48, 48, 58)
GRID_LINE = (72, 72, 85)
PLAYER_A_COLOR = (220, 62, 62)
PLAYER_B_COLOR = (62, 120, 220)
HOVER_VALID = (80, 160, 80)
HOVER_INVALID = (100, 50, 50)
WIN_BORDER = (255, 215, 0)
HINT_COLOR = (100, 220, 100)
SOLUTION_COLOR = (255, 255, 255)
TEXT_COLOR = (220, 220, 230)
SUBTLE_TEXT = (130, 130, 150)
SUCCESS_COLOR = (80, 220, 80)
FAIL_COLOR = (220, 80, 80)

DEFENDER_DELAY = 400  # ms before auto-playing defender's response


def _hex_distance(dq, dr):
    return max(abs(dq), abs(dr), abs(dq + dr))


_VISIBLE_OFFSETS = tuple(
    (dq, dr)
    for dq in range(-VISIBLE_DIST, VISIBLE_DIST + 1)
    for dr in range(-VISIBLE_DIST, VISIBLE_DIST + 1)
    if _hex_distance(dq, dr) <= VISIBLE_DIST
)


def hex_corners(cx, cy, size):
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


def get_visible_cells(board):
    if not board:
        return {(oq, or_) for oq, or_ in _VISIBLE_OFFSETS}
    cells = set()
    for q, r in board:
        for oq, or_ in _VISIBLE_OFFSETS:
            cells.add((q + oq, r + or_))
    return cells


def compute_view(visible_cells):
    if not visible_cells:
        return MAX_HEX_SIZE, WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2
    S3 = math.sqrt(3)
    uxs = [S3 * (q + r * 0.5) for q, r in visible_cells]
    uys = [1.5 * r for q, r in visible_cells]
    min_ux, max_ux = min(uxs), max(uxs)
    min_uy, max_uy = min(uys), max(uys)
    ext_x = max_ux - min_ux + S3
    ext_y = max_uy - min_uy + 2
    avail_x = WINDOW_WIDTH - 60
    avail_y = WINDOW_HEIGHT - 160
    size = MAX_HEX_SIZE
    if ext_x > 0:
        size = min(size, avail_x / ext_x)
    if ext_y > 0:
        size = min(size, avail_y / ext_y)
    size = max(8.0, size)
    ox = WINDOW_WIDTH / 2 - (min_ux + max_ux) / 2 * size
    oy = WINDOW_HEIGHT / 2 - (min_uy + max_uy) / 2 * size + 20
    return size, ox, oy


# ---------------------------------------------------------------------------
# Puzzle state
# ---------------------------------------------------------------------------

class Puzzle:
    def __init__(self, data):
        self.data = data
        self.reset()

    def reset(self):
        # Parse board: "q,r" -> Player
        self.board = {}
        for key, val in self.data["board"].items():
            q, r = map(int, key.split(","))
            self.board[(q, r)] = Player.A if val == "A" else Player.B

        self.to_move = Player.A if self.data["to_move"] == "A" else Player.B
        self.winner_player = Player.A if self.data["winner"] == "A" else Player.B

        # Parse solution sequence
        self.sequence = []
        for turn in self.data["sequence"]:
            player = Player.A if turn["player"] == "A" else Player.B
            moves = [tuple(m) for m in turn["moves"]]
            self.sequence.append((player, moves))

        # Attacker is the winner; defender is the other
        self.attacker = self.winner_player
        self.defender = Player.B if self.attacker == Player.A else Player.A

        # Game state for the puzzle
        self.placed = []          # stones placed this turn
        self.turn_idx = 0         # which turn in the sequence we're on
        self.solved = False
        self.gave_up = False
        self.show_solution = False
        self.solution_highlights = []  # cells to highlight as solution
        self.wrong_flash = 0      # ticks remaining for wrong-move flash
        self.defender_pending = 0  # tick when defender auto-plays

    @property
    def current_expected(self):
        """The moves expected for the current turn in the sequence."""
        if self.turn_idx < len(self.sequence):
            return self.sequence[self.turn_idx]
        return None

    @property
    def is_attacker_turn(self):
        expected = self.current_expected
        if expected is None:
            return False
        return expected[0] == self.attacker

    @property
    def moves_left(self):
        expected = self.current_expected
        if expected is None:
            return 0
        return len(expected[1]) - len(self.placed)

    def try_place(self, q, r):
        """Try to place a stone. Returns True if correct, False if wrong."""
        expected = self.current_expected
        if expected is None or self.solved or self.gave_up:
            return False

        player, moves = expected
        if player != self.attacker:
            return False  # not attacker's turn

        move_idx = len(self.placed)
        if move_idx >= len(moves):
            return False

        # Check if this move is correct
        expected_move = tuple(moves[move_idx])
        if (q, r) != expected_move:
            # Check if moves can be in either order (both are the attacker's)
            if move_idx == 0 and len(moves) == 2 and (q, r) == tuple(moves[1]):
                # Swap the expected order
                moves[0], moves[1] = moves[1], moves[0]
                expected_move = tuple(moves[0])

        if (q, r) != expected_move:
            self.wrong_flash = 30
            return False

        self.board[(q, r)] = player
        self.placed.append((q, r))

        # Check if turn is complete
        if len(self.placed) >= len(moves):
            self._advance_turn()

        return True

    def _advance_turn(self):
        """Move to the next turn in the sequence."""
        self.placed = []
        self.turn_idx += 1

        if self.turn_idx >= len(self.sequence):
            self.solved = True
            return

        # If next turn is defender's, schedule auto-play
        expected = self.current_expected
        if expected and expected[0] == self.defender:
            self.defender_pending = pygame.time.get_ticks() + DEFENDER_DELAY

    def auto_play_defender(self):
        """Play the defender's response."""
        expected = self.current_expected
        if expected is None:
            return
        player, moves = expected
        for q, r in moves:
            self.board[(q, r)] = player
        self.solution_highlights = []
        self._advance_turn()

    def give_up(self):
        """Show the full remaining solution."""
        self.gave_up = True
        self.show_solution = True
        # Gather all remaining moves
        self.solution_highlights = []
        for i in range(self.turn_idx, len(self.sequence)):
            player, moves = self.sequence[i]
            for q, r in moves:
                if (q, r) not in self.board:
                    self.solution_highlights.append(((q, r), player))
                    self.board[(q, r)] = player


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_puzzle(screen, puzzle, visible_cells, hover_hex, hex_size, ox, oy,
                fonts, puzzle_idx, total_puzzles):
    font_big, font_med, font_sm = fonts
    screen.fill(BG_COLOR)

    board = puzzle.board
    attacker_color = PLAYER_A_COLOR if puzzle.attacker == Player.A else PLAYER_B_COLOR
    defender_color = PLAYER_B_COLOR if puzzle.attacker == Player.A else PLAYER_A_COLOR

    # Gather solution highlight cells for quick lookup
    solution_cells = {}
    if puzzle.show_solution:
        for (q, r), player in puzzle.solution_highlights:
            solution_cells[(q, r)] = player

    # Draw hex cells
    for (q, r) in visible_cells:
        cx, cy = hex_to_pixel(q, r, hex_size, ox, oy)
        corners = hex_corners(cx, cy, hex_size)

        player = board.get((q, r))
        if player == Player.A:
            fill = PLAYER_A_COLOR
        elif player == Player.B:
            fill = PLAYER_B_COLOR
        elif hover_hex == (q, r) and puzzle.is_attacker_turn and not puzzle.solved and not puzzle.gave_up:
            fill = HOVER_VALID
        else:
            fill = EMPTY_FILL

        pygame.draw.polygon(screen, fill, corners)
        pygame.draw.polygon(screen, GRID_LINE, corners, 2)

    # Highlight placed stones this turn
    for (q, r) in puzzle.placed:
        cx, cy = hex_to_pixel(q, r, hex_size, ox, oy)
        corners = hex_corners(cx, cy, hex_size)
        pygame.draw.polygon(screen, HINT_COLOR, corners, 3)

    # Highlight solution moves
    if puzzle.show_solution:
        for (q, r), player in puzzle.solution_highlights:
            cx, cy = hex_to_pixel(q, r, hex_size, ox, oy)
            corners = hex_corners(cx, cy, hex_size)
            pygame.draw.polygon(screen, SOLUTION_COLOR, corners, 3)

    # Wrong move flash
    if puzzle.wrong_flash > 0:
        puzzle.wrong_flash -= 1
        flash_surf = pygame.Surface((WINDOW_WIDTH, 6))
        flash_surf.fill(FAIL_COLOR)
        screen.blit(flash_surf, (0, 75))

    # Status text
    y = 15
    header = f"Puzzle {puzzle_idx + 1} / {total_puzzles}"
    screen.blit(font_med.render(header, True, SUBTLE_TEXT),
                (WINDOW_WIDTH // 2 - font_med.size(header)[0] // 2, y))
    y += 30

    if puzzle.solved:
        status = font_big.render("Solved!", True, SUCCESS_COLOR)
    elif puzzle.gave_up:
        status = font_big.render("Solution shown", True, FAIL_COLOR)
    elif puzzle.is_attacker_turn:
        moves = puzzle.moves_left
        label = f"Your turn — place {moves} stone{'s' if moves != 1 else ''}"
        status = font_big.render(label, True, attacker_color)
    else:
        status = font_big.render("Opponent responding...", True, defender_color)

    screen.blit(status, status.get_rect(centerx=WINDOW_WIDTH // 2, y=y))

    # You play as...
    y += 40
    role = f"You are {'Red (A)' if puzzle.attacker == Player.A else 'Blue (B)'} — find the winning setup"
    screen.blit(font_sm.render(role, True, SUBTLE_TEXT),
                (WINDOW_WIDTH // 2 - font_sm.size(role)[0] // 2, y))

    # Info bar
    info = f"{puzzle.data['num_stones']} stones | {len(puzzle.sequence)} turns to win"
    screen.blit(font_sm.render(info, True, SUBTLE_TEXT),
                (WINDOW_WIDTH // 2 - font_sm.size(info)[0] // 2, WINDOW_HEIGHT - 55))

    instr = "Click = place  |  G = give up  |  N/Right = next  |  P/Left = prev  |  R = reset  |  Q = quit"
    screen.blit(font_sm.render(instr, True, SUBTLE_TEXT),
                (WINDOW_WIDTH // 2 - font_sm.size(instr)[0] // 2, WINDOW_HEIGHT - 30))

    pygame.display.flip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_path = os.path.join(os.path.dirname(__file__), "data", "forced_wins.json")
    if not os.path.exists(data_path):
        print(f"No puzzle data found at {data_path}")
        print("Run find_forced_wins.py first.")
        sys.exit(1)

    with open(data_path) as f:
        puzzles_data = json.load(f)

    if not puzzles_data:
        print("No puzzles in data file.")
        sys.exit(1)

    print(f"Loaded {len(puzzles_data)} puzzles")

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Hex Tic-Tac-Toe Puzzles")
    clock = pygame.time.Clock()

    fonts = (
        pygame.font.SysFont("Arial", 28, bold=True),
        pygame.font.SysFont("Arial", 20),
        pygame.font.SysFont("Arial", 16),
    )

    puzzle_idx = 0
    puzzle = Puzzle(puzzles_data[puzzle_idx])
    hover_hex = None

    while True:
        now = pygame.time.get_ticks()

        visible_cells = get_visible_cells(puzzle.board)
        hex_size, ox, oy = compute_view(visible_cells)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEMOTION:
                if puzzle.is_attacker_turn and not puzzle.solved and not puzzle.gave_up:
                    q, r = pixel_to_hex(*event.pos, hex_size, ox, oy)
                    if (q, r) in visible_cells and (q, r) not in puzzle.board:
                        hover_hex = (q, r)
                    else:
                        hover_hex = None
                else:
                    hover_hex = None

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if puzzle.is_attacker_turn and not puzzle.solved and not puzzle.gave_up:
                    q, r = pixel_to_hex(*event.pos, hex_size, ox, oy)
                    if (q, r) in visible_cells:
                        puzzle.try_place(q, r)
                        hover_hex = None

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_g:
                    if not puzzle.solved:
                        puzzle.give_up()
                elif event.key in (pygame.K_n, pygame.K_RIGHT):
                    puzzle_idx = (puzzle_idx + 1) % len(puzzles_data)
                    puzzle = Puzzle(puzzles_data[puzzle_idx])
                    hover_hex = None
                elif event.key in (pygame.K_p, pygame.K_LEFT):
                    puzzle_idx = (puzzle_idx - 1) % len(puzzles_data)
                    puzzle = Puzzle(puzzles_data[puzzle_idx])
                    hover_hex = None
                elif event.key == pygame.K_r:
                    puzzle = Puzzle(puzzles_data[puzzle_idx])
                    hover_hex = None

        # Auto-play defender response after delay
        if (puzzle.defender_pending > 0
                and now >= puzzle.defender_pending
                and not puzzle.solved and not puzzle.gave_up):
            puzzle.defender_pending = 0
            puzzle.auto_play_defender()

        draw_puzzle(screen, puzzle, visible_cells, hover_hex, hex_size, ox, oy,
                    fonts, puzzle_idx, len(puzzles_data))
        clock.tick(60)


if __name__ == "__main__":
    main()
