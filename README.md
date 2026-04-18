# Hex Tic-Tac-Toe

A hexagonal variant of Tic-Tac-Toe (Connect6 on a hex grid) with an AI opponent.

## Game Rules

- Played on an **infinite hexagonal grid** using axial coordinates.
- **Player A** places **1 stone** on the first turn. After that, players alternate placing **2 stones each turn** (B gets 2, A gets 2, B gets 2, ...). This balances first-move advantage.
- **Win condition**: First player to get **6 in a row** along any of the three hex axes wins.
- The board grows dynamically — there are no fixed boundaries.

## Quick Start

```bash
# Clone the repo
git clone https://github.com/Rom142857/HexTicTacToe.git
cd HexTicTacToe

# Install dependencies
pip install -r requirements.txt

# Play!
python play.py
```

## How to Play

Run `play.py` to launch the game GUI. You play as **Player A** (red) against the **AI** (blue).

**Controls:**
- **Click** on an empty hex cell to place a stone
- **R** — restart the game
- **Q** — quit

## Project Structure

| File | Description |
|------|-------------|
| `play.py` | **Main entry point** — Pygame GUI for playing against the AI |
| `game.py` | Game rules, board state, move/undo, win detection |
| `bot.py` | Abstract `Bot` base class and `RandomBot` |
| `ai.py` | AI bot |

## Requirements

- Python 3.10+
- pygame
- tqdm (for evaluation progress bars)
