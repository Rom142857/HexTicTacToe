# autoresearch — Hex Tic-Tac-Toe Bot Optimization

This is an experiment to have the LLM autonomously improve a game-playing bot.

## Working directory

You are launched from `autoresearch/`, but all game files (`ai.py`, `game.py`, `evaluate.py`, etc.) live in the parent directory.

**Command rules:**
- **Python/shell commands**: Use `../` paths instead of `cd .. &&`. For example: `python ../run_eval.py`, `tail ../ai.py`, `cp ../ai.py ../og_ai.py`.
- **Git commands**: Run them as-is from `autoresearch/` — git works from any subdirectory. Never prefix git commands with `cd`.
- **NEVER use `cd .. &&` or `cd ../` prefixes** for any command.
- **Ad-hoc testing**: Never use `python -c "..."` for inline scripts. Instead, write your code to `../test.py` and run `python ../test.py`. Delete it when done.
- File edits (Edit/Write/Read tools) use absolute paths and don't need any of this.

## Game rules

Hex Tic-Tac-Toe is played on a hexagonal board of infinite size using axial coordinates. Two players (A and B) take turns placing stones:

- **Turn order**: Player A places **1** stone first. After that, players alternate placing **2** stones each (B gets 2, then A gets 2, then B gets 2, ...). The 1-then-2-2-2 structure balances first-move advantage.
- **Win condition**: First player to get **6 in a row** along any of the three hex axes wins.
- **Draw**: If the board fills with no winner, it's a draw.

The board uses axial coordinates `(q, r)` with implicit `s = -q - r`. The three line directions are `(1,0)`, `(0,1)`, and `(1,-1)`.

This is essentially **Connect6 on a hex grid** — a well-studied game family. Literature on Connect6 strategy, threat-space search, and hex-grid evaluation may be useful.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar19`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current HEAD.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `game.py` — game rules, board representation, move/undo. **Read-only.**
   - `bot.py` — `Bot` base class and `RandomBot`. **Read-only.**
   - `evaluate.py` — plays two bots head-to-head, reports win rates and depth stats. The `evaluate()` function controls the time limit for both bots via `time_limit=` (default 1s). **Read-only.**
   - `ai.py` — **the file you modify.** Contains the bot implementation (search, heuristics, candidate generation, etc).
4. **Snapshot the champion**: Copy `ai.py` to `og_ai.py`. This is the current best bot — the one you must beat.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first evaluation.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment is an edit to `ai.py` followed by a correctness and speed test for time based changes, and always a head-to-head evaluation against the champion (`og_ai.py`) to make sure everything works. Evaluation runs **400 games** (sides swapped each game) with a **0.1s time limit per move** for both bots.

You launch an evaluation like this:

```bash
python ../run_eval.py
```

If your bot class has been renamed, pass it as an argument:

```bash
python ../run_eval.py --new-class MyBot
```

Full options: `--new-module`, `--new-class`, `--old-module`, `--old-class`, `--games`, `--time-limit`. Defaults are `ai.MinimaxBot` vs `og_ai.MinimaxBot`, 400 games, 0.1s time limit.

`og_ai.py` must always be importable — never modify it directly.

**What you CAN do:**
- Modify `ai.py` — this is the only file you edit.
- **Speed optimizations**: Make the bot faster so it searches deeper in the same time. These must produce identical output to the current champion — run `python ../test_correctness.py` to verify before evaluating.
- **Hyperparameter tuning**: Adjust numeric values (LINE_SCORES, defensive multipliers, candidate caps, etc.) to find better settings.

**What you CANNOT do:**
- **Change the architecture.** Do not add new search algorithms, new heuristic systems, new data structures for the search, or fundamentally restructure the bot. The architecture is locked — only speed and tuning.
- Modify `game.py`, `bot.py`, or `evaluate.py`. They are read-only.
- Modify `og_ai.py` directly. It is only updated by copying `ai.py` over it when a new champion is crowned.
- Install new packages or add dependencies.
- Change the time limit. Both bots always get **0.1s per move**.

## Two optimization tracks

### Track 1: Speed optimizations
The goal is to make the existing code run faster so the bot searches deeper within the same time limit. Speed optimizations must **not change the bot's decisions** — they must produce identical moves to the champion at the same depth.

**Workflow for speed changes:**
1. Make the speed optimization in `ai.py`
2. Run `python ../test_correctness.py` to verify identical output to `og_ai.py`
3. Run `python ../test_profile.py --competition` to run ai.py vs og_ai.py head-to-head and confirm the speed change actually helps
4. If correctness passes, run `python ../run_eval.py` to measure improvement via full evaluation
5. The win should come from deeper search, not different evaluation

Check `autoresearch/ideas.md` for speed ideas to try.

### Track 2: Hyperparameter tuning
The goal is to find optimal values for existing numeric parameters.

**Tuning strategy:**
1. Pick a parameter to tune (e.g., a LINE_SCORE value, a defensive multiplier, a candidate cap)
2. Try changing it in one direction (e.g., increase by ~20-50%)
3. If that fails (win rate < 55%), try the other direction (decrease)
4. If both directions fail, try smaller step sizes
5. If small steps also fail, conclude the current value is at or near optimal and move on
6. If a direction works, keep pushing further in that direction until it stops helping

**The goal is simple: achieve the highest win rate against the previous champion** through speed optimizations and hyperparameter tuning only. The architecture is locked — no new algorithms, heuristics, or structural changes.

**og_ai.py is your safeguard**: `og_ai.py` always holds the current best bot. Never modify it directly — only update it by copying `ai.py` over it when a new champion is crowned. This ensures you always have a known-good baseline to compare against and revert to.

**Simplicity criterion**: All else being equal, simpler is better. A marginal speed improvement that adds ugly complexity is not worth it. Removing something and getting equal or better speed is a great outcome. Weigh the complexity cost against the improvement magnitude.

**One evaluation at a time**: Evaluations use all CPU cores via multiprocessing. Never run multiple evaluations in parallel — always wait for one to finish before starting the next, even during parameter sweeps.

**Parameter tuning**: When a numeric change improves win rate, always explore the neighborhood. Push the value further — if that helps, keep going. Then try a smaller change in the same direction. Squeeze out the gains before moving on. Experiments are cheap (~2 minutes each), so spend 2–3 extra runs optimizing a value that's clearly helping. Don't sweep unimportant parameters, but always optimize when you've found a good direction.

**Ideas tracking**: Check `autoresearch/ideas.md` at the start and periodically during the loop. Use it to keep track of overarching ideas, promising directions, and things to try next. Update it as you go — add new ideas that occur to you, mark ones you've tried, note which worked and which didn't. This prevents losing track of good ideas across many iterations.

## Output format

The evaluation prints a summary like this:

```
==================================================
  MinimaxBot vs MinimaxBot  —  200 games in 25.3s
==================================================
      MinimaxBot:  120 wins (60%)
      MinimaxBot:   75 wins (38%)
          Draws:    5      (2%)

  MinimaxBot search depth: avg 3.2, range [1-6]
    d1:5  d2:40  d3:80  d4:50  d5:20  d6:5
  MinimaxBot search depth: avg 2.8, range [1-5]
    d1:8  d2:50  d3:70  d4:45  d5:12
==================================================
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	win_rate	avg_depth	status	description
```

1. git commit hash (short, 7 chars)
2. win rate of new bot vs champion (e.g. 0.60 for 60%) — use 0.00 for crashes
3. average search depth achieved (e.g. 3.2) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	win_rate	avg_depth	status	description
a1b2c3d	0.50	2.0	keep	baseline (og vs og)
b2c3d4e	0.62	3.1	keep	add threat-counting heuristic
c3d4e5f	0.45	4.2	discard	MCTS with random rollouts
d4e5f6g	0.00	0.0	crash	syntax error in eval function
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar19`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Edit `ai.py` with an experimental idea.
3. git commit `ai.py`.
4. Run the evaluation: `python ../run_eval.py` — output goes straight into context, no log file needed.
5. If the output is empty or shows a traceback, the run crashed. Attempt a fix. If you can't get things to work after a few attempts, give up on this idea.
7. Record the results in the tsv. (NOTE: do not commit results.tsv — leave it untracked.)
8. **Decide whether to keep or discard.** The threshold depends on the type of change:
   - **Speed optimizations**: Be forgiving — **45%+ win rate is fine**, but only if `test_correctness.py` passes AND `test_profile.py --competition` shows a clear speed improvement. If either of those don't confirm the change, or win rate is below 45%, something may have gone wrong; investigate or revert.
   - **Hyperparameter tuning**: A win rate around **55%+** is a clear improvement — keep it. But context matters:
     - A simplification that wins 51%? Keep it — simpler code at equal strength is a win.
     - A complex change that wins 53%? Probably noise — discard unless you have strong reason to believe it's real.
     - When in doubt, 55% is a good rule of thumb for "meaningfully stronger."
9. **If keeping**: the new bot is champion.
   - Copy `ai.py` to `og_ai.py` (replacing the old champion).
   - git commit `og_ai.py` with a message like "promote: <description>".
   - This is now the new baseline to beat.
10. **If discarding**:
   - `git checkout ai.py` to revert back to the current champion's code.

**One change per experiment**: Each commit/experiment should test exactly one thing. Don't bundle multiple independent changes (e.g. new heuristic + move ordering + candidate narrowing) into a single experiment — it's impossible to tell what helped. Change one variable at a time so results are attributable.

**Crashes**: If a run crashes, use your judgment. If it's a typo or simple bug, fix and re-run. If the idea is fundamentally broken, log it as `crash`, revert, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away and expects you to continue working *indefinitely* until manually stopped. You are autonomous. If you run out of ideas, think harder — re-read the game logic for new angles, research hex game heuristics, try combining previous near-misses, try radically different approaches. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. Each experiment takes ~5 minutes to evaluate, so you can run many iterations per hour. The user wakes up to a much stronger bot, all improved autonomously while they slept.
