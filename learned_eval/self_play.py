"""Batched self-play game generation for MCTS training.

Runs 256 games in lockstep: all slots search simultaneously, batch NN evals
on GPU, then advance games together. Finished games are replaced until
n_games total are complete.

Output: list of training examples saved as parquet.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from tqdm import tqdm

from game import HexGame, Player
from learned_eval.mcts import (
    MCTSTree, create_tree, create_trees_batched, select_leaf,
    expand_and_backprop, get_pair_visits, select_move_pair,
    select_single_move, _idx_to_cell,
)
from learned_eval.resnet_model import board_to_planes

MAX_GAME_MOVES = 100


@dataclass
class SelfPlaySlot:
    game: HexGame
    tree: MCTSTree | None = None
    sims_done: int = 0
    turn_number: int = 0
    game_id: int = 0
    examples: list[dict] = field(default_factory=list)


class SelfPlayManager:
    def __init__(self, model, device, n_games=5000, batch_size=256, n_sims=200):
        self.model = model
        self.device = device
        self.n_games = n_games
        self.batch_size = batch_size
        self.n_sims = n_sims

    def generate(self, round_id: int) -> list[dict]:
        """Generate n_games of self-play data. Returns list of example dicts."""
        model = self.model
        device = self.device
        n_sims = self.n_sims
        batch_size = self.batch_size

        all_examples: list[dict] = []
        games_completed = 0
        total_positions = 0
        total_moves_in_completed = 0
        max_h = 0
        max_w = 0
        next_game_id = 0

        # Initialize slots
        slots: list[SelfPlaySlot] = []
        for _ in range(batch_size):
            slot = self._new_slot(next_game_id)
            next_game_id += 1
            slots.append(slot)

        pbar = tqdm(total=self.n_games, desc="Games", unit="game", position=0)
        pos_bar = tqdm(desc="Positions", unit="pos", position=1)

        # Timing accumulators (seconds)
        t_tree_create = 0.0
        t_select = 0.0
        t_collect = 0.0
        t_batch_eval = 0.0
        t_backprop = 0.0
        t_move = 0.0
        n_turns = 0
        # Sub-timers inside _batch_eval
        self._t_b2p = 0.0      # board_to_planes
        self._t_pad = 0.0      # pad + stack
        self._t_forward = 0.0  # transfer + model forward + transfer back

        while games_completed < self.n_games:
            # --- Phase 1: Create trees for slots that need them ---
            needs_tree = [i for i, s in enumerate(slots) if s.tree is None]
            if needs_tree:
                _t0 = time.monotonic()
                self._batch_create_trees(slots, needs_tree, model, device)
                t_tree_create += time.monotonic() - _t0

            # --- Phase 2: Run n_sims simulations in lockstep ---
            for _sim in range(n_sims):
                # Select leaves for all slots
                _t0 = time.monotonic()
                leaves = []
                for slot in slots:
                    leaf = select_leaf(slot.tree, slot.game)
                    leaves.append(leaf)
                t_select += time.monotonic() - _t0

                # Collect non-terminal leaves for batched NN eval
                _t0 = time.monotonic()
                eval_indices = []
                for i, leaf in enumerate(leaves):
                    if not leaf.is_terminal and leaf.deltas:
                        eval_indices.append(i)
                t_collect += time.monotonic() - _t0

                # Batch NN eval using delta planes
                if eval_indices:
                    _t0 = time.monotonic()
                    eval_leaves = [leaves[i] for i in eval_indices]
                    eval_trees = [slots[i].tree for i in eval_indices]
                    values = self._batch_eval_delta(
                        eval_leaves, eval_trees, model, device)
                    val_iter = iter(values)
                    t_batch_eval += time.monotonic() - _t0

                # Expand and backprop
                _t0 = time.monotonic()
                eval_set = set(eval_indices)
                for i, leaf in enumerate(leaves):
                    if leaf.is_terminal:
                        expand_and_backprop(slots[i].tree, leaf, 0.0)
                    elif i in eval_set:
                        nn_val = next(val_iter)
                        expand_and_backprop(slots[i].tree, leaf, nn_val)
                    else:
                        expand_and_backprop(slots[i].tree, leaf, 0.0)
                t_backprop += time.monotonic() - _t0

            n_turns += 1

            # Periodic timing breakdown
            if n_turns % 10 == 0:
                t_tot = t_tree_create + t_select + t_collect + t_batch_eval + t_backprop + t_move
                if t_tot > 0:
                    pbar.write(
                        f"  [turn {n_turns}] "
                        f"tree {t_tree_create/t_tot*100:.0f}% "
                        f"select {t_select/t_tot*100:.0f}% "
                        f"b2p {self._t_b2p/t_tot*100:.0f}% "
                        f"pad {self._t_pad/t_tot*100:.0f}% "
                        f"fwd {self._t_forward/t_tot*100:.0f}% "
                        f"| {n_turns/t_tot:.1f} turns/s"
                    )

            # --- Phase 3: Pick moves, record examples, advance games ---
            _t0 = time.monotonic()
            # Track max board grid size across all active trees
            for slot in slots:
                if slot.tree is not None:
                    if slot.tree.h > max_h:
                        max_h = slot.tree.h
                    if slot.tree.w > max_w:
                        max_w = slot.tree.w

            for slot in slots:
                turn_number = slot.turn_number
                temperature = 1.0 if turn_number < 15 else 0.1

                if slot.game.moves_left_in_turn == 1:
                    cell = select_single_move(slot.tree)
                    moves = [cell]
                    pair_visits = {}
                    for s1_idx, s1_node in slot.tree.root.children.items():
                        if s1_node.visit_count > 0:
                            pair_visits[(s1_idx, s1_idx)] = s1_node.visit_count
                else:
                    s1, s2 = select_move_pair(slot.tree, temperature=temperature)
                    moves = [s1, s2]
                    pair_visits = get_pair_visits(slot.tree)

                # Record training example
                example = {
                    "board": json.dumps({
                        f"{q},{r}": v.value if isinstance(v, Player) else v
                        for (q, r), v in slot.game.board.items()
                    }),
                    "current_player": slot.game.current_player.value,
                    "pair_visits": json.dumps({
                        f"{a},{b}": c for (a, b), c in pair_visits.items()
                    }),
                    "value_target": 0.0,  # backfilled after game ends
                    "game_id": slot.game_id,
                    "round_id": round_id,
                }
                slot.examples.append(example)
                total_positions += 1
                pos_bar.update(1)
                pos_bar.set_postfix(grid=f"{max_h}x{max_w}")

                # Apply moves
                for q, r in moves:
                    if slot.game.game_over:
                        break
                    slot.game.make_move(q, r)

                slot.turn_number += 1
                slot.tree = None  # will be re-created next iteration
                slot.sims_done = 0

            t_move += time.monotonic() - _t0

            # --- Phase 4: Check for finished games, backfill values ---
            for i, slot in enumerate(slots):
                game_done = slot.game.game_over or slot.game.move_count >= MAX_GAME_MOVES

                if game_done:
                    # Determine outcome
                    if slot.game.game_over and slot.game.winner != Player.NONE:
                        winner = slot.game.winner
                    else:
                        winner = Player.NONE  # draw

                    # Backfill value_target
                    for ex in slot.examples:
                        cp = Player(ex["current_player"])
                        if winner == Player.NONE:
                            ex["value_target"] = 0.0
                        elif cp == winner:
                            ex["value_target"] = 1.0
                        else:
                            ex["value_target"] = -1.0

                    all_examples.extend(slot.examples)
                    games_completed += 1
                    total_moves_in_completed += slot.game.move_count
                    avg_moves = total_moves_in_completed / games_completed
                    pbar.update(1)
                    pbar.set_postfix(avg_moves=f"{avg_moves:.0f}")

                    # Replace with new game if we still need more
                    if next_game_id < self.n_games:
                        slots[i] = self._new_slot(next_game_id)
                        next_game_id += 1
                    else:
                        # Mark slot as done (will idle)
                        slots[i] = self._new_slot(next_game_id)
                        next_game_id += 1

            # Check if all needed games are done
            if games_completed >= self.n_games:
                break

        pos_bar.close()
        pbar.close()

        # Timing breakdown
        t_total = t_tree_create + t_select + t_collect + t_batch_eval + t_backprop + t_move
        if t_total > 0 and n_turns > 0:
            print(f"\n  Timing breakdown ({n_turns} turns, {t_total:.1f}s total):")
            for label, t in [
                ("tree_create", t_tree_create),
                ("select_leaf", t_select),
                ("collect",     t_collect),
                ("batch_eval",  t_batch_eval),
                ("  board2plane", self._t_b2p),
                ("  pad+stack",   self._t_pad),
                ("  forward+xfer", self._t_forward),
                ("backprop",    t_backprop),
                ("move+record", t_move),
            ]:
                pct = 100 * t / t_total
                per_turn = 1000 * t / n_turns
                print(f"    {label:>15s}: {t:7.1f}s ({pct:5.1f}%)  {per_turn:6.1f}ms/turn")

        return all_examples[:self._max_examples()]

    def _max_examples(self) -> int:
        """No hard limit — return all examples."""
        return 10_000_000

    def _new_slot(self, game_id: int) -> SelfPlaySlot:
        """Create a new game slot. First move is always (0, 0)."""
        game = HexGame()
        game.make_move(0, 0)  # First move always (0, 0)
        return SelfPlaySlot(game=game, game_id=game_id)

    def _batch_create_trees(
        self,
        slots: list[SelfPlaySlot],
        indices: list[int],
        model: torch.nn.Module,
        device: torch.device,
    ):
        """Batch-create trees with a single forward pass."""
        active = [i for i in indices
                  if not slots[i].game.game_over
                  and slots[i].game.move_count < MAX_GAME_MOVES]
        if not active:
            return
        games = [slots[i].game for i in active]
        trees = create_trees_batched(games, model, device, add_noise=True)
        for i, tree in zip(active, trees):
            slots[i].tree = tree

    @torch.no_grad()
    def _batch_eval_delta(
        self,
        leaves: list,
        trees: list,
        model: torch.nn.Module,
        device: torch.device,
    ) -> list[float]:
        """Batch NN eval using delta from root planes — no board_to_planes."""
        B = len(leaves)
        if B == 0:
            return []

        _t0 = time.monotonic()
        # Root planes may have different sizes across trees (from different
        # batch creations). Find max dims for this eval batch.
        ph = max(t.root_planes.shape[1] for t in trees)
        pw = max(t.root_planes.shape[2] for t in trees)

        # Pre-allocate batch on CPU
        batch = torch.zeros(B, 2, ph, pw)
        mask = torch.zeros(B, 1, ph, pw)

        for i, (leaf, tree) in enumerate(zip(leaves, trees)):
            rp = tree.root_planes
            rh, rw = rp.shape[1], rp.shape[2]

            # Start from root planes (copy into batch directly)
            if leaf.player_flipped:
                batch[i, 0, :rh, :rw] = rp[1]
                batch[i, 1, :rh, :rw] = rp[0]
            else:
                batch[i, :, :rh, :rw] = rp

            # Apply deltas: stone placed by root's player → ch0
            # After flip, ch0 becomes ch1 and vice versa
            for gq, gr, ch in leaf.deltas:
                if 0 <= gq < rh and 0 <= gr < rw:
                    actual_ch = (1 - ch) if leaf.player_flipped else ch
                    batch[i, actual_ch, gq, gr] = 1.0

            mask[i, 0, :rh, :rw] = 1.0
        self._t_b2p += time.monotonic() - _t0

        _t0 = time.monotonic()
        batch = batch.to(device)
        mask = mask.to(device)
        values, _ = model(batch, mask)
        result = values.cpu().tolist()
        self._t_forward += time.monotonic() - _t0

        return result

    def save_round(self, examples: list[dict], round_id: int, output_dir: str):
        """Save examples as parquet."""
        import pandas as pd

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"round_{round_id}.parquet")
        df = pd.DataFrame(examples)
        df.to_parquet(path, index=False)
        print(f"Saved {len(examples):,} examples to {path}")
        return path
