"""Batched self-play game generation for MCTS training.

Runs 256 games in lockstep on a toroidal board: all slots search
simultaneously, batch NN evals on GPU, then advance games together.
Each round generates exactly COMPLETED_PER_ROUND (256) completed games.
In-progress games are saved and resumed across rounds so no work is wasted.

Multi-ply MCTS: after enough visits to a pair, child PosNodes are created
from the NN's pair logits, allowing the tree to search deeper.

Output: list of training examples saved as parquet.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import torch
from tqdm import tqdm

from game import Player
import torch.nn.functional as F

from learned_eval.mcts import (
    MCTSTree, N_CELLS, NON_ROOT_TOP_K, create_trees_batched, select_leaf,
    expand_and_backprop, maybe_expand_leaf, get_pair_visits, get_single_visits,
    select_move_pair, select_single_move,
)
from learned_eval.resnet_model import BOARD_SIZE
from toroidal_game import ToroidalHexGame, TORUS_SIZE

MAX_GAME_MOVES = 150
COMPLETED_PER_ROUND = 256

# Center of torus — first move always here
_CENTER = TORUS_SIZE // 2


@dataclass
class SelfPlaySlot:
    game: ToroidalHexGame
    tree: MCTSTree | None = None
    sims_done: int = 0
    turn_number: int = 0
    game_id: int = 0
    examples: list[dict] = field(default_factory=list)


class SelfPlayManager:
    def __init__(self, model, device, batch_size=256, n_sims=200,
                 data_dir="learned_eval/data/selfplay", viewer=None):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.n_sims = n_sims
        self.data_dir = data_dir
        self.viewer = viewer

    def _load_or_create_slots(self) -> tuple[list[SelfPlaySlot], int]:
        """Load pending games from previous round, or create all fresh slots."""
        batch_size = self.batch_size
        pending_path = os.path.join(self.data_dir, "pending.json")

        if os.path.exists(pending_path):
            with open(pending_path, 'r') as f:
                pending_data = json.load(f)

            slots = []
            for item in pending_data["games"]:
                game = ToroidalHexGame.from_dict(item["game"])
                slot = SelfPlaySlot(
                    game=game,
                    game_id=item["game_id"],
                    turn_number=item["turn_number"],
                    examples=item["examples"],
                )
                slots.append(slot)

            next_game_id = pending_data["next_game_id"]
            n_resumed = len(slots)

            # Fill remaining slots with fresh games
            while len(slots) < batch_size:
                slots.append(self._new_slot(next_game_id))
                next_game_id += 1

            print(f"Resumed {n_resumed} in-progress games")
        else:
            slots = []
            next_game_id = 0
            for _ in range(batch_size):
                slots.append(self._new_slot(next_game_id))
                next_game_id += 1

        return slots, next_game_id

    def _save_pending(self, slots: list[SelfPlaySlot], next_game_id: int):
        """Save in-progress games for next round."""
        pending = []
        for slot in slots:
            if not slot.game.game_over and slot.game.move_count < MAX_GAME_MOVES:
                pending.append({
                    "game": slot.game.to_dict(),
                    "game_id": slot.game_id,
                    "turn_number": slot.turn_number,
                    "examples": slot.examples,
                })

        data = {"games": pending, "next_game_id": next_game_id}
        path = os.path.join(self.data_dir, "pending.json")
        tmp = path + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(data, f)
        os.replace(tmp, path)
        print(f"Saved {len(pending)} in-progress games to {path}")

    def generate(self, round_id: int) -> list[dict]:
        """Generate COMPLETED_PER_ROUND completed games. Returns example dicts.

        Resumes in-progress games from the previous round. Saves pending
        games at the end for the next round.
        """
        model = self.model
        device = self.device
        n_sims = self.n_sims
        batch_size = self.batch_size

        all_examples: list[dict] = []
        games_completed = 0
        wins_a = 0
        wins_b = 0
        total_positions = 0
        total_moves_in_completed = 0

        # Load pending games or create fresh slots
        slots, next_game_id = self._load_or_create_slots()

        # Pre-allocate eval buffer (reused every simulation, no zeroing needed)
        eval_buf = torch.empty(batch_size, 2, BOARD_SIZE, BOARD_SIZE)

        pbar = tqdm(total=COMPLETED_PER_ROUND, desc="Games", unit="game", position=0)
        pos_bar = tqdm(desc="Positions", unit="pos", position=1)

        # Timing accumulators (seconds)
        t_tree_create = 0.0
        t_select = 0.0
        t_collect = 0.0
        t_batch_eval = 0.0
        t_backprop = 0.0
        t_move = 0.0
        n_turns = 0
        self._t_delta = 0.0    # delta plane construction
        self._t_forward = 0.0  # transfer + model forward + transfer back

        while games_completed < COMPLETED_PER_ROUND:
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
                eval_values = []
                expand_data = {}
                if eval_indices:
                    _t0 = time.monotonic()
                    eval_leaves = [leaves[i] for i in eval_indices]
                    eval_trees = [slots[i].tree for i in eval_indices]
                    eval_values, expand_data = self._batch_eval_delta(
                        eval_leaves, eval_trees, model, device, eval_buf)
                    t_batch_eval += time.monotonic() - _t0

                # Expand and backprop
                _t0 = time.monotonic()
                eval_map = {gi: j for j, gi in enumerate(eval_indices)}
                for i, leaf in enumerate(leaves):
                    j = eval_map.get(i)
                    if leaf.is_terminal:
                        expand_and_backprop(slots[i].tree, leaf, 0.0)
                    elif j is not None:
                        nn_val = eval_values[j]
                        expand_and_backprop(slots[i].tree, leaf, nn_val)
                        # Create child PosNode if expansion data available
                        data = expand_data.get(j)
                        if data is not None:
                            maybe_expand_leaf(
                                slots[i].tree, leaf, *data)
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
                        f"delta {self._t_delta/t_tot*100:.0f}% "
                        f"fwd {self._t_forward/t_tot*100:.0f}% "
                        f"| {n_turns/t_tot:.1f} turns/s"
                    )

            # --- Phase 3: Pick moves, record examples, advance games ---
            _t0 = time.monotonic()

            for slot in slots:
                turn_number = slot.turn_number
                temperature = 1.0 if turn_number < 15 else 0.1

                if slot.game.moves_left_in_turn == 1:
                    cell = select_single_move(slot.tree)
                    moves = [cell]
                    pair_visits = get_single_visits(slot.tree)
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
                    "move_count": slot.game.move_count,
                    "moves_left": 0,      # backfilled after game ends
                    "game_drawn": False,   # backfilled after game ends
                    "game_id": slot.game_id,
                    "round_id": round_id,
                }
                slot.examples.append(example)
                total_positions += 1
                pos_bar.update(1)

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

                    # Backfill value_target, moves_left, game_drawn
                    total_moves = slot.game.move_count
                    is_drawn = (winner == Player.NONE)
                    for ex in slot.examples:
                        ex["round_id"] = round_id
                        ex["moves_left"] = total_moves - ex["move_count"]
                        ex["game_drawn"] = is_drawn
                        cp = Player(ex["current_player"])
                        if is_drawn:
                            ex["value_target"] = 0.0
                        elif cp == winner:
                            ex["value_target"] = 1.0
                        else:
                            ex["value_target"] = -1.0

                    all_examples.extend(slot.examples)
                    if self.viewer:
                        self.viewer.add_finished(slot)
                    games_completed += 1
                    if winner == Player.A:
                        wins_a += 1
                    elif winner == Player.B:
                        wins_b += 1
                    total_moves_in_completed += slot.game.move_count
                    avg_moves = total_moves_in_completed / games_completed
                    draws = games_completed - wins_a - wins_b
                    pbar.update(1)
                    n = games_completed
                    pbar.set_postfix(
                        avg_moves=f"{avg_moves:.0f}",
                        A=f"{100*wins_a/n:.0f}%",
                        B=f"{100*wins_b/n:.0f}%",
                        draw=f"{100*draws/n:.0f}%",
                    )

                    # Always replace with a fresh game
                    slots[i] = self._new_slot(next_game_id)
                    next_game_id += 1

            # Update viewer (just 4 attribute sets — ~0 cost)
            if self.viewer:
                self.viewer.update_slots(
                    slots, games_completed, COMPLETED_PER_ROUND, round_id)

            # Check if all needed games are done
            if games_completed >= COMPLETED_PER_ROUND:
                break

        pos_bar.close()
        pbar.close()

        # Save in-progress games for next round
        self._save_pending(slots, next_game_id)

        # Timing breakdown
        t_total = t_tree_create + t_select + t_collect + t_batch_eval + t_backprop + t_move
        if t_total > 0 and n_turns > 0:
            print(f"\n  Timing breakdown ({n_turns} turns, {t_total:.1f}s total):")
            for label, t in [
                ("tree_create", t_tree_create),
                ("select_leaf", t_select),
                ("collect",     t_collect),
                ("batch_eval",  t_batch_eval),
                ("  delta",      self._t_delta),
                ("  forward+xfer", self._t_forward),
                ("backprop",    t_backprop),
                ("move+record", t_move),
            ]:
                pct = 100 * t / t_total
                per_turn = 1000 * t / n_turns
                print(f"    {label:>15s}: {t:7.1f}s ({pct:5.1f}%)  {per_turn:6.1f}ms/turn")

        return all_examples

    def _new_slot(self, game_id: int) -> SelfPlaySlot:
        """Create a new game slot on a toroidal board. First move at center."""
        game = ToroidalHexGame()
        game.make_move(_CENTER, _CENTER)  # First move at torus center
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
        eval_buf: torch.Tensor,
    ) -> tuple[list[float], dict]:
        """Batch NN eval using delta from root planes.

        Returns (values, expand_data) where expand_data maps eval-batch index
        to (marginal, top_indices, top_values) for leaves needing expansion.
        """
        B = len(leaves)
        if B == 0:
            return [], {}

        _t0 = time.monotonic()
        batch = eval_buf[:B]

        for i, (leaf, tree) in enumerate(zip(leaves, trees)):
            rp = tree.root_planes

            # Start from root planes (overwrites entire slice)
            if leaf.player_flipped:
                batch[i, 0] = rp[1]
                batch[i, 1] = rp[0]
            else:
                batch[i] = rp

            # Apply deltas
            for gq, gr, ch in leaf.deltas:
                actual_ch = (1 - ch) if leaf.player_flipped else ch
                batch[i, actual_ch, gq, gr] = 1.0
        self._t_delta += time.monotonic() - _t0

        _t0 = time.monotonic()
        batch = batch.to(device)
        values, pair_logits, _, _ = model(batch)
        result = values.cpu().tolist()

        # Batched expansion: avoid full N² softmax.
        # Top-K pairs from raw logits (monotonic, same indices as softmax).
        # Marginal via logsumexp + softmax over N (not N²).
        expand_data = {}
        need_expand = [i for i, lf in enumerate(leaves) if lf.needs_expansion]
        if need_expand:
            ne = len(need_expand)
            exp_logits = pair_logits[need_expand]              # [K, N, N]
            flat_logits = exp_logits.reshape(ne, -1)           # [K, N²]

            # Top-K pairs from raw logits (same ranking as softmax)
            top_raw, top_idxs = flat_logits.topk(200, dim=-1)  # [K, 200]
            # Normalize to probabilities over just the selected pairs
            top_vals = F.softmax(top_raw, dim=-1)              # [K, 200]

            # Marginal via logsumexp (avoids materializing N² softmax)
            marginal_logits = exp_logits.logsumexp(dim=-1)     # [K, N]
            marginals = F.softmax(marginal_logits, dim=-1)     # [K, N]

            del exp_logits, flat_logits, top_raw, marginal_logits
            marginals_cpu = marginals.cpu()
            top_idxs_cpu = top_idxs.cpu()
            top_vals_cpu = top_vals.cpu()
            del marginals, top_vals, top_idxs
            for j, i in enumerate(need_expand):
                expand_data[i] = (
                    marginals_cpu[j], top_idxs_cpu[j], top_vals_cpu[j])

        del pair_logits
        self._t_forward += time.monotonic() - _t0

        return result, expand_data

    def save_round(self, examples: list[dict], round_id: int, output_dir: str):
        """Save examples as parquet."""
        import pandas as pd

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"round_{round_id}.parquet")
        df = pd.DataFrame(examples)
        df.to_parquet(path, index=False)
        print(f"Saved {len(examples):,} examples to {path}")
        return path
