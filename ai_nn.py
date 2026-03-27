"""Neural-network bot for HexTicTacToe.

Uses the trained ResNet model's policy and value heads to select moves.
Policy head ranks candidate moves; value head evaluates positions.
Translates between real game coordinates and 19x19 torus internally.
Compatible with evaluate.py and play.py (pair_moves=True).
"""

import os

import torch

from bot import Bot
from learned_eval.resnet_model import HexResNet, board_to_planes_torus, BOARD_SIZE
from toroidal_game import TORUS_SIZE

_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "learned_eval", "resnet_results", "best.pt"
)

# Anchor: maps real (0,0) to torus center
_ANCHOR_Q = TORUS_SIZE // 2
_ANCHOR_R = TORUS_SIZE // 2


def _get_candidates_torus(board):
    """Return all empty cells on the torus."""
    all_cells = {(q, r) for q in range(TORUS_SIZE) for r in range(TORUS_SIZE)}
    return all_cells - board.keys()


class MinimaxBot(Bot):
    """Neural-network bot using ResNet policy + value heads."""

    pair_moves = True

    def __init__(self, time_limit=0.05, model_path=None):
        super().__init__(time_limit)
        self._nodes = 0
        self.last_score = 0

        if model_path is None:
            model_path = _DEFAULT_MODEL_PATH

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(model_path, map_location=self.device, weights_only=True)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        self.model = HexResNet()
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _real_to_torus(self, q, r):
        return (q + _ANCHOR_Q) % TORUS_SIZE, (r + _ANCHOR_R) % TORUS_SIZE

    def _torus_to_real(self, tq, tr):
        return tq - _ANCHOR_Q, tr - _ANCHOR_R

    @torch.no_grad()
    def get_move(self, game):
        self._nodes = 0
        self.last_depth = 1
        self.last_score = 0

        if not game.board:
            return [(0, 0)]

        # Translate real board to torus coords
        torus_board = {}
        for (q, r), player in game.board.items():
            tq, tr = self._real_to_torus(q, r)
            torus_board[(tq, tr)] = player

        candidates = _get_candidates_torus(torus_board)
        if not candidates:
            return [(0, 0)]

        n_moves = game.moves_left_in_turn

        # Forward pass on current position
        planes = board_to_planes_torus(torus_board, game.current_player)
        value, policy, _, _ = self.model(planes.unsqueeze(0).to(self.device))
        policy = policy[0]
        self._nodes = 1

        # Map candidates to grid indices (all valid on torus)
        cand_idx = {}
        for q, r in candidates:
            cand_idx[(q, r)] = q * BOARD_SIZE + r

        if n_moves == 1:
            best_t = max(cand_idx, key=lambda m: policy[cand_idx[m]].item())
            self.last_score = value[0].item()
            rq, rr = self._torus_to_real(*best_t)
            return [(rq, rr)]

        # --- 2 moves: pick the best pair ---
        # Rank first moves by policy
        k1 = min(15, len(cand_idx))
        ranked = sorted(cand_idx, key=lambda m: policy[cand_idx[m]].item(),
                        reverse=True)[:k1]

        # Batch-evaluate positions after each candidate first move
        batch_planes = []
        batch_meta = []  # (m1_torus,)
        cur = game.current_player

        for m1 in ranked:
            new_board = dict(torus_board)
            new_board[m1] = cur
            p2 = board_to_planes_torus(new_board, cur)
            batch_planes.append(p2)
            batch_meta.append(m1)

        batch = torch.stack(batch_planes).to(self.device)
        vals, pols, _, _ = self.model(batch)
        self._nodes += len(ranked)

        best_pair = None
        best_val = float("-inf")

        for i, m1 in enumerate(batch_meta):
            pol2 = pols[i]
            # All empty cells except m1
            cands2 = {}
            for q, r in candidates:
                if (q, r) == m1:
                    continue
                cands2[(q, r)] = q * BOARD_SIZE + r

            if not cands2:
                continue

            m2 = max(cands2, key=lambda m: pol2[cands2[m]].item())
            val = vals[i].item()
            if val > best_val:
                best_val = val
                best_pair = [m1, m2]

        self.last_score = best_val
        if best_pair:
            return [self._torus_to_real(*m) for m in best_pair]

        # Fallback: top two by policy from initial position
        fallback = sorted(cand_idx, key=lambda m: policy[cand_idx[m]].item(),
                          reverse=True)[:2]
        return [self._torus_to_real(*m) for m in fallback]
