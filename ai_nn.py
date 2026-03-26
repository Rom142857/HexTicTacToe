"""Neural-network bot for HexTicTacToe.

Uses the trained ResNet model's policy and value heads to select moves.
Policy head ranks candidate moves; value head evaluates positions.
Compatible with evaluate.py and play.py (pair_moves=True).
"""

import os

import torch

from bot import Bot, BoardTooLargeError, _D2_OFFSETS
from learned_eval.resnet_model import HexResNet, board_to_planes, BOARD_SIZE

_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "learned_eval", "resnet_results", "best.pt"
)


def _get_candidates(board):
    """Return empty cells within hex-distance 2 of any occupied cell."""
    candidates = set()
    for q, r in board:
        for dq, dr in _D2_OFFSETS:
            nb = (q + dq, r + dr)
            if nb not in board:
                candidates.add(nb)
    return candidates


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

    @torch.no_grad()
    def get_move(self, game):
        self._nodes = 0
        self.last_depth = 1
        self.last_score = 0

        if not game.board:
            return [(0, 0)]

        candidates = _get_candidates(game.board)
        if not candidates:
            return [(0, 0)]

        n_moves = game.moves_left_in_turn

        # Check board fits in grid
        board = game.board
        qs = [q for q, _r in board]
        rs = [r for _q, r in board]
        span_q = max(qs) - min(qs) + 1
        span_r = max(rs) - min(rs) + 1
        if span_q > BOARD_SIZE or span_r > BOARD_SIZE:
            raise BoardTooLargeError(
                f"Board span {span_q}x{span_r} exceeds {BOARD_SIZE}x{BOARD_SIZE}")

        # Forward pass on current position
        planes, off_q, off_r, _, _ = board_to_planes(board, game.current_player)
        value, policy = self.model(planes.unsqueeze(0).to(self.device))
        policy = policy[0]
        self._nodes = 1

        # Map candidates to grid indices, filter out-of-bounds
        cand_idx = {}
        for q, r in candidates:
            gq, gr = q + off_q, r + off_r
            if 0 <= gq < BOARD_SIZE and 0 <= gr < BOARD_SIZE:
                cand_idx[(q, r)] = gq * BOARD_SIZE + gr

        if n_moves == 1:
            best = max(cand_idx, key=lambda m: policy[cand_idx[m]].item())
            self.last_score = value[0].item()
            return [best]

        # --- 2 moves: pick the best pair ---
        # Rank first moves by policy
        k1 = min(15, len(cand_idx))
        ranked = sorted(cand_idx, key=lambda m: policy[cand_idx[m]].item(),
                        reverse=True)[:k1]

        # Batch-evaluate positions after each candidate first move
        batch_planes = []
        batch_meta = []  # (m1, off_q2, off_r2)
        cur = game.current_player

        for m1 in ranked:
            new_board = dict(game.board)
            new_board[m1] = cur
            p2, oq2, or2, _, _ = board_to_planes(new_board, cur)
            batch_planes.append(p2)
            batch_meta.append((m1, oq2, or2))

        batch = torch.stack(batch_planes).to(self.device)
        vals, pols = self.model(batch)
        self._nodes += len(ranked)

        best_pair = None
        best_val = float("-inf")

        for i, (m1, oq2, or2) in enumerate(batch_meta):
            pol2 = pols[i]
            # Build second-move candidates (original candidates minus m1,
            # plus new neighbours of m1)
            cands2 = {}
            for q, r in candidates:
                if (q, r) == m1:
                    continue
                gq, gr = q + oq2, r + or2
                if 0 <= gq < BOARD_SIZE and 0 <= gr < BOARD_SIZE:
                    cands2[(q, r)] = gq * BOARD_SIZE + gr
            for dq, dr in _D2_OFFSETS:
                nb = (m1[0] + dq, m1[1] + dr)
                if nb not in game.board and nb != m1 and nb not in cands2:
                    gq, gr = nb[0] + oq2, nb[1] + or2
                    if 0 <= gq < BOARD_SIZE and 0 <= gr < BOARD_SIZE:
                        cands2[nb] = gq * BOARD_SIZE + gr

            if not cands2:
                continue

            m2 = max(cands2, key=lambda m: pol2[cands2[m]].item())
            val = vals[i].item()
            if val > best_val:
                best_val = val
                best_pair = [m1, m2]

        self.last_score = best_val
        if best_pair:
            return best_pair

        # Fallback: top two by policy from initial position
        fallback = sorted(cand_idx, key=lambda m: policy[cand_idx[m]].item(),
                          reverse=True)[:2]
        return fallback
