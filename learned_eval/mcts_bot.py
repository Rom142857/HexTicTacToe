"""MCTS-powered bot for HexTicTacToe evaluation.

Wraps the MCTS engine into a Bot subclass compatible with evaluate.py.
Loads its own model so it works in subprocesses (no CUDA fork issues).
"""

import os

import torch

from bot import Bot
from game import HexGame
from learned_eval.resnet_model import HexResNet

_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "resnet_results", "best.pt"
)


class MCTSBot(Bot):
    """Bot that uses MCTS with a neural network for move selection."""

    pair_moves = True

    def __init__(self, time_limit=1.0, model_path=None, n_sims=200,
                 device=None):
        super().__init__(time_limit)
        self._nodes = 0
        self.n_sims = n_sims

        if model_path is None:
            model_path = _DEFAULT_MODEL_PATH
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        ckpt = torch.load(model_path, map_location=self.device,
                          weights_only=True)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        self.model = HexResNet()
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def get_move(self, game: HexGame) -> list[tuple[int, int]]:
        from learned_eval.mcts import (
            create_tree, select_leaf, expand_and_backprop,
            select_move_pair, select_single_move,
        )
        from learned_eval.resnet_model import board_to_planes

        self.last_depth = self.n_sims
        self._nodes = 0

        # Empty board: always play (0, 0)
        if not game.board:
            return [(0, 0)]

        # Create tree (1 NN eval)
        tree = create_tree(game, self.model, self.device, add_noise=False)
        self._nodes = 1

        # Run simulations
        for _ in range(self.n_sims):
            leaf = select_leaf(tree, game)
            if leaf.is_terminal:
                expand_and_backprop(tree, leaf, 0.0)
            else:
                # NN eval at leaf
                planes, off_q, off_r, h, w = board_to_planes(
                    leaf.board_dict, leaf.current_player,
                )
                x = planes.unsqueeze(0).to(self.device)
                mask = torch.ones(1, 1, h, w, device=self.device)
                value, _ = self.model(x, mask)
                expand_and_backprop(tree, leaf, value[0].item())
                self._nodes += 1

        # Select move
        if game.moves_left_in_turn == 1:
            cell = select_single_move(tree)
            return [cell]
        else:
            s1, s2 = select_move_pair(tree, temperature=0.1)
            return [s1, s2]

    def __str__(self):
        return f"MCTSBot({self.n_sims})"
