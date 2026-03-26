"""MCTS engine for HexTicTacToe with two-level tree (stone_1 → stone_2 → leaf).

Single NN forward pass at root caches the full N×N pair attention matrix.
Level-1 priors come from marginalizing over columns; level-2 priors come from
the conditional row pair_probs[stone1_idx, :].

Designed for batched self-play: select_leaf returns board state for NN eval,
expand_and_backprop integrates the result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from bot import _D2_OFFSETS
from game import HexGame, Player

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

PUCT_C = 2.5
TOP_K = 50


@dataclass
class MCTSNode:
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "MCTSNode"] = field(default_factory=dict)
    is_terminal: bool = False
    terminal_value: float = 0.0


@dataclass
class LeafInfo:
    """Info returned by select_leaf for batched NN eval."""
    path: list[tuple[MCTSNode, int]]  # [(parent_node, action_idx), ...]
    board_dict: dict | None = None  # board state at leaf (for NN eval)
    current_player: Player | None = None
    is_terminal: bool = False
    terminal_value: float = 0.0
    # Delta from root position: cells placed as (grid_row, grid_col, channel)
    # channel 0 = current player at root, channel 1 = opponent at root
    deltas: list[tuple[int, int, int]] = field(default_factory=list)
    player_flipped: bool = False  # True if leaf's current_player != root's


@dataclass
class MCTSTree:
    root: MCTSNode
    pair_probs: torch.Tensor  # [N, N] softmax of pair logits
    marginal_probs: torch.Tensor  # [N] marginalized priors
    root_planes: torch.Tensor | None = None  # [2, h, w] cached root planes
    root_player: Player | None = None  # current_player at root
    off_q: int = 0
    off_r: int = 0
    h: int = 0
    w: int = 0
    root_value: float = 0.0


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _get_candidates(board: dict) -> set[tuple[int, int]]:
    """Return empty cells within hex-distance 2 of any occupied cell."""
    candidates = set()
    for q, r in board:
        for dq, dr in _D2_OFFSETS:
            nb = (q + dq, r + dr)
            if nb not in board:
                candidates.add(nb)
    return candidates


def _cell_to_idx(q: int, r: int, off_q: int, off_r: int, w: int) -> int:
    return (q + off_q) * w + (r + off_r)


def _idx_to_cell(idx: int, off_q: int, off_r: int, w: int) -> tuple[int, int]:
    return idx // w - off_q, idx % w - off_r


def _valid_idx(q: int, r: int, off_q: int, off_r: int, h: int, w: int) -> bool:
    gq, gr = q + off_q, r + off_r
    return 0 <= gq < h and 0 <= gr < w


# ---------------------------------------------------------------------------
# PUCT
# ---------------------------------------------------------------------------

def _puct_select(node: MCTSNode, c: float = PUCT_C) -> int:
    """Select child with highest PUCT score. Returns action index."""
    best_score = -float("inf")
    best_action = -1
    sqrt_parent = math.sqrt(node.visit_count)

    for action, child in node.children.items():
        if child.is_terminal:
            q_val = child.terminal_value
        elif child.visit_count == 0:
            q_val = 0.0
        else:
            q_val = child.value_sum / child.visit_count
        score = q_val + c * child.prior * sqrt_parent / (1 + child.visit_count)
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


# ---------------------------------------------------------------------------
# Dirichlet noise
# ---------------------------------------------------------------------------

def _add_dirichlet_noise(node: MCTSNode, alpha: float = 0.3, frac: float = 0.25):
    """Add Dirichlet noise to priors of node's children (in-place)."""
    if not node.children:
        return
    actions = list(node.children.keys())
    noise = np.random.dirichlet([alpha] * len(actions))
    for a, n in zip(actions, noise):
        child = node.children[a]
        child.prior = (1 - frac) * child.prior + frac * n


# ---------------------------------------------------------------------------
# Core MCTS operations
# ---------------------------------------------------------------------------

def _build_tree_from_eval(
    game: HexGame,
    root_value: float,
    pair_probs: torch.Tensor,
    marginal: torch.Tensor,
    root_planes: torch.Tensor,
    off_q: int, off_r: int, h: int, w: int,
    add_noise: bool = True,
) -> MCTSTree:
    """Build an MCTSTree from pre-computed NN outputs (no model call)."""
    board = game.board
    root = MCTSNode()

    # Candidates for stone_1: empty cells near occupied
    if board:
        cands = _get_candidates(board)
    else:
        cands = {(0, 0)}

    # Filter to valid grid indices and get priors
    cand_priors = []
    for q, r in cands:
        if _valid_idx(q, r, off_q, off_r, h, w):
            idx = _cell_to_idx(q, r, off_q, off_r, w)
            cand_priors.append((idx, marginal[idx].item()))

    # Top-K by prior
    cand_priors.sort(key=lambda x: x[1], reverse=True)
    top_cands = cand_priors[:TOP_K]

    total_prior = sum(p for _, p in top_cands)
    if total_prior > 0:
        for idx, p in top_cands:
            root.children[idx] = MCTSNode(prior=p / total_prior)
    else:
        for idx, _ in top_cands:
            root.children[idx] = MCTSNode(prior=1.0 / len(top_cands))

    if add_noise:
        _add_dirichlet_noise(root)

    return MCTSTree(
        root=root,
        pair_probs=pair_probs,
        marginal_probs=marginal,
        root_planes=root_planes,
        root_player=game.current_player,
        off_q=off_q, off_r=off_r, h=h, w=w,
        root_value=root_value,
    )


def create_tree(
    game: HexGame,
    model: torch.nn.Module,
    device: torch.device,
    add_noise: bool = True,
) -> MCTSTree:
    """Create a single MCTS tree with one B=1 NN forward pass."""
    from learned_eval.resnet_model import board_to_planes

    planes, off_q, off_r, h, w = board_to_planes(
        game.board, game.current_player)
    N = h * w

    x = planes.unsqueeze(0).to(device)
    mask = torch.ones(1, 1, h, w, device=device)
    with torch.no_grad():
        value, pair_logits = model(x, mask)

    root_value = value[0].item()
    pair_probs = F.softmax(pair_logits[0].reshape(-1), dim=0).reshape(N, N).cpu()
    marginal = pair_probs.sum(dim=-1)

    return _build_tree_from_eval(
        game, root_value, pair_probs, marginal, planes,
        off_q, off_r, h, w, add_noise)


@torch.no_grad()
def create_trees_batched(
    games: list[HexGame],
    model: torch.nn.Module,
    device: torch.device,
    add_noise: bool = True,
) -> list[MCTSTree]:
    """Create trees for multiple games in one batched forward pass."""
    from learned_eval.resnet_model import board_to_planes

    B = len(games)
    if B == 0:
        return []

    # Build planes and find max dims for padding
    planes_list = []
    offsets = []
    for game in games:
        planes, off_q, off_r, h, w = board_to_planes(
            game.board, game.current_player)
        planes_list.append(planes)
        offsets.append((off_q, off_r, h, w))

    max_h = max(h for _, _, h, _ in offsets)
    max_w = max(w for _, _, _, w in offsets)

    batch = torch.zeros(B, 2, max_h, max_w)
    mask = torch.zeros(B, 1, max_h, max_w)
    for i, (planes, (_, _, h, w)) in enumerate(zip(planes_list, offsets)):
        batch[i, :, :h, :w] = planes
        mask[i, 0, :h, :w] = 1.0

    batch = batch.to(device)
    mask = mask.to(device)
    values, pair_logits = model(batch, mask)

    N = max_h * max_w
    trees = []
    for i, game in enumerate(games):
        root_value = values[i].item()
        pp = F.softmax(pair_logits[i].reshape(-1), dim=0).reshape(N, N).cpu()
        mg = pp.sum(dim=-1)
        off_q, off_r, h, w = offsets[i]
        # Store padded planes (max_h × max_w) so delta eval works on uniform grid
        root_planes = batch[i].cpu()
        tree = _build_tree_from_eval(
            game, root_value, pp, mg, root_planes, off_q, off_r, h, w,
            add_noise)
        trees.append(tree)

    return trees


def _expand_level2(
    tree: MCTSTree,
    parent_node: MCTSNode,
    stone1_idx: int,
    game: HexGame,
    add_noise: bool = True,
):
    """Expand level-2 children for a stone_1 node using cached pair_probs."""
    off_q, off_r, h, w = tree.off_q, tree.off_r, tree.h, tree.w
    stone1_q, stone1_r = _idx_to_cell(stone1_idx, off_q, off_r, w)

    # Conditional priors: pair_probs[stone1_idx, :]
    cond_probs = tree.pair_probs[stone1_idx]  # [N]

    # Candidates: empty cells near occupied + neighbors of stone_1
    board = game.board
    cands = _get_candidates(board) if board else set()
    # Add neighbors of stone_1
    for dq, dr in _D2_OFFSETS:
        nb = (stone1_q + dq, stone1_r + dr)
        if nb not in board:
            cands.add(nb)
    # Remove stone_1 itself
    cands.discard((stone1_q, stone1_r))

    cand_priors = []
    for q, r in cands:
        if _valid_idx(q, r, off_q, off_r, h, w):
            idx = _cell_to_idx(q, r, off_q, off_r, w)
            if idx != stone1_idx:
                cand_priors.append((idx, cond_probs[idx].item()))

    cand_priors.sort(key=lambda x: x[1], reverse=True)
    top_cands = cand_priors[:TOP_K]

    total_prior = sum(p for _, p in top_cands)
    if total_prior > 0:
        for idx, p in top_cands:
            parent_node.children[idx] = MCTSNode(prior=p / total_prior)
    elif top_cands:
        for idx, _ in top_cands:
            parent_node.children[idx] = MCTSNode(prior=1.0 / len(top_cands))

    if add_noise:
        _add_dirichlet_noise(parent_node)


def select_leaf(tree: MCTSTree, game: HexGame) -> LeafInfo:
    """Select a leaf node via PUCT at both levels.

    Makes temporary moves on the game, then undoes them.
    Returns LeafInfo with the board state for NN eval (or terminal info).
    path stores (parent_node, child_array_index) for backprop.
    """
    path = []
    off_q, off_r, w = tree.off_q, tree.off_r, tree.w
    root_cp = tree.root_player

    # Level 1: select stone_1
    s1_idx = _puct_select(tree.root)
    s1_node = tree.root.children[s1_idx]
    s1_q = s1_idx // w - off_q
    s1_r = s1_idx % w - off_r

    path.append((tree.root, s1_idx))

    # Make stone_1 move
    state1 = game.save_state()
    game.make_move(s1_q, s1_r)

    # Check terminal after stone_1
    if game.game_over:
        game.undo_move(s1_q, s1_r, state1)
        s1_node.is_terminal = True
        s1_node.terminal_value = 1.0
        return LeafInfo(path=path, is_terminal=True, terminal_value=1.0)

    # stone_1 grid coords for delta
    s1_gq, s1_gr = s1_q + off_q, s1_r + off_r
    deltas = [(s1_gq, s1_gr, 0)]

    # If turn ended after stone_1 (first-move-of-game case)
    if game.moves_left_in_turn == 0:
        cp = game.current_player
        game.undo_move(s1_q, s1_r, state1)
        return LeafInfo(
            path=path, current_player=cp, is_terminal=False,
            deltas=deltas, player_flipped=(cp != root_cp),
        )

    # Expand level-2 if needed
    if not s1_node.children:
        _expand_level2(tree, s1_node, s1_idx, game, add_noise=False)

    if not s1_node.children:
        cp = game.current_player
        game.undo_move(s1_q, s1_r, state1)
        return LeafInfo(
            path=path, current_player=cp, is_terminal=False,
            deltas=deltas, player_flipped=(cp != root_cp),
        )

    # Level 2: select stone_2
    s2_idx = _puct_select(s1_node)
    s2_node = s1_node.children[s2_idx]
    s2_q = s2_idx // w - off_q
    s2_r = s2_idx % w - off_r

    path.append((s1_node, s2_idx))

    # Make stone_2 move
    state2 = game.save_state()
    game.make_move(s2_q, s2_r)

    # Check terminal after stone_2
    if game.game_over:
        game.undo_move(s2_q, s2_r, state2)
        game.undo_move(s1_q, s1_r, state1)
        s2_node.is_terminal = True
        s2_node.terminal_value = 1.0
        return LeafInfo(path=path, is_terminal=True, terminal_value=1.0)

    # stone_2 delta
    s2_gq, s2_gr = s2_q + off_q, s2_r + off_r
    deltas.append((s2_gq, s2_gr, 0))

    cp = game.current_player

    # Undo both moves
    game.undo_move(s2_q, s2_r, state2)
    game.undo_move(s1_q, s1_r, state1)

    return LeafInfo(
        path=path, current_player=cp, is_terminal=False,
        deltas=deltas, player_flipped=(cp != root_cp),
    )


def expand_and_backprop(
    tree: MCTSTree,
    leaf: LeafInfo,
    nn_value: float,
):
    """Backpropagate a value through the path.

    nn_value is from the opponent's perspective (the NN evaluates the leaf
    position where it's the opponent's turn). Negate to get our perspective.
    """
    if leaf.is_terminal:
        value = leaf.terminal_value
    else:
        value = -nn_value  # flip: opponent's eval → our perspective

    # Backprop along path
    for parent_node, action_idx in leaf.path:
        child = parent_node.children[action_idx]
        child.visit_count += 1
        child.value_sum += value
    tree.root.visit_count += 1


def get_pair_visits(tree: MCTSTree) -> dict[tuple[int, int], int]:
    """Collect visit counts for (stone1_idx, stone2_idx) pairs."""
    visits = {}
    for s1_idx, s1_node in tree.root.children.items():
        for s2_idx, s2_node in s1_node.children.items():
            if s2_node.visit_count > 0:
                visits[(s1_idx, s2_idx)] = s2_node.visit_count
    return visits


def select_move_pair(
    tree: MCTSTree,
    temperature: float = 1.0,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Select a (stone1, stone2) move pair based on visit counts.

    Returns ((q1,r1), (q2,r2)).
    """
    pair_visits = get_pair_visits(tree)
    if not pair_visits:
        # Fallback: best stone_1 by visits
        best_s1 = max(tree.root.children,
                      key=lambda a: tree.root.children[a].visit_count)
        s1_cell = _idx_to_cell(best_s1, tree.off_q, tree.off_r, tree.w)
        s1_node = tree.root.children[best_s1]
        if s1_node.children:
            best_s2 = max(s1_node.children,
                          key=lambda a: s1_node.children[a].visit_count)
            s2_cell = _idx_to_cell(best_s2, tree.off_q, tree.off_r, tree.w)
        else:
            s2_cell = s1_cell
        return s1_cell, s2_cell

    pairs = list(pair_visits.keys())
    counts = torch.tensor([pair_visits[p] for p in pairs], dtype=torch.float32)

    if temperature < 0.05:
        best_idx = counts.argmax().item()
    else:
        logits = counts.log() / temperature
        probs = F.softmax(logits, dim=0)
        best_idx = torch.multinomial(probs, 1).item()

    s1_idx, s2_idx = pairs[best_idx]
    s1_cell = _idx_to_cell(s1_idx, tree.off_q, tree.off_r, tree.w)
    s2_cell = _idx_to_cell(s2_idx, tree.off_q, tree.off_r, tree.w)
    return s1_cell, s2_cell


def select_single_move(tree: MCTSTree) -> tuple[int, int]:
    """Select a single move (for moves_left == 1) from marginalized visits."""
    best_s1 = max(tree.root.children,
                  key=lambda a: tree.root.children[a].visit_count)
    return _idx_to_cell(best_s1, tree.off_q, tree.off_r, tree.w)
