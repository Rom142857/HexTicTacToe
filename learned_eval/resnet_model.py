"""ResNet + attention pair policy for HexTicTacToe.

Dual-head model predicting win rate (value) and move PAIR probabilities
(policy) from board positions. Fully convolutional — works at any board size.

Architecture:
  - Conv stem + N residual blocks (GroupNorm, size-independent)
  - Value head: masked global average pooling → FC → tanh
  - Policy head: bilinear attention over cell embeddings → N×N pair logits
    Symmetrized (order of stones in a pair doesn't matter),
    diagonal masked (can't place both on same cell).
"""

import json

import torch
import torch.nn as nn
import torch.nn.functional as F

BOARD_SIZE = 19


class ResBlock(nn.Module):
    def __init__(self, channels, gn_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(gn_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(gn_groups, channels)

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return F.relu(out + x)


class PairPolicyHead(nn.Module):
    """Bilinear attention head producing N×N pair logits.

    A(i,j) = (q_i · k_j + q_j · k_i) / 2  (symmetrized)
    Diagonal masked to -inf. Padding cells masked if mask provided.
    """

    def __init__(self, channels, head_dim=64):
        super().__init__()
        self.q_proj = nn.Conv2d(channels, head_dim, 1)
        self.k_proj = nn.Conv2d(channels, head_dim, 1)
        self.scale = head_dim ** -0.5

    def forward(self, trunk_features, mask=None):
        B, C, H, W = trunk_features.shape
        N = H * W

        Q = self.q_proj(trunk_features).flatten(2)  # [B, d, N]
        K = self.k_proj(trunk_features).flatten(2)  # [B, d, N]

        A = torch.bmm(Q.transpose(1, 2), K) * self.scale  # [B, N, N]
        A = (A + A.transpose(1, 2)) / 2  # symmetrize

        # Mask diagonal (can't place both stones on same cell)
        diag = torch.eye(N, device=A.device, dtype=torch.bool).unsqueeze(0)
        A = A.masked_fill(diag, float("-inf"))

        # Mask padding cells
        if mask is not None:
            mask_flat = mask.reshape(B, -1)  # [B, N]
            pair_mask = mask_flat.unsqueeze(2) * mask_flat.unsqueeze(1)  # [B, N, N]
            A = A.masked_fill(pair_mask == 0, float("-inf"))

        return A


class HexResNet(nn.Module):
    def __init__(self, in_channels=2, num_blocks=10, num_filters=128,
                 gn_groups=8, v_channels=32, pair_head_dim=64):
        super().__init__()

        # Stem
        self.stem_conv = nn.Conv2d(in_channels, num_filters, 3, padding=1,
                                   bias=False)
        self.stem_gn = nn.GroupNorm(gn_groups, num_filters)

        # Residual trunk
        self.blocks = nn.Sequential(
            *[ResBlock(num_filters, gn_groups) for _ in range(num_blocks)]
        )

        # Value head: conv → mean+max pool → FC
        self.v_conv = nn.Conv2d(num_filters, v_channels, 1, bias=False)
        self.v_gn = nn.GroupNorm(gn_groups, v_channels)
        self.v_fc1 = nn.Linear(v_channels * 2, 256)  # mean + max = 2x channels
        self.v_fc2 = nn.Linear(256, 1)

        # Pair policy head
        self.pair_head = PairPolicyHead(num_filters, pair_head_dim)

    def forward(self, x, mask=None):
        """Forward pass.

        Args:
            x: [B, C, H, W] board planes
            mask: [B, 1, H, W] float mask (1=valid, 0=padding). None=all valid.

        Returns:
            value: [B] scalar in [-1, 1]
            pair_logits: [B, N, N] raw pair logits (diagonal=-inf, padding=-inf)
        """
        s = F.relu(self.stem_gn(self.stem_conv(x)))
        t = self.blocks(s)

        # Value head: mean + max pooling for discriminative features
        v = F.relu(self.v_gn(self.v_conv(t)))  # [B, v_ch, H, W]
        if mask is not None:
            v_mean = (v * mask).sum(dim=[2, 3]) / mask.sum(dim=[2, 3]).clamp(min=1)
            v_max = (v + (mask - 1) * 1e9).amax(dim=[2, 3])
        else:
            v_mean = v.mean(dim=[2, 3])
            v_max = v.amax(dim=[2, 3])
        v = torch.cat([v_mean, v_max], dim=-1)  # [B, 2*v_ch]
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v)).squeeze(-1)

        # Pair policy head
        pair_logits = self.pair_head(t, mask)

        return v, pair_logits

    @staticmethod
    def marginalize(pair_logits):
        """Marginalize pair logits to single-move logits.

        P(cell_i) = sum_j P(i,j) → use logsumexp for numerical stability.
        Returns [B, N] logits.
        """
        return pair_logits.logsumexp(dim=-1)


def board_to_planes(board_dict, current_player, pad_to=None):
    """Convert {(q,r): player_int} board to planes tensor.

    Channel 0: current player's stones.
    Channel 1: opponent's stones.

    If pad_to is given, centers in a (pad_to x pad_to) grid.
    Otherwise uses tight bounding box + 6-cell margin on each side.

    Returns (planes, offset_q, offset_r, board_h, board_w).
    """
    if not board_dict:
        size = pad_to or 13
        return torch.zeros(2, size, size), 0, 0, size, size

    qs = [q for q, _r in board_dict]
    rs = [r for _q, r in board_dict]
    min_q, max_q = min(qs), max(qs)
    min_r, max_r = min(rs), max(rs)

    if pad_to is not None:
        h = w = pad_to
    else:
        margin = 2
        h = max_q - min_q + 1 + 2 * margin
        w = max_r - min_r + 1 + 2 * margin

    off_q = (h - (max_q - min_q + 1)) // 2 - min_q
    off_r = (w - (max_r - min_r + 1)) // 2 - min_r

    planes = torch.zeros(2, h, w)
    for (q, r), player in board_dict.items():
        gq = q + off_q
        gr = r + off_r
        if player == current_player:
            planes[0, gq, gr] = 1.0
        else:
            planes[1, gq, gr] = 1.0

    return planes, off_q, off_r, h, w


def parse_board_json(board_json):
    """Parse board JSON string to {(q,r): player_int} dict."""
    return {
        tuple(int(x) for x in k.split(",")): v
        for k, v in json.loads(board_json).items()
    }


def move_to_index(q, r, off_q, off_r, width):
    return (q + off_q) * width + (r + off_r)


def index_to_move(idx, off_q, off_r, width):
    return idx // width - off_q, idx % width - off_r


if __name__ == "__main__":
    model = HexResNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    for size in [11, 19, 25]:
        x = torch.randn(4, 2, size, size)
        mask = torch.ones(4, 1, size, size)
        v, pair = model(x, mask)
        N = size * size
        single = HexResNet.marginalize(pair)
        print(f"  {size}x{size}: value={v.shape}, pair={pair.shape}, "
              f"single={single.shape}, "
              f"v=[{v.min().item():.3f}, {v.max().item():.3f}]")

        # Verify symmetry and diagonal masking
        assert torch.allclose(pair, pair.transpose(1, 2)), "Not symmetric!"
        assert (pair[:, range(N), range(N)] == float("-inf")).all(), "Diagonal not masked!"
    print("All checks passed.")
