"""MCTS self-play training loop: generate → train → evaluate.

Usage:
    python -m learned_eval.train_loop --rounds 10 --amp
"""

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from game import HexGame, HEX_DIRECTIONS, Player
from learned_eval.resnet_model import BOARD_SIZE, HexResNet, board_to_planes_torus
from learned_eval.symmetry import (
    apply_symmetry_planes, PERMS, N as SYM_N,
)

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ---------------------------------------------------------------------------
# Chain target precomputation (vectorized with numpy)
# ---------------------------------------------------------------------------

def _precompute_chain_tables(N=BOARD_SIZE, win_len=6):
    """Precompute window cell indices and per-cell window membership."""
    n_windows = N * N * 3  # 3 directions
    win_qs = np.zeros((n_windows, win_len), dtype=np.int32)
    win_rs = np.zeros((n_windows, win_len), dtype=np.int32)

    w_idx = 0
    for dq, dr in HEX_DIRECTIONS:
        for sq in range(N):
            for sr in range(N):
                for i in range(win_len):
                    win_qs[w_idx, i] = (sq + i * dq) % N
                    win_rs[w_idx, i] = (sr + i * dr) % N
                w_idx += 1

    # Cell-to-windows: for each cell, which windows contain it
    membership = [[] for _ in range(N * N)]
    for w in range(n_windows):
        for i in range(win_len):
            flat = win_qs[w, i] * N + win_rs[w, i]
            membership[flat].append(w)

    max_per_cell = max(len(m) for m in membership)
    cell_windows = np.zeros((N * N, max_per_cell), dtype=np.int32)
    cell_mask = np.zeros((N * N, max_per_cell), dtype=bool)
    for idx, m in enumerate(membership):
        cell_windows[idx, :len(m)] = m
        cell_mask[idx, :len(m)] = True

    return win_qs, win_rs, cell_windows, cell_mask


_WIN_QS, _WIN_RS, _CELL_WINDOWS, _CELL_MASK = _precompute_chain_tables()


def compute_chain_targets(board_dict, current_player):
    """Compute chain length targets [2, N, N] and loss mask [2, N, N].

    Channel 0: current player's max unblocked chain through each cell.
    Channel 1: opponent's max unblocked chain through each cell.

    Mask: 0 on cells occupied by the other player (since those block chains).
    """
    N = BOARD_SIZE
    opponent = Player.B if current_player == Player.A else Player.A

    cur_board = np.zeros((N, N), dtype=np.int8)
    opp_board = np.zeros((N, N), dtype=np.int8)
    for (q, r), p in board_dict.items():
        if p == current_player:
            cur_board[q, r] = 1
        else:
            opp_board[q, r] = 1

    targets = np.zeros((2, N, N), dtype=np.float32)

    for ch, (player_b, blocker_b) in enumerate(
            [(cur_board, opp_board), (opp_board, cur_board)]):
        # Count player stones in each window
        player_in = player_b[_WIN_QS, _WIN_RS]    # [n_windows, 6]
        blocker_in = blocker_b[_WIN_QS, _WIN_RS]   # [n_windows, 6]
        counts = player_in.sum(axis=1)              # [n_windows]
        blocked = blocker_in.any(axis=1)            # [n_windows]
        unblocked = np.where(blocked, 0, counts).astype(np.float32)

        # For each cell, max unblocked count across its windows
        vals = unblocked[_CELL_WINDOWS]             # [N*N, max_per_cell]
        vals[~_CELL_MASK] = 0
        targets[ch] = vals.max(axis=1).reshape(N, N)

    # Loss mask: don't predict opponent chains on current player's cells
    mask = np.ones((2, N, N), dtype=np.float32)
    for (q, r), p in board_dict.items():
        if p == current_player:
            mask[1, q, r] = 0.0   # mask opponent channel on current's cells
        else:
            mask[0, q, r] = 0.0   # mask current channel on opponent's cells

    return torch.from_numpy(targets), torch.from_numpy(mask)


# ---------------------------------------------------------------------------
# Training dataset with sparse visits, D6 augmentation, per-round weights
# ---------------------------------------------------------------------------

class SelfPlayDataset(torch.utils.data.Dataset):
    """Dataset of self-play positions with D6 augmentation and round weights.

    Visits are stored sparsely (~200 entries per sample) and densified on the
    fly in __getitem__. Chain targets and moves-left are included for
    auxiliary losses. A random D6 symmetry is applied at access time.
    """

    def __init__(self, planes, visit_dicts, values, round_ids,
                 chain_targets, chain_masks, moves_left, draw_mask,
                 current_round, decay=0.75, augment=True):
        self.planes = planes              # [N, 2, 25, 25]
        self.visit_dicts = visit_dicts    # list of list[(flat_pair_idx, prob)]
        self.values = values              # [N]
        self.chain_targets = chain_targets  # [N, 2, 25, 25]
        self.chain_masks = chain_masks    # [N, 2, 25, 25]
        self.moves_left = moves_left      # [N]
        self.draw_mask = draw_mask        # [N] bool — True = drawn, mask out
        self.augment = augment
        self._NN = BOARD_SIZE * BOARD_SIZE

        ages = current_round - round_ids.float()
        self.weights = decay ** ages

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        planes = self.planes[idx]
        value = self.values[idx]
        visit_entries = self.visit_dicts[idx]
        chain_t = self.chain_targets[idx]   # [2, 25, 25]
        chain_m = self.chain_masks[idx]     # [2, 25, 25]
        ml = self.moves_left[idx]
        drawn = self.draw_mask[idx]

        if self.augment:
            k = random.randint(0, 11)
        else:
            k = 0

        if k != 0:
            planes = apply_symmetry_planes(planes, k)
            chain_t = apply_symmetry_planes(chain_t, k)
            chain_m = apply_symmetry_planes(chain_m, k)

        # Build dense visit vector, applying symmetry to sparse entries
        NN = self._NN
        visit_vec = torch.zeros(NN * NN)
        if visit_entries:
            if k != 0:
                perm = PERMS[k]
                for flat_idx, prob in visit_entries:
                    a = flat_idx // NN
                    b = flat_idx % NN
                    new_a = int(perm[a])
                    new_b = int(perm[b])
                    visit_vec[new_a * NN + new_b] = prob
            else:
                for flat_idx, prob in visit_entries:
                    visit_vec[flat_idx] = prob

        return planes, visit_vec, value, chain_t, chain_m, ml, drawn


def load_selfplay_rounds(data_dir: str, current_round: int,
                         window: int = 4, decay: float = 0.75,
                         augment: bool = True) -> SelfPlayDataset:
    """Load the last `window` rounds of self-play data into a SelfPlayDataset.

    Computes chain targets from board positions. Visits stored sparsely.
    """
    rounds = range(max(0, current_round - window + 1), current_round + 1)
    dfs = []
    for r in rounds:
        path = os.path.join(data_dir, f"round_{r}.parquet")
        if os.path.exists(path):
            dfs.append(pd.read_parquet(path))

    if not dfs:
        raise FileNotFoundError(f"No self-play data found for rounds {list(rounds)}")

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df):,} examples from rounds {list(rounds)}")

    N = BOARD_SIZE * BOARD_SIZE  # 625
    n = len(df)
    planes_tensor = torch.zeros(n, 2, BOARD_SIZE, BOARD_SIZE)
    visit_dicts: list[list[tuple[int, float]]] = []
    value_tensor = torch.zeros(n)
    round_ids = torch.zeros(n, dtype=torch.int64)
    chain_targets = torch.zeros(n, 2, BOARD_SIZE, BOARD_SIZE)
    chain_masks = torch.zeros(n, 2, BOARD_SIZE, BOARD_SIZE)
    moves_left_tensor = torch.zeros(n)
    draw_mask = torch.zeros(n, dtype=torch.bool)

    has_moves_left = "moves_left" in df.columns

    for i, row in enumerate(tqdm(df.itertuples(), total=n,
                                  desc="Preprocessing", unit="ex",
                                  mininterval=2)):
        board_dict = {
            tuple(int(x) for x in k.split(",")): v
            for k, v in json.loads(row.board).items()
        }
        cp = Player(row.current_player)
        planes_tensor[i] = board_to_planes_torus(board_dict, cp)

        # Chain targets
        ct, cm = compute_chain_targets(board_dict, cp)
        chain_targets[i] = ct
        chain_masks[i] = cm

        # Sparse visit entries
        pair_visits_raw = json.loads(row.pair_visits)
        total_visits = sum(pair_visits_raw.values())
        entries: list[tuple[int, float]] = []
        if total_visits > 0:
            for key, count in pair_visits_raw.items():
                parts = key.split(",")
                a, b = int(parts[0]), int(parts[1])
                entries.append((a * N + b, count / total_visits))
        visit_dicts.append(entries)

        value_tensor[i] = row.value_target
        round_ids[i] = row.round_id

        if has_moves_left:
            moves_left_tensor[i] = row.moves_left
            draw_mask[i] = bool(row.game_drawn)
        else:
            # Old data without moves_left — mask out from loss
            draw_mask[i] = True

    return SelfPlayDataset(
        planes_tensor, visit_dicts, value_tensor, round_ids,
        chain_targets, chain_masks, moves_left_tensor, draw_mask,
        current_round, decay=decay, augment=augment,
    )


def compute_selfplay_loss(value_pred, pair_logits, moves_left_pred, chain_pred,
                          visit_dist, value_target,
                          moves_left_target, draw_mask,
                          chain_target, chain_mask):
    """Combined loss: value + policy + moves_left + chain.

    Primary losses (value, policy) are weighted ~10x the auxiliary losses
    (moves_left, chain).
    """
    B, N_sq, _ = pair_logits.shape

    # --- Primary losses ---
    value_loss = F.mse_loss(value_pred, value_target)

    flat_logits = pair_logits.reshape(B, -1)
    log_probs = F.log_softmax(flat_logits, dim=-1)
    # nan_to_num: 0 * -inf (diagonal) -> nan -> 0
    policy_loss = -(visit_dist * log_probs).nan_to_num(0.0).sum(dim=-1).mean()

    # --- Auxiliary: moves left (mask drawn games, normalize to [0,1]) ---
    valid = ~draw_mask
    if valid.any():
        ml_pred_norm = moves_left_pred[valid] / 150.0
        ml_tgt_norm = moves_left_target[valid] / 150.0
        ml_loss = F.mse_loss(ml_pred_norm, ml_tgt_norm)
    else:
        ml_loss = torch.zeros(1, device=value_pred.device).squeeze()

    # --- Auxiliary: chain length (masked MSE, values 0-6) ---
    chain_diff_sq = (chain_pred - chain_target) ** 2  # [B, 2, H, W]
    masked = chain_diff_sq * chain_mask
    chain_loss = masked.sum() / chain_mask.sum().clamp(min=1)

    # Primary losses ~10x auxiliary
    total = (value_loss + policy_loss
             + 0.1 * ml_loss + 0.1 * chain_loss)

    return total, value_loss, policy_loss, ml_loss, chain_loss


def train_one_epoch(model, optimizer, dataset, device, batch_size=512,
                    use_amp=False, scaler=None, grad_clip=50.0):
    """Train one epoch on self-play data with weighted sampling."""
    model.train()
    sampler = WeightedRandomSampler(
        weights=dataset.weights,
        num_samples=len(dataset),
        replacement=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        num_workers=0, pin_memory=True)

    total_loss = 0.0
    total_vloss = 0.0
    total_ploss = 0.0
    total_ml_loss = 0.0
    total_chain_loss = 0.0
    n_batches = 0

    for (planes, visit_dist, value_target,
         chain_target, chain_mask, ml_target, drawn) in tqdm(
            loader, desc="Training", unit="batch"):
        planes = planes.to(device)
        visit_dist = visit_dist.to(device)
        value_target = value_target.to(device)
        chain_target = chain_target.to(device)
        chain_mask = chain_mask.to(device)
        ml_target = ml_target.to(device)
        drawn = drawn.to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast("cuda"):
                value_pred, pair_logits, ml_pred, chain_pred = model(planes)
                loss, vloss, ploss, ml_loss, cl = compute_selfplay_loss(
                    value_pred, pair_logits, ml_pred, chain_pred,
                    visit_dist, value_target, ml_target, drawn,
                    chain_target, chain_mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            value_pred, pair_logits, ml_pred, chain_pred = model(planes)
            loss, vloss, ploss, ml_loss, cl = compute_selfplay_loss(
                value_pred, pair_logits, ml_pred, chain_pred,
                visit_dist, value_target, ml_target, drawn,
                chain_target, chain_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        total_vloss += vloss.item()
        total_ploss += ploss.item()
        total_ml_loss += ml_loss.item()
        total_chain_loss += cl.item()
        n_batches += 1

    d = max(n_batches, 1)
    return (total_loss / d, total_vloss / d, total_ploss / d,
            total_ml_loss / d, total_chain_loss / d)


# ---------------------------------------------------------------------------
# Evaluation vs MinimaxBot (sequential, no multiprocessing for GPU bot)
# ---------------------------------------------------------------------------

def evaluate_vs_minimax(model_path: str, n_games: int = 100,
                        n_sims: int = 200, minimax_time: float = 0.1,
                        device: str = "cuda") -> float:
    """Play MCTSBot vs MinimaxBot(ai_cpp), return MCTSBot win rate."""
    from learned_eval.mcts_bot import MCTSBot

    mcts_bot = MCTSBot(model_path=model_path, n_sims=n_sims, device=device)

    # Import minimax bot
    try:
        import ai_cpp
        minimax_bot = ai_cpp.MinimaxBot(time_limit=minimax_time)
    except ImportError:
        from bot import RandomBot
        print("WARNING: ai_cpp not available, using RandomBot for evaluation")
        minimax_bot = RandomBot(time_limit=minimax_time)

    wins = 0
    losses = 0
    draws = 0

    for game_idx in tqdm(range(n_games), desc="Evaluating", unit="game"):
        game = HexGame()
        swapped = game_idx % 2 == 1

        if swapped:
            bots = {Player.A: minimax_bot, Player.B: mcts_bot}
        else:
            bots = {Player.A: mcts_bot, Player.B: minimax_bot}

        move_count = 0
        while not game.game_over and move_count < 200:
            bot = bots[game.current_player]
            try:
                result = bot.get_move(game)
            except Exception as e:
                print(f"  Bot error: {e}")
                break

            if bot.pair_moves:
                moves = result
            else:
                moves = [result]

            for q, r in moves:
                if game.game_over:
                    break
                if not game.make_move(q, r):
                    break
                move_count += 1

        # Determine winner from MCTS perspective
        if game.game_over and game.winner != Player.NONE:
            mcts_player = Player.B if swapped else Player.A
            if game.winner == mcts_player:
                wins += 1
            else:
                losses += 1
        else:
            draws += 1

    total = max(wins + losses + draws, 1)
    win_rate = (wins + 0.5 * draws) / total
    print(f"  MCTSBot vs Minimax: {wins}W / {losses}L / {draws}D "
          f"= {100 * win_rate:.1f}% win rate")
    return win_rate


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scaler, round_num, output_dir,
                    best_win_rate=0.0):
    """Save model checkpoint atomically."""
    os.makedirs(output_dir, exist_ok=True)
    ckpt = {
        "round": round_num,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "best_win_rate": best_win_rate,
    }
    path = os.path.join(output_dir, f"round_{round_num}.pt")
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)

    # Also save as best.pt
    best_path = os.path.join(output_dir, "best.pt")
    tmp = best_path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, best_path)

    print(f"Checkpoint saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MCTS self-play training loop")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Number of training rounds")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Number of parallel games in self-play")
    parser.add_argument("--n-sims", type=int, default=200,
                        help="MCTS simulations per turn")
    parser.add_argument("--train-batch-size", type=int, default=512,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--eval-games", type=int, default=100,
                        help="Evaluation games per round")
    parser.add_argument("--eval-sims", type=int, default=200,
                        help="MCTS sims for evaluation bot")
    parser.add_argument("--minimax-time", type=float, default=0.1,
                        help="Time limit for minimax opponent")
    parser.add_argument("--window", type=int, default=4,
                        help="Sliding window of rounds for training data")
    parser.add_argument("--decay", type=float, default=0.75,
                        help="Exponential weight decay per round age")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint to resume from")
    parser.add_argument("--output-dir", type=str,
                        default="learned_eval/mcts_results",
                        help="Output directory for checkpoints")
    parser.add_argument("--data-dir", type=str,
                        default="learned_eval/data/selfplay",
                        help="Directory for self-play data")
    parser.add_argument("--amp", action="store_true",
                        help="Use automatic mixed precision")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument("--num-blocks", type=int, default=10)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--viewer", action="store_true",
                        help="Launch live game viewer in browser")
    parser.add_argument("--viewer-port", type=int, default=8765,
                        help="Port for game viewer (default 8765)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    model = HexResNet(
        num_blocks=args.num_blocks, num_filters=args.num_filters
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.num_blocks} blocks, {args.num_filters} filters, "
          f"{n_params:,} params")



    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-4)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Resume
    start_round = 0
    best_win_rate = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scaler and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_round = ckpt.get("round", 0) + 1
        best_win_rate = ckpt.get("best_win_rate", 0.0)
        print(f"Resumed from {args.resume} (round {start_round})")
    elif os.path.exists(os.path.join(args.output_dir, "best.pt")):
        # Auto-resume from best checkpoint
        ckpt_path = os.path.join(args.output_dir, "best.pt")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scaler and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_round = ckpt.get("round", 0) + 1
        best_win_rate = ckpt.get("best_win_rate", 0.0)
        print(f"Auto-resumed from {ckpt_path} (round {start_round})")

    # wandb
    use_wandb = args.wandb and HAS_WANDB
    if use_wandb:
        try:
            wandb.init(
                project="hex-mcts-selfplay",
                config=vars(args),
                name=f"mcts_r{start_round}",
            )
        except Exception as e:
            print(f"WARNING: wandb init failed ({e})")
            use_wandb = False

    # Game viewer
    viewer = None
    if args.viewer:
        from learned_eval.game_viewer import GameViewer
        viewer = GameViewer(port=args.viewer_port)
        viewer.start()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    for round_num in range(start_round, start_round + args.rounds):
        print(f"\n{'='*60}")
        print(f"  ROUND {round_num}")
        print(f"{'='*60}")
        t0 = time.time()

        # --- 1. Self-play ---
        from learned_eval.self_play import SelfPlayManager, COMPLETED_PER_ROUND
        print(f"\n--- Self-play: {COMPLETED_PER_ROUND} games ---")
        model.eval()
        manager = SelfPlayManager(
            model, device,
            batch_size=args.batch_size,
            n_sims=args.n_sims,
            data_dir=args.data_dir,
            viewer=viewer,
        )
        examples = manager.generate(round_num)
        manager.save_round(examples, round_num, args.data_dir)
        t_gen = time.time() - t0

        # --- 2. Train ---
        print(f"\n--- Training (window={args.window}, decay={args.decay}) ---")
        t1 = time.time()
        dataset = load_selfplay_rounds(args.data_dir, round_num,
                                       window=args.window,
                                       decay=args.decay)
        losses = train_one_epoch(
            model, optimizer, dataset, device,
            batch_size=args.train_batch_size,
            use_amp=use_amp,
            scaler=scaler,
        )
        avg_loss, avg_vloss, avg_ploss, avg_ml, avg_chain = losses
        t_train = time.time() - t1
        print(f"  Loss: {avg_loss:.4f} (value={avg_vloss:.4f}, "
              f"policy={avg_ploss:.4f}, ml={avg_ml:.4f}, "
              f"chain={avg_chain:.4f})")

        # --- 3. Checkpoint ---
        ckpt_path = save_checkpoint(model, optimizer, scaler, round_num,
                                    args.output_dir, best_win_rate)

        # --- 4. Evaluate ---
        print(f"\n--- Evaluation: {args.eval_games} games vs Minimax ---")
        t2 = time.time()
        win_rate = evaluate_vs_minimax(
            ckpt_path, n_games=args.eval_games,
            n_sims=args.eval_sims,
            minimax_time=args.minimax_time,
            device=str(device),
        )
        t_eval = time.time() - t2

        if win_rate > best_win_rate:
            best_win_rate = win_rate
            print(f"  New best win rate: {100 * best_win_rate:.1f}%")

        t_total = time.time() - t0

        # --- Log ---
        print(f"\n  Round {round_num} summary:")
        print(f"    Examples: {len(examples):,}")
        print(f"    Loss: {avg_loss:.4f}")
        print(f"    Win rate: {100 * win_rate:.1f}%")
        print(f"    Time: {t_gen:.0f}s gen + {t_train:.0f}s train "
              f"+ {t_eval:.0f}s eval = {t_total:.0f}s total")

        if use_wandb:
            wandb.log({
                "round": round_num,
                "loss": avg_loss,
                "value_loss": avg_vloss,
                "policy_loss": avg_ploss,
                "moves_left_loss": avg_ml,
                "chain_loss": avg_chain,
                "win_rate": win_rate,
                "best_win_rate": best_win_rate,
                "examples": len(examples),
                "time_gen": t_gen,
                "time_train": t_train,
                "time_eval": t_eval,
            })

    print(f"\n{'='*60}")
    print(f"  Training complete. Best win rate: {100 * best_win_rate:.1f}%")
    print(f"{'='*60}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
