"""MCTS self-play training loop: generate → train → evaluate.

Usage:
    python -m learned_eval.train_loop --rounds 10 --games-per-round 5000 --amp
"""

import argparse
import json
import math
import os
import time

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from game import HexGame, Player
from learned_eval.resnet_model import HexResNet, board_to_planes

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ---------------------------------------------------------------------------
# Training on self-play data
# ---------------------------------------------------------------------------

def load_selfplay_rounds(data_dir: str, current_round: int,
                         window: int = 3) -> TensorDataset:
    """Load the last `window` rounds of self-play data into a TensorDataset.

    Each example has: board (JSON), current_player, pair_visits (JSON), value_target.
    We convert to planes + visit distribution + value target.
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

    # Pre-compute all planes and visit distributions
    # Use dynamic sizing: find max board size across all examples
    all_planes = []
    all_visit_dists = []
    all_values = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing",
                       unit="ex", mininterval=2):
        board_dict = {
            tuple(int(x) for x in k.split(",")): v
            for k, v in json.loads(row["board"]).items()
        }
        cp = Player(row["current_player"])

        planes, off_q, off_r, h, w = board_to_planes(board_dict, cp)
        N = h * w

        # Parse visit distribution
        pair_visits_raw = json.loads(row["pair_visits"])
        total_visits = sum(pair_visits_raw.values())

        visit_dist = torch.zeros(N * N)
        if total_visits > 0:
            for key, count in pair_visits_raw.items():
                parts = key.split(",")
                a, b = int(parts[0]), int(parts[1])
                if 0 <= a < N and 0 <= b < N:
                    visit_dist[a * N + b] = count / total_visits

        all_planes.append((planes, h, w))
        all_visit_dists.append((visit_dist, N))
        all_values.append(row["value_target"])

    # Find max dimensions for uniform padding
    max_h = max(h for _, h, _ in all_planes)
    max_w = max(w for _, _, w in all_planes)
    max_N = max_h * max_w

    # Pad everything to uniform size
    n = len(all_planes)
    planes_tensor = torch.zeros(n, 2, max_h, max_w)
    mask_tensor = torch.zeros(n, 1, max_h, max_w)
    visit_tensor = torch.zeros(n, max_N * max_N)
    value_tensor = torch.zeros(n)

    for i, ((p, h, w), (vd, N), val) in enumerate(
        zip(all_planes, all_visit_dists, all_values)
    ):
        planes_tensor[i, :, :h, :w] = p
        mask_tensor[i, 0, :h, :w] = 1.0
        # Remap visit distribution from old N×N to new max_N×max_N
        if N == max_N:
            visit_tensor[i, :N*N] = vd
        else:
            # Need to remap indices: old (a,b) in N×N → new in max_N×max_N
            old_N = int(math.isqrt(len(vd)))
            for idx in vd.nonzero(as_tuple=True)[0]:
                old_a = idx.item() // old_N
                old_b = idx.item() % old_N
                # Convert old grid coords to (row, col)
                old_row_a, old_col_a = old_a // w, old_a % w
                old_row_b, old_col_b = old_b // w, old_b % w
                new_a = old_row_a * max_w + old_col_a
                new_b = old_row_b * max_w + old_col_b
                if new_a < max_N and new_b < max_N:
                    visit_tensor[i, new_a * max_N + new_b] = vd[idx].item()
        value_tensor[i] = val

    return TensorDataset(planes_tensor, mask_tensor, visit_tensor, value_tensor)


def compute_selfplay_loss(value_pred, pair_logits, visit_dist, value_target,
                          mask, value_weight=1.0, policy_weight=1.0):
    """KL divergence policy loss + MSE value loss for self-play training.

    visit_dist: [B, N²] visit-count probability distribution (target)
    pair_logits: [B, N, N] raw pair logits from model
    """
    B, N, _ = pair_logits.shape

    value_loss = F.mse_loss(value_pred, value_target)

    # Policy: KL divergence with visit distribution
    flat_logits = pair_logits.reshape(B, -1)  # [B, N²]
    log_probs = F.log_softmax(flat_logits, dim=-1)

    # Only compute loss where visit_dist > 0 to avoid 0 * log(0)
    # KL = sum(target * (log(target) - log(pred)))
    # Equivalent to: -sum(target * log_pred) + const (since target is fixed)
    policy_loss = -(visit_dist * log_probs).sum(dim=-1).mean()

    total = value_weight * value_loss + policy_weight * policy_loss
    return total, value_loss, policy_loss


def train_one_epoch(model, optimizer, dataset, device, batch_size=512,
                    use_amp=False, scaler=None, grad_clip=50.0):
    """Train one epoch on self-play data."""
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)

    total_loss = 0.0
    total_vloss = 0.0
    total_ploss = 0.0
    n_batches = 0

    for planes, mask, visit_dist, value_target in tqdm(loader, desc="Training",
                                                        unit="batch"):
        planes = planes.to(device)
        mask = mask.to(device)
        visit_dist = visit_dist.to(device)
        value_target = value_target.to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast("cuda"):
                value_pred, pair_logits = model(planes, mask)
                loss, vloss, ploss = compute_selfplay_loss(
                    value_pred, pair_logits, visit_dist, value_target, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            value_pred, pair_logits = model(planes, mask)
            loss, vloss, ploss = compute_selfplay_loss(
                value_pred, pair_logits, visit_dist, value_target, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        total_vloss += vloss.item()
        total_ploss += ploss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_vloss = total_vloss / max(n_batches, 1)
    avg_ploss = total_ploss / max(n_batches, 1)
    return avg_loss, avg_vloss, avg_ploss


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
    parser.add_argument("--games-per-round", type=int, default=5000,
                        help="Self-play games per round")
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
    parser.add_argument("--window", type=int, default=3,
                        help="Sliding window of rounds for training data")
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
        model.load_state_dict(ckpt["model_state_dict"])
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
        model.load_state_dict(ckpt["model_state_dict"])
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

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    for round_num in range(start_round, start_round + args.rounds):
        print(f"\n{'='*60}")
        print(f"  ROUND {round_num}")
        print(f"{'='*60}")
        t0 = time.time()

        # --- 1. Self-play ---
        print(f"\n--- Self-play: {args.games_per_round} games ---")
        from learned_eval.self_play import SelfPlayManager
        model.eval()
        manager = SelfPlayManager(
            model, device,
            n_games=args.games_per_round,
            batch_size=args.batch_size,
            n_sims=args.n_sims,
        )
        examples = manager.generate(round_num)
        manager.save_round(examples, round_num, args.data_dir)
        t_gen = time.time() - t0

        # --- 2. Train ---
        print(f"\n--- Training (window={args.window}) ---")
        t1 = time.time()
        dataset = load_selfplay_rounds(args.data_dir, round_num,
                                       window=args.window)
        avg_loss, avg_vloss, avg_ploss = train_one_epoch(
            model, optimizer, dataset, device,
            batch_size=args.train_batch_size,
            use_amp=use_amp,
            scaler=scaler,
        )
        t_train = time.time() - t1
        print(f"  Loss: {avg_loss:.4f} (value={avg_vloss:.4f}, "
              f"policy={avg_ploss:.4f})")

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
