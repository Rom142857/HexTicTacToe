"""Train HexResNet with pair attention policy on distillation data.

Trains a dual-head ResNet: value (win rate) + pair policy (attention over cell
embeddings producing N×N pair logits). Only double-move examples are used for
policy training (single-move mid-turn examples are filtered out).

Usage:
  python -m learned_eval.train_resnet --epochs 5 --batch-size 512 --amp --wandb
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
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from learned_eval.resnet_model import BOARD_SIZE, HexResNet


# ---------------------------------------------------------------------------
# Preprocessing: parquet → .npy cache (single-process, ~5 GB RAM)
# ---------------------------------------------------------------------------

def preprocess_to_cache(parquet_path, cache_dir):
    """Parse parquet into .npy files. Single-process to keep RAM low."""
    df = pd.read_parquet(parquet_path)
    n = len(df)
    bs = BOARD_SIZE
    print(f"  Preprocessing {n:,} rows to {cache_dir} (one-time)...")
    os.makedirs(cache_dir, exist_ok=True)

    planes_mm = np.lib.format.open_memmap(
        os.path.join(cache_dir, "planes.npy"), mode="w+",
        dtype=np.uint8, shape=(n, 2, bs, bs),
    )
    moves_mm = np.lib.format.open_memmap(
        os.path.join(cache_dir, "moves.npy"), mode="w+",
        dtype=np.int16, shape=(n, 2),
    )
    wins_mm = np.lib.format.open_memmap(
        os.path.join(cache_dir, "wins.npy"), mode="w+",
        dtype=np.int8, shape=(n,),
    )
    gids_mm = np.lib.format.open_memmap(
        os.path.join(cache_dir, "game_ids.npy"), mode="w+",
        dtype=np.int32, shape=(n,),
    )

    boards = df["board"].values
    cps = df["current_player"].values
    moves_col = df["moves"].values
    win_col = df["win_score"].values
    gid_col = df["game_id"].values
    del df

    for i in tqdm(range(n), desc="  Parsing", unit="pos", mininterval=2):
        board_dict = {
            tuple(int(x) for x in k.split(",")): v
            for k, v in json.loads(boards[i]).items()
        }
        cp = int(cps[i])
        off_q, off_r = 0, 0

        if board_dict:
            qs = [q for q, _r in board_dict]
            rs = [r for _q, r in board_dict]
            min_q, max_q = min(qs), max(qs)
            min_r, max_r = min(rs), max(rs)
            off_q = (bs - (max_q - min_q + 1)) // 2 - min_q
            off_r = (bs - (max_r - min_r + 1)) // 2 - min_r

            for (q, r), player in board_dict.items():
                gq, gr = q + off_q, r + off_r
                if player == cp:
                    planes_mm[i, 0, gq, gr] = 1
                else:
                    planes_mm[i, 1, gq, gr] = 1

        raw_moves = moves_col[i]
        move_indices = []
        for m in raw_moves:
            gq = int(m[0]) + off_q
            gr = int(m[1]) + off_r
            if 0 <= gq < bs and 0 <= gr < bs:
                move_indices.append(gq * bs + gr)

        moves_mm[i, 0] = move_indices[0] if len(move_indices) >= 1 else 0
        moves_mm[i, 1] = move_indices[1] if len(move_indices) >= 2 else -1
        wins_mm[i] = 1 if float(win_col[i]) > 0 else -1
        gids_mm[i] = int(gid_col[i])

    for mm in (planes_mm, moves_mm, wins_mm, gids_mm):
        mm.flush()

    with open(os.path.join(cache_dir, "DONE"), "w") as f:
        f.write(str(n))

    size_mb = (planes_mm.nbytes + moves_mm.nbytes + wins_mm.nbytes) / 1e6
    print(f"  Cache written: {size_mb:.0f} MB")


# ---------------------------------------------------------------------------
# Loading: cache → CPU tensors, filtered to double-move only for pair policy
# ---------------------------------------------------------------------------

def load_data(parquet_path, cache_dir, val_fraction=0.2, seed=42):
    """Load cache, filter to double-move examples, split by game_id."""
    done_path = os.path.join(cache_dir, "DONE")
    if not os.path.exists(done_path):
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
        preprocess_to_cache(parquet_path, cache_dir)

    print("Loading cache...")
    moves_mm = np.load(os.path.join(cache_dir, "moves.npy"), mmap_mode="r")
    gids_mm = np.load(os.path.join(cache_dir, "game_ids.npy"), mmap_mode="r")
    n_total = len(moves_mm)

    # Filter to double-move examples only (m2 >= 0)
    has_pair = moves_mm[:, 1] >= 0
    pair_idx = np.where(has_pair)[0]
    print(f"  {n_total:,} total rows, {len(pair_idx):,} double-move "
          f"({100*len(pair_idx)/n_total:.0f}%)")

    # Split by game_id (on filtered set)
    gids_filtered = np.array(gids_mm[pair_idx])
    unique_gids = sorted(set(gids_filtered.tolist()))
    rng = random.Random(seed)
    rng.shuffle(unique_gids)
    n_val_games = max(1, int(len(unique_gids) * val_fraction))
    val_set = set(unique_gids[:n_val_games])

    val_mask = np.isin(gids_filtered, list(val_set))
    train_sub = pair_idx[~val_mask]
    val_sub = pair_idx[val_mask]
    del gids_filtered, gids_mm
    print(f"  Split: {len(train_sub):,} train / {len(val_sub):,} val "
          f"({len(unique_gids) - n_val_games} / {n_val_games} games)")

    # Load data
    planes_mm = np.load(os.path.join(cache_dir, "planes.npy"), mmap_mode="r")
    wins_mm = np.load(os.path.join(cache_dir, "wins.npy"), mmap_mode="r")

    def make_dataset(idx):
        idx = np.sort(idx)
        p = torch.from_numpy(np.array(planes_mm[idx], dtype=np.float32))
        m = torch.from_numpy(np.array(moves_mm[idx], dtype=np.int64))
        w = torch.from_numpy(np.array(wins_mm[idx], dtype=np.float32))
        return TensorDataset(p, m, w)

    print("  Loading train split...")
    train_ds = make_dataset(train_sub)
    print("  Loading val split...")
    val_ds = make_dataset(val_sub)

    train_gb = sum(t.nbytes for t in train_ds.tensors) / 1e9
    val_gb = sum(t.nbytes for t in val_ds.tensors) / 1e9
    print(f"  CPU RAM: {train_gb:.1f} GB train + {val_gb:.1f} GB val")

    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(value_pred, pair_logits, wins, moves,
                 value_weight=1.0, policy_weight=1.0, entropy_weight=0.01):
    """Pair policy CE + value MSE + entropy regularization.

    pair_logits: [B, N, N] — symmetrized pair scores
    moves: [B, 2] — (m1, m2) flat cell indices, both guaranteed >= 0
    """
    B, N, _ = pair_logits.shape
    m1 = moves[:, 0]
    m2 = moves[:, 1]

    value_loss = F.mse_loss(value_pred, wins)

    # Pair CE: target is flat index m1*N + m2
    flat_logits = pair_logits.reshape(B, -1)  # [B, N²]
    pair_target = m1 * N + m2
    policy_loss = F.cross_entropy(flat_logits, pair_target)

    # Entropy regularization
    if entropy_weight > 0:
        probs = F.softmax(flat_logits.float(), dim=-1)
        entropy = -(probs * probs.clamp(min=1e-10).log()).sum(dim=-1).mean()
        entropy_loss = -entropy
        total = (value_weight * value_loss + policy_weight * policy_loss
                 + entropy_weight * entropy_loss)
    else:
        entropy = torch.tensor(0.0, device=flat_logits.device)
        total = value_weight * value_loss + policy_weight * policy_loss

    return total, value_loss, policy_loss, entropy


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cache_dir = os.path.splitext(args.input)[0] + "_cache"
    train_ds, val_ds = load_data(args.input, cache_dir, args.val_fraction)

    if args.overfit_batches > 0:
        n = min(args.overfit_batches * args.batch_size, len(train_ds))
        train_ds = torch.utils.data.Subset(train_ds, range(n))
        val_ds = torch.utils.data.Subset(val_ds, range(min(n, len(val_ds))))
        print(f"  Overfit mode: {len(train_ds)} train / {len(val_ds)} val")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    model = HexResNet(
        num_blocks=args.num_blocks, num_filters=args.num_filters,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.num_blocks} blocks, {args.num_filters} filters, "
          f"{n_params:,} params")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    ckpt_path = os.path.join(args.output_dir, "checkpoint.pt")

    # wandb
    use_wandb = args.wandb and HAS_WANDB
    if use_wandb:
        try:
            wandb.init(
                project="hex-tictactoe-resnet",
                config=vars(args),
                name=f"pair_b{args.num_blocks}_f{args.num_filters}_lr{args.lr}",
            )
            wandb.watch(model, log="gradients", log_freq=200)
        except Exception as e:
            print(f"WARNING: wandb init failed ({e}), continuing without wandb")
            use_wandb = False

    # Resume
    start_epoch = 0
    global_step = 0
    resume_path = args.resume or (ckpt_path if os.path.exists(ckpt_path) else None)
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scaler and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        global_step = ckpt.get("global_step", 0)
        print(f"Resumed from {resume_path} (epoch {start_epoch}, "
              f"step {global_step}, best_val={best_val_loss:.4f})")

    # Scheduler (created after subsetting so step count is correct)
    import math
    n_batches = len(train_loader)
    total_steps = args.epochs * n_batches

    def lr_lambda(step):
        progress = step / max(total_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if resume_path and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    log_every = args.log_every
    n_train = len(train_ds)
    n_val = len(val_ds)
    N = BOARD_SIZE * BOARD_SIZE
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_v = 0.0
        train_p = 0.0
        n_seen = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for planes, moves, wins in pbar:
            planes = planes.to(device, non_blocking=True)
            moves = moves.to(device, non_blocking=True)
            wins = wins.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    v_pred, pair_logits, _, _ = model(planes)
                    loss, v_loss, p_loss, ent = compute_loss(
                        v_pred, pair_logits, wins, moves,
                        args.value_weight, args.policy_weight,
                        args.entropy_weight,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                v_pred, pair_logits, _, _ = model(planes)
                loss, v_loss, p_loss, ent = compute_loss(
                    v_pred, pair_logits, wins, moves,
                    args.value_weight, args.policy_weight,
                    args.entropy_weight,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
                optimizer.step()

            bs = len(wins)
            v_val = v_loss.item()
            p_val = p_loss.item()
            ent_val = ent.item()
            loss_val = loss.item()

            # NaN detection with full diagnostic dump
            if not (loss_val == loss_val):  # fast NaN check
                logit_fin = pair_logits[pair_logits.isfinite()]
                grad_norm = sum(
                    p.grad.norm().item() for p in model.parameters()
                    if p.grad is not None
                )
                # Which individual losses are NaN?
                nan_parts = []
                if not (v_val == v_val): nan_parts.append("value")
                if not (p_val == p_val): nan_parts.append("policy")
                if not (ent_val == ent_val): nan_parts.append("entropy")
                print(
                    f"\n  NaN at step {global_step} | "
                    f"nan_in: {','.join(nan_parts) or 'total_only'} | "
                    f"v={v_val:.4f} p={p_val:.4f} H={ent_val:.4f} | "
                    f"logits: [{pair_logits.min().item():.1f}, "
                    f"{pair_logits.max().item():.1f}] "
                    f"inf={pair_logits.isinf().sum().item()} "
                    f"nan={pair_logits.isnan().sum().item()} | "
                    f"finite_logits: [{logit_fin.min().item():.1f}, "
                    f"{logit_fin.max().item():.1f}] "
                    f"mean={logit_fin.mean().item():.2f} | "
                    f"value_pred: [{v_pred.min().item():.3f}, "
                    f"{v_pred.max().item():.3f}] | "
                    f"grad_norm={grad_norm:.1f} | "
                    f"scaler={scaler.get_scale():.0f}" if scaler else ""
                )
                continue  # skip accumulation to avoid poisoning averages

            train_v += v_val * bs
            train_p += p_val * bs
            n_seen += bs
            global_step += 1

            # Periodic health check — track drift before it becomes NaN
            if global_step % 200 == 0:
                logit_abs_max = pair_logits.detach().abs().max().item()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float("inf")
                ).item()
                qk_norms = {}
                for name, param in model.named_parameters():
                    if "pair_head" in name and "proj" in name:
                        qk_norms[name.split(".")[-1]] = (
                            f"{param.data.norm().item():.2f}"
                        )
                pbar.set_postfix(
                    v=f"{train_v/n_seen:.4f}",
                    p=f"{train_p/n_seen:.4f}",
                    H=f"{ent_val:.2f}",
                    logit_max=f"{logit_abs_max:.1f}",
                    gnorm=f"{grad_norm:.1f}",
                    **qk_norms,
                )
            elif global_step % 50 == 0:
                pbar.set_postfix(
                    v=f"{train_v/n_seen:.4f}",
                    p=f"{train_p/n_seen:.4f}",
                    H=f"{ent_val:.2f}",
                )

            scheduler.step()

            if use_wandb and global_step % log_every == 0:
                logit_abs_max = pair_logits.detach().abs().max().item()
                wandb.log({
                    "step": global_step,
                    "train/value_loss_step": v_val,
                    "train/policy_loss_step": p_val,
                    "train/total_loss_step": loss_val,
                    "train/entropy": ent_val,
                    "train/logit_abs_max": logit_abs_max,
                    "lr": optimizer.param_groups[0]["lr"],
                }, step=global_step)

        # Validation
        model.eval()
        val_v = 0.0
        val_p = 0.0
        v_correct = 0
        pair_correct = 0
        either_correct = 0

        with torch.no_grad():
            for planes, moves, wins in val_loader:
                planes = planes.to(device, non_blocking=True)
                moves = moves.to(device, non_blocking=True)
                wins = wins.to(device, non_blocking=True)

                v_pred, pair_logits, _, _ = model(planes)
                _, vl, pl, _ = compute_loss(
                    v_pred, pair_logits, wins, moves,
                    args.value_weight, args.policy_weight,
                )
                bs = len(wins)
                val_v += vl.item() * bs
                val_p += pl.item() * bs
                v_correct += ((v_pred > 0) == (wins > 0)).sum().item()

                # Pair accuracy: top predicted pair matches target (either order)
                m1 = moves[:, 0]
                m2 = moves[:, 1]
                flat = pair_logits.reshape(bs, -1)
                top_flat = flat.argmax(dim=-1)
                pred_a = top_flat // N
                pred_b = top_flat % N
                exact = ((pred_a == m1) & (pred_b == m2))
                flipped = ((pred_a == m2) & (pred_b == m1))
                pair_correct += (exact | flipped).sum().item()

                # Either-cell accuracy: at least one predicted cell is correct
                a_in = (pred_a == m1) | (pred_a == m2)
                b_in = (pred_b == m1) | (pred_b == m2)
                either_correct += (a_in | b_in).sum().item()

        val_combined = val_v / n_val + val_p / n_val
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"  train v={train_v/n_train:.4f} p={train_p/n_train:.4f} | "
            f"val v={val_v/n_val:.4f} p={val_p/n_val:.4f} | "
            f"v_acc={v_correct/n_val:.3f} pair_acc={pair_correct/n_val:.3f} "
            f"either={either_correct/n_val:.3f} | "
            f"lr={lr_now:.6f} | {elapsed:.0f}s"
        )

        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/value_loss": train_v / n_train,
                "train/policy_loss": train_p / n_train,
                "train/total_loss": train_v / n_train + train_p / n_train,
                "val/value_loss": val_v / n_val,
                "val/policy_loss": val_p / n_val,
                "val/total_loss": val_combined,
                "val/value_acc": v_correct / n_val,
                "val/pair_acc": pair_correct / n_val,
                "val/either_cell_acc": either_correct / n_val,
            }, step=global_step)

        # Checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "best_val_loss": min(best_val_loss, val_combined),
            "global_step": global_step,
            "args": vars(args),
        }
        tmp_path = ckpt_path + ".tmp"
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, ckpt_path)

        if val_combined < best_val_loss:
            best_val_loss = val_combined
            print(f"  -> New best (val_loss={val_combined:.4f})")

    print(f"\nDone in {time.time()-t0:.0f}s. Best val_loss={best_val_loss:.4f}")
    if use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train HexResNet (pair policy)")
    parser.add_argument("--input", default=os.path.join(
        os.path.dirname(__file__), "data", "distill_100k.parquet"))
    parser.add_argument("--output-dir", default=os.path.join(
        os.path.dirname(__file__), "resnet_results"))
    parser.add_argument("--num-blocks", type=int, default=10)
    parser.add_argument("--num-filters", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--entropy-weight", type=float, default=0.01,
                        help="Entropy regularization weight (default: 0.01)")
    parser.add_argument("--amp", action="store_true",
                        help="Use mixed precision training")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint to resume from")
    parser.add_argument("--overfit-batches", type=int, default=0,
                        help="Train on N batches only (debug)")
    parser.add_argument("--wandb", action="store_true",
                        help="Log to Weights & Biases")
    parser.add_argument("--log-every", type=int, default=50,
                        help="Log to wandb every N steps (default: 50)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
