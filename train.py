"""
PAR Training Script

Usage:
    # Single GPU
    python train.py --config configs/par_60m.yaml --data_dir /path/to/afdb

    # Multi-GPU (8× H100, as in paper)
    torchrun --nproc_per_node=8 train.py --config configs/par_400m.yaml \
             --data_dir /path/to/afdb --wandb

    # PDB fine-tuning
    python train.py --config configs/par_400m.yaml \
             --data_dir /path/to/pdb_designable \
             --resume checkpoints/par_400m_afdb.pt \
             --max_steps 5000 --min_length 32 --max_length 256
"""

import argparse
import os
import time
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

try:
    import yaml
except ImportError:
    yaml = None

from models.par import PAR
from data.dataset import NumpyBackboneDataset, RandomBackboneDataset, collate_backbones
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train PAR protein backbone model")
    parser.add_argument("--config",      type=str, default="configs/par_60m.yaml")
    parser.add_argument("--data_dir",    type=str, default=None,
                        help="Path to directory with .npy backbone files. "
                             "If not given, uses synthetic random data for testing.")
    parser.add_argument("--output_dir",  type=str, default="checkpoints")
    parser.add_argument("--resume",      type=str, default=None,
                        help="Checkpoint to resume from")
    parser.add_argument("--max_steps",   type=int, default=None,
                        help="Override max_steps from config")
    parser.add_argument("--batch_size",  type=int, default=None,
                        help="Override batch_size per GPU")
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--min_length",  type=int, default=None)
    parser.add_argument("--max_length",  type=int, default=None)
    parser.add_argument("--wandb",       action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--debug",       action="store_true",
                        help="Quick smoke test with 100 random samples")
    parser.add_argument("--local_rank",  type=int, default=-1)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    if yaml is None:
        raise ImportError("pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Setup distributed training
# ---------------------------------------------------------------------------

def setup_distributed():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_world_size(), True
    return 0, 1, False


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    local_rank, world_size, distributed = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = (local_rank == 0)

    # ---- Load config -------------------------------------------------------
    cfg = load_config(args.config)
    model_cfg   = cfg["model"]
    train_cfg   = cfg["training"]

    # CLI overrides
    if args.max_steps   is not None: train_cfg["max_steps"]   = args.max_steps
    if args.batch_size  is not None: train_cfg["batch_size"]  = args.batch_size
    if args.lr          is not None: train_cfg["lr"]          = args.lr
    if args.min_length  is not None: train_cfg["min_length"]  = args.min_length
    if args.max_length  is not None: train_cfg["max_length"]  = args.max_length

    if is_main:
        print(f"[Config] {cfg}")
        os.makedirs(args.output_dir, exist_ok=True)

    # ---- Build model -------------------------------------------------------
    model = PAR(**model_cfg).to(device)
    if is_main:
        params = model.count_parameters()
        print(f"[Model] AR: {params['ar_transformer']/1e6:.1f}M  "
              f"Decoder: {params['flow_decoder']/1e6:.1f}M  "
              f"Total: {params['total']/1e6:.1f}M params")

    if distributed:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if distributed else model

    # ---- Optimizer ---------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])

    # ---- Resume ------------------------------------------------------------
    start_step = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        if is_main:
            print(f"[Resume] Loaded from {args.resume}, step {start_step}")

    # ---- Dataset -----------------------------------------------------------
    if args.debug or args.data_dir is None:
        dataset = RandomBackboneDataset(
            n_samples  = 100,
            min_length = train_cfg.get("min_length", 32),
            max_length = train_cfg.get("max_length", 256),
        )
        if is_main:
            print("[Dataset] Using synthetic random backbones (debug mode)")
    else:
        dataset = NumpyBackboneDataset(
            data_dir   = args.data_dir,
            min_length = train_cfg.get("min_length", 32),
            max_length = train_cfg.get("max_length", 256),
        )

    sampler = DistributedSampler(dataset) if distributed else None
    loader  = DataLoader(
        dataset,
        batch_size  = train_cfg["batch_size"],
        shuffle     = (sampler is None),
        sampler     = sampler,
        num_workers = 0 if args.debug else 4,
        collate_fn  = collate_backbones,
        pin_memory  = True,
        drop_last   = True,
    )

    # ---- WandB -------------------------------------------------------------
    if args.wandb and is_main:
        try:
            import wandb
            wandb.init(project="par-protein", config=cfg)
        except ImportError:
            print("[WandB] Not installed; skipping.")
            args.wandb = False

    # ---- Training loop -----------------------------------------------------
    model.train()
    step       = start_step
    max_steps  = train_cfg["max_steps"]
    log_every  = train_cfg.get("log_every",  100)
    save_every = train_cfg.get("save_every", 5000)
    grad_clip  = train_cfg.get("grad_clip",  1.0)

    t0 = time.time()

    while step < max_steps:
        if distributed:
            sampler.set_epoch(step // len(loader))

        for batch in loader:
            if step >= max_steps:
                break
            if batch is None:
                continue

            coords  = batch["coords"].to(device)    # (B, L_max, 3)
            lengths = batch["lengths"].to(device)   # (B,)

            # Forward pass
            optimizer.zero_grad()
            out = model(coords, lengths)
            loss = out["loss"]

            # Backward
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            step += 1

            # Logging
            if is_main and step % log_every == 0:
                elapsed = time.time() - t0
                per_scale = [f"{l.item():.4f}" for l in out["per_scale_losses"]]
                print(f"[Step {step:>7d}/{max_steps}] "
                      f"loss={loss.item():.4f}  "
                      f"per_scale={per_scale}  "
                      f"elapsed={elapsed:.1f}s")

                if args.wandb:
                    import wandb
                    log_dict = {"train/loss": loss.item(), "step": step}
                    for i, sl in enumerate(out["per_scale_losses"]):
                        log_dict[f"train/loss_scale{i}"] = sl.item()
                    wandb.log(log_dict)

            # Checkpoint
            if is_main and step % save_every == 0:
                ckpt_path = os.path.join(
                    args.output_dir, f"par_step{step:07d}.pt"
                )
                torch.save({
                    "step":                step,
                    "model_state_dict":    raw_model.state_dict(),
                    "optimizer_state_dict":optimizer.state_dict(),
                    "config":              model_cfg,
                }, ckpt_path)
                print(f"[Checkpoint] Saved to {ckpt_path}")

    if is_main:
        final_path = os.path.join(args.output_dir, "par_final.pt")
        torch.save({
            "step":             step,
            "model_state_dict": raw_model.state_dict(),
            "config":           model_cfg,
        }, final_path)
        print(f"[Done] Final checkpoint saved to {final_path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
