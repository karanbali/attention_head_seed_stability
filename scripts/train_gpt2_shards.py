#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sharded training for a TransformerLens GPT-2 config.

- Keeps your shard-to-shard checkpoint chaining.
- No hardcoded WandB key; opt-out via --no_wandb.
- Robust seeding & deterministic knobs.
- Safer dataset path resolution (SLURM_TMPDIR or --data_dir).
- Uniform save/load layout under --scratch_dir/<attn_type>/...
"""

import os
from pathlib import Path
import argparse
import json
import random
import numpy as np
import torch as t
from datasets import load_from_disk
from transformer_lens import HookedTransformer, loading_from_pretrained
from transformer_lens import HookedTransformerConfig
from transformer_lens.train import HookedTransformerTrainConfig
from datetime import datetime
import importlib
import wandb

# ------------------- CLI -------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    # Core
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--save_every", type=int, default=66285)

    # Model/config
    p.add_argument("--pretrained_name", type=str, default="gpt2",
                   help="Name for loading TLens pretrained config (e.g., gpt2).")
    p.add_argument("--attn_type", type=str, default="gpt2")
    p.add_argument("--inst_cfg", type=str, default="owt")

    # Sharding & IO
    p.add_argument("--shard", type=int, default=0, help="Current shard index (0 => first).")
    p.add_argument("--scratch_dir", type=str, default="[Replace with your Root directory]",
                   help="Root directory for checkpoints.")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Optional override to 'load_from_disk' dataset directory. "
                        "If not set, uses $SLURM_TMPDIR/partition_{shard} or ../partition_{shard}.")
    p.add_argument("--resume_from", type=str, default=None,
                   help="Path to a directory or a .pt file to resume from (overrides shard chaining).")

    # WandB
    p.add_argument("--wandb_project", type=str, default="gpt2")
    p.add_argument("--wandb_key", type=str, default="None", help="WandB API key.")
    # NEW: choose config file & config name
    p.add_argument("--config_module", type=str, default="model_configs",
                   help="Python module path that exposes CONFIGS dict (default: model_configs)")
    p.add_argument("--config_name", type=str, default="gpt2",
                   help="Name of the config to use (default: gpt2)")
    return p.parse_args()

args = parse_args()

# ------------------- WandB Setup -------------------

if args.wandb_key and args.wandb_key.lower() != "none":
    wandb.login(key=args.wandb_key)

# ------------------- Device -------------------

device = (
    t.device("cuda") if t.cuda.is_available()
    else (t.device("mps") if t.backends.mps.is_available() else t.device("cpu"))
)
print(f"[Device] {device}")
if device.type == "mps":
    print("[Warn] MPS can be slower / numerically different vs CUDA.")
if device.type == "cpu":
    print("[Warn] CUDA not available. Training will be slow.")


# ------------------- Load Model Config -------------------

def load_named_config(module_name: str, config_name: str) -> dict:
    """
    Import a module that defines CONFIGS: Dict[str, Dict[str, Any]]
    and return CONFIGS[config_name].
    """
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        raise ImportError(f"Could not import config module '{module_name}': {e}") from e

    if not hasattr(mod, "CONFIGS"):
        raise AttributeError(f"Module '{module_name}' does not define CONFIGS.")

    CONFIGS = getattr(mod, "CONFIGS")
    if config_name not in CONFIGS:
        available = ", ".join(sorted(CONFIGS.keys()))
        raise KeyError(f"Config '{config_name}' not found in {module_name}. Available: {available}")

    return dict(CONFIGS[config_name])  # copy so we can tweak

cfg_dict = load_named_config(args.config_module, args.config_name)

# ------------------- Paths -------------------

scratch = Path(args.scratch_dir).resolve()
model_root = scratch / args.config_name

run_dir = model_root / f"{args.attn_type}_seed{args.seed}_shard{args.shard}_epoch{args.epochs}_{args.inst_cfg}"
run_dir.mkdir(parents=True, exist_ok=True)
print(f"[IO] Save dir: {run_dir}")

# Save run metadata
with open(run_dir / "run_args.json", "w") as f:
    json.dump(vars(args), f, indent=2, sort_keys=True)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
with open(run_dir / f"started_{timestamp}.txt", "w") as f:
    f.write("run started\n")

# ------------------- Config & Model -------------------

# Load a TLens config from a pretrained name, but initialize fresh weights
cfg: HookedTransformerConfig = loading_from_pretrained.get_pretrained_model_config(args.pretrained_name)
cfg.seed = args.seed
cfg.init_weights = True  # initialize randomly rather than loading pretrained weights

model = HookedTransformer(cfg).to(device)
print(f"[Model] seed={model.cfg.seed} layers={cfg.n_layers} d_model={cfg.d_model} "
      f"heads={cfg.n_heads} d_head={cfg.d_head} d_mlp={getattr(cfg, 'd_mlp', None)} "
      f"n_ctx={cfg.n_ctx}")


# ------------------- Optional resume / shard chaining -------------------

def load_ckpt_into_model(path: Path):
    print(f"[Resume] Loading checkpoint: {path}")
    state = t.load(str(path), map_location="cpu")
    try:
        model.load_and_process_state_dict(state)
    except Exception:
        model.load_state_dict(state)

def prefer_full_state(dir_path: Path) -> Path | None:
    """Prefer a full-state checkpoint if present in a run dir, else weights-only."""
    full = dir_path / "final_state.pt"
    if full.exists(): return full
    w = dir_path / "final.pt"
    if w.exists(): return w
    return None

resume_ckpt: Path | None = None

# 1) Explicit resume override
if args.resume_from:
    p = Path(args.resume_from)
    if p.is_dir():
        candidate = prefer_full_state(p)
        if candidate is None:
            raise FileNotFoundError(f"--resume_from dir given but neither final_state.pt nor final.pt found in {p}")
        resume_ckpt = candidate
    elif p.is_file():
        resume_ckpt = p
    else:
        raise FileNotFoundError(f"--resume_from path not found: {p}")

# 2) Otherwise, if shard > 0, chain from previous shard
elif args.shard > 0:
    prev = args.shard - 1
    prev_dir = model_root / f"{args.attn_type}_seed{args.seed}_shard{prev}_epoch{args.epochs}_{args.inst_cfg}"
    candidate = prefer_full_state(prev_dir)
    if candidate is not None:
        resume_ckpt = candidate
        print(f"[Resume] Found previous shard checkpoint: {candidate}")
    else:
        print(f"[Resume] Previous shard checkpoint not found in {prev_dir}; starting fresh.")

# Only pre-load weights here if it's a weights-only file; full-state is loaded inside train()
if resume_ckpt is not None and resume_ckpt.name != "final_state.pt" and not resume_ckpt.name.startswith("state_"):
    load_ckpt_into_model(resume_ckpt)

# ------------------- Dataset -------------------

def resolve_dataset_path() -> Path:
    if args.data_dir:
        return Path(args.data_dir)
    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
    base = Path(slurm_tmpdir) if slurm_tmpdir else Path("..")
    return base / f"partition_{args.shard}"

dataset_path = resolve_dataset_path()
if not dataset_path.exists():
    raise FileNotFoundError(
        f"Dataset directory not found: {dataset_path}\n"
        f"Provide --data_dir to a HuggingFace load_from_disk dataset folder containing a 'tokens' column."
    )

tokenized_dataset = load_from_disk(str(dataset_path))
tokenized_dataset.set_format(type="torch", columns=["tokens"])
print(f"[Data] Loaded from {dataset_path} | len={len(tokenized_dataset)}")

# ------------------- Train config -------------------

if args.config_name.endswith("_wd"):
    COMMON_TRAIN = dict(
        batch_size=args.batch_size,
        lr=None,
        optimizer_name="AdamW",
        weight_decay=0.10,
        num_epochs=args.epochs,
        save_every=args.save_every,
        save_dir=str(run_dir),
    )
    TRAIN_CFGS = {**COMMON_TRAIN, "lr": cfg_dict["lr"]}
    train_config = HookedTransformerTrainConfig(**TRAIN_CFGS)
else:
    # Default training config: Adam optimizer, no weight decay
    train_config = HookedTransformerTrainConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_every=args.save_every,
        save_dir=str(run_dir),
    )

print("Training config:", train_config)


# ------------------- Train (minimal edits: AMP + full-state save/resume) -------------------

from dataclasses import dataclass
from typing import Optional

import torch
import torch.optim as optim
import wandb
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from transformer_lens import utils
from transformer_lens.HookedTransformer import HookedTransformer

# --- NEW: small helpers to save/load full state ---
def save_full_state(path: Path, model, optimizer, scheduler, scaler, epoch: int, global_step: int):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "global_step": global_step,
        },
        str(path),
    )

def load_full_state(path: Path, model, optimizer, scheduler, scaler):
    state = torch.load(str(path), map_location="cpu")
    sd = state.get("model_state_dict", state)
    try:
        model.load_and_process_state_dict(sd)
    except Exception:
        model.load_state_dict(sd)
    if optimizer is not None and state.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    if scaler is not None and state.get("scaler_state_dict") is not None:
        scaler.load_state_dict(state["scaler_state_dict"])
    return int(state.get("epoch", 0)), int(state.get("global_step", 0))


# Code has been adapted from transformer_lens.train.train()
def train(
    model: HookedTransformer,
    config: HookedTransformerTrainConfig,
    dataset: Dataset,
    *,                                # --- NEW: keyword-only additions ---
    resume_ckpt: Optional[Path] = None,
    save_dir: Optional[str] = None,
) -> HookedTransformer:
    """
    Trains an HookedTransformer model on an autoregressive language modeling task.
    """
    torch.manual_seed(config.seed)
    model.train()
    if config.wandb:
        if config.wandb_project_name is None:
            config.wandb_project_name = "easy-transformer"
        wandb.init(project=config.wandb_project_name, config=vars(config))

    if config.device is None:
        config.device = utils.get_device()

    optimizer: Optimizer
    if config.optimizer_name in ["Adam", "AdamW"]:
        if config.weight_decay is not None:
            optimizer = optim.AdamW(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        else:
            optimizer = optim.Adam(
                model.parameters(),
                lr=config.lr,
            )
    elif config.optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.lr,
            weight_decay=(config.weight_decay if config.weight_decay is not None else 0.0),
            momentum=config.momentum,
        )
    else:
        raise ValueError(f"Optimizer {config.optimizer_name} not supported")

    scheduler = None
    if config.warmup_steps and config.warmup_steps > 0:
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, step / config.warmup_steps),
        )

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model.to(config.device)

    use_amp = False
    autocast_dtype = None


    start_epoch = 1
    global_step = 0

    if resume_ckpt is not None:
        if resume_ckpt.name == "final_state.pt" or resume_ckpt.name.startswith("state_"):
            print(f"[Resume] Loading FULL state from: {resume_ckpt}")
            start_epoch, global_step = load_full_state(resume_ckpt, model, optimizer, scheduler, scaler)
            start_epoch = max(1, start_epoch)  # safety
        else:
            print(f"[Resume] (weights-only) {resume_ckpt} already loaded before train()")

    if save_dir is None and config.save_dir is not None:
        save_dir = config.save_dir

    print_every = config.print_every if config.print_every is not None else 100

    for epoch in tqdm(range(start_epoch, config.num_epochs + 1)):
        samples = 0
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            tokens = batch["tokens"].to(config.device)

            with t.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype):
                loss = model(tokens, return_type="loss")

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            if config.max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            global_step += 1
            samples += tokens.shape[0]

            if config.wandb:
                wandb.log({"train_loss": loss.item(),
                           "lr": optimizer.param_groups[0]["lr"],
                           "samples": samples, "epoch": epoch, "step": global_step})

            if config.print_every is not None and (global_step % print_every == 0):
                print(f"Epoch {epoch} Samples {samples} Step {global_step} Loss {loss.item()}")

            # --- NEW: periodic FULL-STATE checkpoint ---
            if (config.save_every is not None and save_dir is not None
                and global_step % config.save_every == 0):
                save_full_state(Path(save_dir) / f"state_{global_step}.pt",
                                model, optimizer, scheduler, scaler,
                                epoch=epoch, global_step=global_step)

            if config.max_steps is not None and step >= config.max_steps:
                break


    # --- NEW: final FULL-STATE checkpoint (alongside your weights-only save below) ---
    if save_dir is not None:
        save_full_state(Path(save_dir) / "final_state.pt",
                        model, optimizer, scheduler, scaler,
                        epoch=config.num_epochs, global_step=global_step)

    return model

print("[Train] Starting …")
try:
    model_trained = train(
        model,
        train_config,
        tokenized_dataset,
        resume_ckpt=resume_ckpt,          # <-- NEW: pass resume path
        save_dir=str(run_dir),            # <-- NEW: explicit save dir
    )
except KeyboardInterrupt:
    print("\n[Train] Interrupted. Saving partial weights …")
    t.save(model.state_dict(), run_dir / "interrupt.pt")
    raise

# ------------------- Save final -------------------

final_model_path = run_dir / "final.pt"
t.save(model_trained.state_dict(), final_model_path)
print(f"[Done] Saved to {final_model_path} (weights) and {run_dir/'final_state.pt'} (full state)")
