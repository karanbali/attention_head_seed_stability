# train_tlens.py
import os
import sys
from pathlib import Path
import argparse
import importlib
import random
import numpy as np
import pandas as pd
import torch as t
from datasets import load_from_disk
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
)
from transformer_lens.train import HookedTransformerTrainConfig
import wandb

# ------------------- Configuration & Argument Parsing -------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train a TransformerLens model from a named config.")

    parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility.")
    parser.add_argument("--config_name", type=str, required=True,
                        help="Key in CONFIGS to select model hyperparameters.")
    
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs (TOTAL target epochs after this run).")
    parser.add_argument("--prev_epoch", type=int, default=0,
                        help="Number of training epochs completed previously.")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--scratch_dir", type=str, default="[Replace with your Root directory]",
                        help="Scratch directory for saving models.")
    parser.add_argument("--wandb_key", type=str, default="None",
                        help="WandB API key.")

    # choose config file & config name
    parser.add_argument("--config_module", type=str, default="model_configs",
                        help="Python module path that exposes CONFIGS dict (default: model_configs)")
    return parser.parse_args()

args = parse_args()

# ------------------- WandB Setup -------------------

if args.wandb_key and args.wandb_key.lower() != "none":
    wandb.login(key=args.wandb_key)

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

# ------------------- Paths & Constants -------------------

SCRATCH = args.scratch_dir
SEED = args.seed
NUM_EPOCHS = args.epochs            # total target epochs after this run
prev_epoch = args.prev_epoch       # epochs done previously

# interpret epochs as total target epochs (not prev + new)
total_epochs = prev_epoch + NUM_EPOCHS

lr = args.lr
batch_size = args.batch_size
save_every = 66285  # unchanged

# Preserve original naming pieces, but incorporate the selected config:
attn_type = "causal_attn"
inst_cfg = "c4_gelu"  # left as-is (data/instruction tag)
model_name_for_dir = args.config_name

root_path = os.path.join(SCRATCH, "chkpts_wd", model_name_for_dir)
os.makedirs(root_path, exist_ok=True)

# ------------------- Device Selection -------------------

device = (
    t.device("mps") if t.backends.mps.is_available()
    else t.device("cuda") if t.cuda.is_available()
    else t.device("cpu")
)
print(f"[Device] {device}")

# ------------------- Model Configuration -------------------

# Build HookedTransformerConfig using the loaded config
ht_cfg = HookedTransformerConfig(
    n_layers=cfg_dict["n_layers"],
    d_model=cfg_dict["d_model"],
    n_heads=cfg_dict["n_heads"],
    d_head=cfg_dict["d_head"],
    d_mlp=cfg_dict.get("d_mlp", None),
    n_ctx=cfg_dict["n_ctx"],
    act_fn=cfg_dict.get("act_fn", "gelu"),
    d_vocab=cfg_dict["d_vocab"],
    init_weights=True,
    tokenizer_name=cfg_dict["tokenizer_name"],
    model_name=cfg_dict.get("model_name", args.config_name),
    attn_only=cfg_dict.get("attn_only", False),
    seed=SEED,
)

model = HookedTransformer(ht_cfg).to(device)
print(f"Model SEED: {model.cfg.seed}")
print(f"Loaded config: {args.config_name} from {args.config_module}")
print(f"Model dims: layers={ht_cfg.n_layers}, d_model={ht_cfg.d_model}, "
      f"heads={ht_cfg.n_heads}, d_head={ht_cfg.d_head}, d_mlp={ht_cfg.d_mlp}")

# ------------------- Save Path Construction -------------------

# New run's directory is keyed by the TOTAL target epochs (NUM_EPOCHS)
save_path = os.path.join(
    root_path,
    f"{attn_type}{'_only' if ht_cfg.attn_only else ''}"
    f"_l{ht_cfg.n_layers}_h{ht_cfg.n_heads}"
    f"_seed{SEED}_epoch{total_epochs}_{inst_cfg}"
)
os.makedirs(save_path, exist_ok=True)
print("Save path:", save_path)

# Previous run's directory (where we resume from)
prev_epoch_path = os.path.join(
    root_path,
    f"{attn_type}{'_only' if ht_cfg.attn_only else ''}"
    f"_l{ht_cfg.n_layers}_h{ht_cfg.n_heads}"
    f"_seed{SEED}_epoch{prev_epoch}_{inst_cfg}"
)
print("Previous epoch path:", prev_epoch_path)

# ------------------- Dataset Loading -------------------

def get_dataset_path():
    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
    if slurm_tmpdir:
        return Path(slurm_tmpdir) / "data"
    else:
        return Path("..") / "data"

dataset_path = get_dataset_path()
tokenized_dataset = load_from_disk(dataset_path)
tokenized_dataset.set_format(type="torch", columns=["tokens"])
print("Loaded dataset:", tokenized_dataset)

# ------------------- Training Configuration -------------------

# Choose which train_config to use: if config_name ends with "_wd"
if args.config_name.endswith("_wd"):
    COMMON_TRAIN = dict(
        batch_size=batch_size,
        lr=None,
        optimizer_name="AdamW",
        weight_decay=0.10,
        num_epochs=NUM_EPOCHS,
        save_every=save_every,
        save_dir=save_path,
    )
    
    TRAIN_CFGS = {**COMMON_TRAIN, "lr": cfg_dict["lr"]}
    train_config = HookedTransformerTrainConfig(**TRAIN_CFGS)
else:
    # Default training config: Adam optimizer, no weight decay
    train_config = HookedTransformerTrainConfig(
        num_epochs=NUM_EPOCHS,           # TOTAL target epochs
        batch_size=batch_size,
        lr=lr,
        save_every=save_every,
        save_dir=save_path,
    )

# ensure seed is in train_config for reproducibility
if hasattr(train_config, "seed"):
    train_config.seed = SEED

print("Training config:", train_config)

# ------------------- Full-state checkpoint helpers -------------------

def save_full_state(path: Path, model, optimizer, scheduler, epoch: int, global_step: int):
    """
    Save full training state so we can resume without losing optimizer/scheduler/etc.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "global_step": global_step,
        },
        str(path),
    )
    print(f"[Checkpoint] Saved full state to {path}")

def load_full_state(path: Path, model, optimizer, scheduler, device: t.device):
    """
    Load full training state (model + optimizer + scheduler + epoch + global_step).
    """
    path = Path(path)
    state = t.load(str(path), map_location=device)
    model_state = state.get("model_state_dict", state)
    model.load_state_dict(model_state)

    opt_state = state.get("optimizer_state_dict", None)
    if optimizer is not None and opt_state is not None:
        optimizer.load_state_dict(opt_state)

    sched_state = state.get("scheduler_state_dict", None)
    if scheduler is not None and sched_state is not None:
        scheduler.load_state_dict(sched_state)

    epoch = int(state.get("epoch", 0))
    global_step = int(state.get("global_step", 0))
    print(f"[Checkpoint] Loaded full state from {path} at epoch={epoch}, global_step={global_step}")
    return epoch, global_step

# ------------------- Custom train() with full-state resume -------------------
# Code has been adapted from transformer_lens.train.train()

def train(model: HookedTransformer,
          config: HookedTransformerTrainConfig,
          dataset) -> HookedTransformer:
    """
    Custom training loop that:
    - Saves and loads full training state (model + optimizer + scheduler + epoch + global_step).
    - Interprets config.num_epochs as *TOTAL target epochs* (matching --epochs).
    - Uses prev_epoch_path (global) to resume from previous epochs.
    """
    # Seeding
    seed = getattr(config, "seed", 1)
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)

    model.to(device)
    model.train()

    # WandB init
    if config.wandb:
        project = getattr(config, "wandb_project_name", None) or "transformer_lens"
        wandb.init(project=project, config=vars(config))

    # Dataloader
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Optimizer
    opt_name = getattr(config, "optimizer_name", "AdamW")
    weight_decay = getattr(config, "weight_decay", None)
    if opt_name in ["Adam", "AdamW"]:
        if weight_decay is not None:
            optimizer = t.optim.AdamW(
                model.parameters(),
                lr=config.lr,
                weight_decay=weight_decay,
            )
        else:
            optimizer = t.optim.Adam(
                model.parameters(),
                lr=config.lr,
            )
    elif opt_name == "SGD":
        momentum = getattr(config, "momentum", 0.0)
        optimizer = t.optim.SGD(
            model.parameters(),
            lr=config.lr,
            weight_decay=(weight_decay if weight_decay is not None else 0.0),
            momentum=momentum,
        )
    else:
        raise ValueError(f"Optimizer {opt_name} not supported")

    # Scheduler (simple linear warmup)
    warmup_steps = getattr(config, "warmup_steps", 0)
    scheduler = None    # type: ignore
    if warmup_steps and warmup_steps > 0:
        scheduler = t.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, float(step) / float(max(1, warmup_steps))),
        )

    use_amp = False
    autocast_dtype = None

    save_dir = Path(config.save_dir) if getattr(config, "save_dir", None) is not None else None
    start_epoch = 1
    global_step = 0

    # --------- NEW: resume from prev_epoch_path instead of save_dir ----------
    if prev_epoch > 0:
        prev_dir = Path(prev_epoch_path)
        if args.config_name.endswith("_wd"):
            prev_state_path = prev_dir / "final_state.pt"
            if prev_state_path.exists():
                print(f"[Resume] Found previous full-state checkpoint at {prev_state_path}, loading...")
                last_epoch, global_step = load_full_state(prev_state_path, model, optimizer, scheduler, device)
                # sanity check vs CLI
                if last_epoch != prev_epoch:
                    print(f"[Warn] Checkpoint epoch ({last_epoch}) != --prev_epoch ({prev_epoch}). "
                        f"Using checkpoint epoch={last_epoch}.")
                start_epoch = last_epoch + 1
            else:
                print(f"[Resume] Expected previous checkpoint at {prev_state_path} but not found; starting from epoch 1.")
        else:
            prev_state_path = prev_dir / "final.pt"
            if prev_state_path.exists():
                print(f"[Resume] Found previous weights-only checkpoint at {prev_state_path}, loading...")
                state = t.load(str(prev_state_path), map_location="cpu")
                try:
                    model.load_and_process_state_dict(state)
                except Exception:
                    model.load_state_dict(state)
                
                start_epoch = prev_epoch + 1
            else:
                print(f"[Resume] Expected previous checkpoint at {prev_state_path} but not found; starting from epoch 1.")
    else:
        print("[Resume] --prev_epoch is 0; starting fresh from epoch 1.")


    print_every = getattr(config, "print_every", 100)
    step_save_every = getattr(config, "save_every", None)

    if args.config_name.endswith("_wd"):
        max_steps = getattr(config, "max_steps", None) * (start_epoch + config.num_epochs)
    else:
        max_steps = getattr(config, "max_steps", None)

    for epoch in range(start_epoch, start_epoch + config.num_epochs):
        samples = 0
        for step, batch in enumerate(dataloader):
            tokens = batch["tokens"].to(device)

            with t.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype):
                loss = model(tokens, return_type="loss")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            max_grad_norm = getattr(config, "max_grad_norm", None)
            if max_grad_norm is not None:
                t.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            global_step += 1
            samples += tokens.shape[0]

            if config.wandb:
                wandb.log({
                    "train_loss": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "samples": samples,
                    "epoch": epoch,
                    "step": global_step,
                })

            if print_every is not None and global_step % print_every == 0:
                print(f"[Train] Epoch {epoch} Samples {samples} Step {global_step} Loss {loss.item()}")

            # Optional step-based checkpointing in the *new* save_dir
            if save_dir is not None and step_save_every is not None and global_step % step_save_every == 0:
                save_full_state(save_dir / f"state_{global_step}.pt",
                                model, optimizer, scheduler,
                                epoch=epoch, global_step=global_step)

            if max_steps is not None and global_step >= max_steps:
                print(f"[Train] Reached max_steps={max_steps}, stopping.")
                break

        # Save final_state.pt in the new save_dir at the end of each epoch
        if save_dir is not None:
            save_full_state(save_dir / "final_state.pt",
                            model, optimizer, scheduler,
                            epoch=epoch, global_step=global_step)

    return model

# ------------------- Training Loop -------------------

print("Starting Training")
model_trained = train(model, train_config, tokenized_dataset)

# ------------------- Save Final Model (weights-only) -------------------

final_model_path = os.path.join(save_path, "final.pt")
t.save(model_trained.state_dict(), final_model_path)
print(f"Model saved to {final_model_path}")
