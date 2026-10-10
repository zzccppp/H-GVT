import datetime
import torch
import os
import numpy as np
import wandb
from accelerate.utils import set_seed
from pathlib import Path

def to_cuda(x, device):

    if isinstance(x, torch.Tensor):
        return x.to(device=device)
    elif isinstance(x, tuple):
        return (to_cuda(v, device) for v in x)
    elif isinstance(x, list):
        return [to_cuda(v, device) for v in x]
    elif isinstance(x, dict):
        return {k: to_cuda(v, device) for k, v in x.items()}
    else:

        return x.to(device=device, non_blocking=True)

def save_model(
    model, epoch, optim, path, config=None, ema=None, scheduler=None, other_info=None
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "config": config,
            "ema": ema.state_dict() if ema is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "other_info": other_info if other_info is not None else {},
        },
        path,
    )

def load_model(model, optim, path, ema=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optim.load_state_dict(checkpoint["optimizer_state_dict"])
    if ema is not None and "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"])
    return model, optim, checkpoint["epoch"], ema

def generate_save_path(parent_path, model_name, epoch):
    return f"{parent_path}/{model_name}_epoch_{epoch}.pt"

def generate_run_name(model_name):
    return f"{model_name}_{datetime.datetime.now().now().strftime('%Y-%m-%d_%H-%M-%S')}"

def create_save_dir(run_name, parent_path="./runs"):
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    path = f"{parent_path}/{run_name}"
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def setup_save_dir(model_name, parent_path="./runs", run_name=None):
    if run_name is None or run_name == "":
        run_name = generate_run_name(model_name)
    return create_save_dir(run_name, parent_path)

def setseed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    set_seed(seed)

def init_wandb(config, project, tags=[], group="default", mode="online", name=None):
    import sys

    if "debugpy" in sys.modules:
        print("Debugger detected, wandb set to offline")
        run = wandb.init(
            project=project,
            config=config,
            mode="offline",
            tags=["debug"] + tags,
            name=name,
        )
    else:
        run = wandb.init(
            project=project,
            config=config,
            tags=tags,
            save_code=True,
            group=group,
            mode=mode,
            name=name,
        )
        wandb.run.log_code("src/")
    return run

def split_dataset(dataset, val_ratio=0.1, test_ratio=0.1, shuffle=True, seed=42):

    if shuffle:
        np.random.seed(seed)
        indices = np.random.permutation(len(dataset))
    else:
        indices = np.arange(len(dataset))

    test_size = int(len(dataset) * test_ratio)
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size - test_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    test_dataset = dataset[test_indices]

    return train_dataset, val_dataset, test_dataset
