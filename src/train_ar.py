import os
from functools import partial
from logging import getLogger

import hydra
import torch
import torch.multiprocessing as mp
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import wandb
from models.gpt2_modify import GPT_Normal
from models.gvar import GVAR
from mydatasets.LatentH5Dataset import LatentH5Dataset
from utils.commons import init_wandb, save_model, setseed
from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs

log = getLogger(__name__)

def get_latnets_dataset(dataset_cfg: DictConfig) -> LatentH5Dataset:
    data = LatentH5Dataset(dataset_cfg.save_path)
    return data

@hydra.main(version_base=None, config_path="config", config_name="ar")
def main(cfg: DictConfig) -> None:
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        **OmegaConf.to_container(cfg.accelerate_args, resolve=True)
    )
    set_seed(cfg.seed)

    log.info(f"Seed set to {cfg.seed}")
    output_dir = HydraConfig.get().runtime.output_dir
    log.info(f"Output directory: {output_dir}")
    job_name = HydraConfig.get().job.name

    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if accelerator.is_main_process:
        run = init_wandb(
            config,
            cfg.wandb.project,
            tags=cfg.wandb.tags,
            group=cfg.wandb.group,
            mode=cfg.wandb.mode,
            name=f"{job_name}-{os.path.basename(output_dir)}",
        )

    device = accelerator.device
    log.info(f"Process {accelerator.process_index} using device: {device}")

    data = get_latnets_dataset(cfg.dataset)

    cfg.model.gpt.vocab_size = (
        data.vae_config.gvar.vq.codebook_size + 2
    )
    if cfg.model.gpt.max_var_len == -1:
        cfg.model.gpt.max_var_len = data.vae_config.gvar.max_graph_nodes + 1
    cfg.model.gpt.pad_token_id = data.vae_config.gvar.vq.codebook_size + 1
    cfg.model.gpt.eos_token_id = data.vae_config.gvar.vq.codebook_size
    cfg.model.gpt.patch_nums = GVAR(data.vae_config.gvar).cluster_sizes[::-1]

    model = GPT_Normal(cfg.model.gpt)
    model_level_lengths = model.level_lengths

    collate_fn = partial(
        LatentH5Dataset.collate_fn_normal_gpt,
        padding_value=model.pad_token_id,
        eos_value=model.eos_token_id,
        sos_value=0,
    )
    loader = DataLoader(
        data,
        batch_size=cfg.train.bs,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_fn,
        persistent_workers=True,
        pin_memory=True,
    )

    optim = instantiate(cfg.train.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.train.scheduler, optimizer=optim)

    model, optim, scheduler, loader = accelerator.prepare(
        model, optim, scheduler, loader
    )

    epoch_pbar = trange(
        1, cfg.train.epochs + 1, disable=not accelerator.is_local_main_process
    )

    for epoch in epoch_pbar:
        total_loss = 0.0
        num_batches = 0
        model.train()

        level_indices = (
            torch.repeat_interleave(
                torch.arange(len(model_level_lengths)),
                torch.tensor(model_level_lengths),
            )
            .unsqueeze(0)
            .to(device)
        )
        level_indices = torch.cat(
            [torch.tensor([[0]], device=device), level_indices], dim=1
        )

        batch_pbar = tqdm(loader, disable=not accelerator.is_local_main_process)

        for batch in batch_pbar:
            src_tokens = batch["src"].to(device)
            tgt_tokens = batch["tgt"].to(device)
            B = src_tokens.size(0)
            lvl_ids = level_indices.expand(B, -1)[:, : src_tokens.size(1)]

            logits, loss = model(src_tokens, lvl_ids, targets=tgt_tokens)

            total_loss += loss.item()
            num_batches += 1

            optim.zero_grad()
            accelerator.backward(loss)
            optim.step()

        avg_loss = total_loss / num_batches
        if accelerator.is_main_process:
            log.info(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")
            wandb.log(
                {
                    "train/avg_loss": avg_loss,
                    "lr": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                },
                step=epoch,
            )

        scheduler.step()

        if epoch % cfg.train.ckpt_freq == 0:
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                ckpt_path = (
                    f"{HydraConfig.get().runtime.output_dir}/checkpoints/gar_{epoch}.pt"
                )
                unwrapped_model = accelerator.unwrap_model(model)
                save_model(
                    unwrapped_model,
                    epoch,
                    optim,
                    path=ckpt_path,
                    config=cfg,
                )

if __name__ == "__main__":
    main()
