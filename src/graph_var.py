import os
from logging import getLogger

import hydra
import torch
from accelerate import Accelerator
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader
from tqdm import tqdm, trange

import wandb
from load_dataset import get_dataset
from models.gvar import GVAR
from utils.commons import init_wandb, save_model, setseed

log = getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="main")
def main(cfg: DictConfig) -> None:
    accelerator = Accelerator(
        **OmegaConf.to_container(cfg.accelerate_args, resolve=True)
    )
    setseed(cfg.seed)
    if accelerator.is_main_process:
        log.info(f"Seed set to {cfg.seed}")
    output_dir = HydraConfig.get().runtime.output_dir
    if accelerator.is_main_process:
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

    data = get_dataset(
        name=cfg.dataset.name,
        split="train",
        pe_dim=cfg.dataset.pe_dim,
    )
    test_data = get_dataset(
        name=cfg.dataset.name,
        split="test",
        pe_dim=cfg.dataset.pe_dim,
    )

    loader = train_loader = DataLoader(
        data,
        batch_size=cfg.train.bs,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=cfg.train.bs,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    model = GVAR(cfg.gvar)

    optim = instantiate(cfg.train.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.train.scheduler, optimizer=optim)

    model, optim, scheduler, train_loader, test_loader = accelerator.prepare(
        model, optim, scheduler, train_loader, test_loader
    )

    for epoch in trange(
        1, cfg.train.epochs + 1, disable=not accelerator.is_local_main_process
    ):
        total_loss = 0.0
        total_feature_rec_loss = 0.0
        total_edge_rec_loss = 0.0
        total_commit_loss = 0.0
        total_mincut_loss = 0.0
        total_ortho_loss = 0.0
        total_spec_loss = 0.0
        total_diff_per_graph = 0.0
        num_batches = 0

        model.train()

        pbar = tqdm(
            train_loader,
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch}",
        )

        for batch_graph in pbar:
            batch_graph = batch_graph
            (
                loss,
                feature_rec_loss,
                edge_rec_loss,
                commit_loss,
                mincut_loss,
                ortho_loss,
                spec_loss,
                diff_per_graph,
            ) = model(
                batch_graph.x,
                batch_graph.edge_attr,
                batch_graph.edge_index,
                batch_graph.pe,
                batch_graph.batch,
            )

            total_loss += loss.item()
            total_feature_rec_loss += feature_rec_loss.item()
            total_edge_rec_loss += edge_rec_loss.item()
            total_commit_loss += commit_loss.item()
            total_mincut_loss += mincut_loss.item()
            total_ortho_loss += ortho_loss.item()
            total_spec_loss += spec_loss.item()
            total_diff_per_graph += diff_per_graph.mean().item()
            num_batches += 1

            optim.zero_grad()
            accelerator.backward(loss)
            optim.step()

        if accelerator.is_main_process:
            run.log(
                {
                    "loss": total_loss / num_batches,
                    "feature_rec_loss": total_feature_rec_loss / num_batches,
                    "edge_rec_loss": total_edge_rec_loss / num_batches,
                    "total_commit_loss": total_commit_loss / num_batches,
                    "total_mincut_loss": total_mincut_loss / num_batches,
                    "total_ortho_loss": total_ortho_loss / num_batches,
                    "total_spec_loss": total_spec_loss / num_batches,
                    "diff_per_graph_mean": total_diff_per_graph / num_batches,
                    "epoch": epoch,
                    "lr": scheduler.get_last_lr()[0],
                }
            )

        scheduler.step()

        if epoch % cfg.train.ckpt_freq == 0:
            if accelerator.is_main_process:
                ckpt_path = f"{HydraConfig.get().runtime.output_dir}/checkpoints/gvar_epoch_{epoch}.pt"
                unwrapped_model = accelerator.unwrap_model(model)
                save_model(
                    unwrapped_model,
                    epoch,
                    optim,
                    path=ckpt_path,
                    config=cfg,
                )
            ...

        if epoch % cfg.train.test_freq == 0:

            ...

    if accelerator.is_main_process:
        ckpt_path = f"{HydraConfig.get().runtime.output_dir}/checkpoints/gvar_final.pt"
        unwrapped_model = accelerator.unwrap_model(model)
        save_model(
            unwrapped_model,
            epoch,
            optim,
            path=ckpt_path,
            config=cfg,
        )

        artifact = wandb.Artifact(
            name=f"model-{job_name}-{os.path.basename(output_dir)}",
            type="model",
            description="Final model",
            metadata={"config": cfg, "job_name": job_name},
        )
        artifact.add_file(ckpt_path)
        run.log_artifact(artifact)

if __name__ == "__main__":
    main()
