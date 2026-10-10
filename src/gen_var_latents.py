import torch
import hydra
from omegaconf import DictConfig
from torch_geometric.utils import to_dense_adj
from torch_scatter import scatter_sum
from models.gvar import GVAR
from load_dataset import get_dataset
from torch_geometric.loader import DataLoader
import h5py
import numpy as np
from tqdm import tqdm

@hydra.main(version_base=None, config_path="config", config_name="gen_latents")
def main(cfg: DictConfig) -> None:
    device = torch.device(cfg.device)
    ckpt = torch.load(cfg.var_ckpt_path, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt["config"]

    model = GVAR(ckpt_cfg.gvar).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    data = get_dataset(
        name=ckpt_cfg.dataset.name,
        split="train",
        pe_dim=ckpt_cfg.dataset.pe_dim,
    )

    loader = DataLoader(
        data,
        batch_size=cfg.bs,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    save_path = cfg.save_path
    print(f"Latents will be saved to: {save_path}")

    with h5py.File(save_path, "w") as f:
        ckpt_group = f.create_group("var_checkpoint")
        with open(cfg.var_ckpt_path, "rb") as ckpt_file:
            ckpt_binary_data = ckpt_file.read()

        ckpt_group.create_dataset(
            "data",
            data=np.frombuffer(ckpt_binary_data, dtype="u1"),
        )

        multi_scale_dim = sum(model.cluster_sizes)
        dset_multi_scale = f.create_dataset(
            "embed_ind_multi_scale",
            shape=(0, multi_scale_dim),
            maxshape=(None, multi_scale_dim),
            dtype=np.int64,
            chunks=(128, multi_scale_dim),
        )

        dset_num_nodes = f.create_dataset(
            "num_nodes", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=(1024,)
        )

        vlen_dtype = h5py.special_dtype(vlen=np.int64)
        dset_embed_ind = f.create_dataset(
            "embed_ind", shape=(0,), maxshape=(None,), dtype=vlen_dtype, chunks=(128,)
        )

        current_size = 0
        all_dropped = 0
        for batch_graph in tqdm(loader, desc="Generating and Saving Latents"):
            batch_graph = batch_graph.to(device)
            with torch.no_grad():
                embed_ind_multi_scale, embed_ind = model.encode(
                    batch_graph.x,
                    batch_graph.edge_attr,
                    batch_graph.edge_index,
                    batch_graph.pe,
                    batch_graph.batch,
                )

                quantized_node, adj_quantized, mask_node, _ = model.decode(
                    embed_ind, return_logits=True
                )
                x_diff = scatter_sum(
                    (
                        quantized_node.argmax(dim=-1) != batch_graph.x.argmax(dim=-1)
                    ).int(),
                    batch_graph.batch,
                    dim=0,
                )
                gt_adj = torch.zeros_like(adj_quantized, device=adj_quantized.device)
                gt_adj[:, :, :, -1] = 0.9
                gt_adj[:, :, :, :-1] = to_dense_adj(
                    batch_graph.edge_index,
                    batch=batch_graph.batch,
                    edge_attr=batch_graph.edge_attr,
                )
                gt_adj = gt_adj.argmax(dim=-1)
                adj_quantized = adj_quantized.argmax(dim=-1)

                diff_per_graph = (adj_quantized != gt_adj).reshape(
                    gt_adj.shape[0], -1
                ).sum(dim=-1) / 2
                recon_flags = torch.logical_and((x_diff == 0), (diff_per_graph == 0))

                all_dropped += (~recon_flags).sum().item()

            num_nodes_per_graph = torch.bincount(batch_graph.batch)[
                recon_flags
            ]

            batch_size = num_nodes_per_graph.size(0)
            new_size = current_size + batch_size

            dset_multi_scale.resize((new_size, multi_scale_dim))
            dset_num_nodes.resize((new_size,))
            dset_embed_ind.resize((new_size,))

            dset_multi_scale[current_size:new_size] = (
                embed_ind_multi_scale[recon_flags, :].cpu().numpy()
            )
            dset_num_nodes[current_size:new_size] = num_nodes_per_graph.cpu().numpy()

            embed_ind_np = [
                embed_ind[i].cpu().numpy()
                for i in range(len(embed_ind))
                if recon_flags[i]
            ]
            dset_embed_ind[current_size:new_size] = embed_ind_np

            current_size = new_size

    print(f"Successfully saved {current_size} graph latents to {save_path}")
    print(f"Total dropped graphs during saving: {all_dropped}")
    print("Drop rate: {:.4f}".format(all_dropped / len(data)))

if __name__ == "__main__":
    main()
