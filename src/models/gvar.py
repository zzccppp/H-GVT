import math

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch_geometric.utils import to_dense_adj, to_dense_batch
from vector_quantize_pytorch import VectorQuantize

from nets.min_cut_pool import MincutPoolLayer

from .decoder import EGTDecoderRoPE
from .encoder import Encoder

def get_laplacian_spectrum(adj, mask, k=None):

    B, N, _ = adj.shape

    d = adj.sum(dim=-1)

    d_inv_sqrt = d.pow(-0.5)
    d_inv_sqrt.masked_fill_(d_inv_sqrt == float("inf"), 0)
    d_inv_sqrt.masked_fill_(~mask, 0)

    d_mat = d_inv_sqrt.unsqueeze(-1) * d_inv_sqrt.unsqueeze(1)
    norm_adj = d_mat * adj

    eye = torch.eye(N, device=adj.device).unsqueeze(0).expand(B, N, N)

    laplacian = eye - norm_adj

    laplacian = (laplacian + laplacian.transpose(1, 2)) / 2

    try:
        vals = torch.linalg.eigvalsh(laplacian)
    except Exception:
        return None

    if k is not None:
        return vals[:, :k]
    return vals

def spectral_topology_loss(adj_fine, mask_fine, adj_coarse, k):

    k = min(k, adj_coarse.size(1), adj_fine.size(1))

    spec_fine = get_laplacian_spectrum(adj_fine, mask_fine, k=k)

    if spec_fine is None:
        print("Error in calculate Spec_fine")
        return torch.tensor(0.0, device=adj_fine.device)

    mask_coarse = torch.ones(
        adj_coarse.shape[0],
        adj_coarse.shape[1],
        device=adj_coarse.device,
        dtype=torch.bool,
    )
    spec_coarse = get_laplacian_spectrum(adj_coarse, mask_coarse, k=k)
    if spec_coarse is None:
        print("Error in calculate spec_coarse")
        return torch.tensor(0.0, device=adj_fine.device)

    return F.mse_loss(spec_fine, spec_coarse)

class GVAR(torch.nn.Module):
    def __init__(self, config):
        super(GVAR, self).__init__()
        self.encoder = Encoder(**config.encoder)

        if OmegaConf.select(config, "cluster_sizes", default=None) is None:

            max_nodes = config.max_graph_nodes

            start_clusters = 2 ** math.floor(math.log2(max_nodes))
            self.cluster_sizes = []
            num_c = start_clusters
            while num_c >= 1:
                self.cluster_sizes.append(int(num_c))
                num_c /= 2
        else:
            self.cluster_sizes = list(config.cluster_sizes)

        self.pooling_layers = torch.nn.ModuleList()

        current_dim = config.encoder.output_dim
        for num_clusters in self.cluster_sizes:
            pool_layer = MincutPoolLayer(
                in_channels=current_dim,
                hidden_channels=current_dim * 2,
                num_clusters=num_clusters,
            )
            self.pooling_layers.append(pool_layer)

        self.vq = VectorQuantize(dim=config.encoder.output_dim, **config.vq)
        self.decoder = EGTDecoderRoPE(**config.decoder)

        self.lamb_node = config.get("lamb_node", 1.0)
        self.lamb_edge = config.get("lamb_edge", 1.0)
        self.lamb_commit = config.get("lamb_commit", 1.0)
        self.lamb_ent = config.get("lamb_ent", 1.0)
        self.lamb_link = config.get("lamb_link", 1.0)
        self.lamb_spec = config.get("lamb_spec", 0.1)
        self.max_graph_nodes = config.max_graph_nodes

    def forward(self, h, edge_attr, edge_index, pe, batch):
        mask = (batch.unsqueeze(-1) == batch.unsqueeze(-2)).float()
        encoded_x, encoded_edge = self.encoder(h, edge_attr, edge_index, pe, batch)
        mask_bool = to_dense_batch(
            torch.zeros(batch.size(0), device=batch.device), batch
        )[1]

        x_dense_original, mask_original = to_dense_batch(encoded_x, batch)
        adj_dense_original = to_dense_adj(edge_index, batch)

        multi_scale_embeddings = []
        total_mincut_loss = 0
        total_ortho_loss = 0

        total_spec_loss = 0

        for i, pool_layer in enumerate(self.pooling_layers):
            num_nodes = x_dense_original.size(1)
            target_clusters = self.cluster_sizes[i]

            x_pooled, adj_pooled, mincut_loss, ortho_loss = pool_layer(
                x_dense_original, adj_dense_original, mask_original
            )

            multi_scale_embeddings.append(x_pooled)

            total_mincut_loss += mincut_loss
            total_ortho_loss += ortho_loss

            k_spec = max(target_clusters // 2, 1)

            spec_loss_layer = spectral_topology_loss(
                adj_dense_original,
                mask_bool,
                adj_pooled,
                k=k_spec,
            )
            total_spec_loss += spec_loss_layer

        total_commit_loss = 0

        _, _, commit_loss = self.vq(torch.cat(multi_scale_embeddings, dim=1))
        total_commit_loss += commit_loss

        quantized, embed_ind, commit_loss = self.vq(encoded_x)
        total_commit_loss += commit_loss

        quantized_node, adj_quantized, mask_node = self.decoder(quantized, batch, mask)

        feature_rec_loss = F.cross_entropy(quantized_node, torch.argmax(h, dim=-1))

        gt_adj = torch.zeros_like(adj_quantized, device=adj_quantized.device)
        gt_adj[:, :, :, -1] = 0.9
        gt_adj[:, :, :, :-1] = to_dense_adj(
            edge_index, batch=batch, edge_attr=edge_attr
        )
        labels = torch.argmax(gt_adj, dim=-1)
        mask_edge = mask_node.unsqueeze(-1) & mask_node.unsqueeze(-2)

        diff_per_graph = (adj_quantized.argmax(dim=-1) != labels).reshape(
            gt_adj.shape[0], -1
        ).sum(dim=-1) / 2

        edge_rec_loss = F.cross_entropy(adj_quantized[mask_edge], labels[mask_edge])

        loss = (
            feature_rec_loss * self.lamb_node
            + edge_rec_loss * self.lamb_edge
            + total_commit_loss * self.lamb_commit
            + total_ortho_loss * self.lamb_ent
            + total_mincut_loss * self.lamb_link
            + total_spec_loss * self.lamb_spec
        )

        return (
            loss,
            feature_rec_loss,
            edge_rec_loss,
            total_commit_loss,
            total_ortho_loss,
            total_mincut_loss,
            total_spec_loss,
            diff_per_graph,
        )

    def encode(self, h, edge_attr, edge_index, pe, batch):
        mask = (batch.unsqueeze(-1) == batch.unsqueeze(-2)).float()
        encoded_x, encoded_edge = self.encoder(h, edge_attr, edge_index, pe, batch)

        x_dense_original, mask_original = to_dense_batch(encoded_x, batch)
        adj_dense_original = to_dense_adj(edge_index, batch)

        multi_scale_embeddings = []
        total_link_loss = 0
        total_ent_loss = 0

        for i, pool_layer in enumerate(self.pooling_layers):
            num_nodes = x_dense_original.size(1)
            target_clusters = self.cluster_sizes[i]

            x_pooled, _, link_loss, ent_loss = pool_layer(
                x_dense_original, adj_dense_original, mask_original
            )

            multi_scale_embeddings.append(x_pooled)

            total_link_loss += link_loss
            total_ent_loss += ent_loss

        multi_scale_embeddings = multi_scale_embeddings[::-1]

        _, embed_ind_multi_scale, commit_loss = self.vq(
            torch.cat(multi_scale_embeddings, dim=1)
        )
        quantized, embed_ind, commit_loss = self.vq(encoded_x)
        embed_ind_list = [embed_ind[batch == i] for i in range(batch.max().item() + 1)]

        return embed_ind_multi_scale, embed_ind_list

    def decode(self, embed_ind_list, return_logits=False):
        len_per_graph = [len(ind) for ind in embed_ind_list]
        batch = torch.cat(
            [
                torch.full(
                    (length,), i, dtype=torch.long, device=embed_ind_list[0].device
                )
                for i, length in enumerate(len_per_graph)
            ],
            dim=0,
        )
        mask = (batch.unsqueeze(-1) == batch.unsqueeze(-2)).float()

        embed_ind = torch.cat(embed_ind_list, dim=0)
        quantized = self.vq.codebook[embed_ind]

        quantized_node, adj_quantized, mask_node = self.decoder(quantized, batch, mask)

        if return_logits:
            return quantized_node, adj_quantized, mask_node, batch

        quantized_node = quantized_node.argmax(dim=-1)
        adj_quantized = adj_quantized.argmax(dim=-1)

        return quantized_node, adj_quantized, mask_node, batch
