import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv, dense_mincut_pool

class MincutPoolLayer(nn.Module):
    def __init__(self, in_channels: int, num_clusters: int, hidden_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.num_clusters = num_clusters
        self.hidden_channels = hidden_channels

        self.gnn = nn.ModuleList([
            DenseSAGEConv(in_channels, hidden_channels),
            DenseSAGEConv(hidden_channels, hidden_channels)
        ])

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, num_clusters)
        )

    def forward(self, x, adj, mask):

        num_nodes_per_graph = mask.sum(dim=1).long()
        batch_size = x.size(0)

        pool_indices = (num_nodes_per_graph > self.num_clusters).nonzero(as_tuple=True)[0]
        passthrough_indices = (num_nodes_per_graph <= self.num_clusters).nonzero(as_tuple=True)[0]

        x_final = torch.zeros(batch_size, self.num_clusters, self.in_channels, device=x.device)
        adj_final = torch.zeros(batch_size, self.num_clusters, self.num_clusters, device=x.device)
        total_mincut_loss = torch.tensor(0.0, device=x.device)
        total_ortho_loss = torch.tensor(0.0, device=x.device)

        if len(pool_indices) > 0:
            x_pool = x[pool_indices]
            adj_pool = adj[pool_indices]
            mask_pool = mask[pool_indices]

            h = x_pool
            for gnn_layer in self.gnn:
                h = gnn_layer(h, adj_pool, mask_pool).relu()
            s = self.mlp(h)

            x_pooled, adj_pooled, mincut_loss, ortho_loss = dense_mincut_pool(
                x_pool, adj_pool, s, mask_pool
            )

            x_final[pool_indices] = x_pooled
            adj_final[pool_indices] = adj_pooled
            total_mincut_loss += mincut_loss
            total_ortho_loss += ortho_loss

        if len(passthrough_indices) > 0:
            x_pass = x[passthrough_indices]
            adj_pass = adj[passthrough_indices]

            for i, idx in enumerate(passthrough_indices):
                num_nodes = num_nodes_per_graph[idx]
                x_final[idx, :num_nodes] = x_pass[i, :num_nodes]
                adj_final[idx, :num_nodes, :num_nodes] = adj_pass[i, :num_nodes, :num_nodes]

        return x_final, adj_final, total_mincut_loss, total_ortho_loss