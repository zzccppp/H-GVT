import torch
import torch.nn as nn
from nets.gt_pyg import GraphTransformerNet
from torch_scatter import scatter_mean

class Encoder(nn.Module):
    def __init__(
        self,
        node_feat_dim,
        hidden_dim,
        output_dim,
        edge_dim,
        num_layers,
        pe_dim,
    ):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers

        self.net = GraphTransformerNet(
            node_dim_in=node_feat_dim,
            edge_dim_in=edge_dim,
            pe_in_dim=pe_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            norm="bn",
            num_gt_layers=num_layers,
            num_heads=8,
            act="relu",
            dropout=0.1,
        )

        self.fusion_layer = nn.Linear(output_dim + output_dim, output_dim)

    def forward(self, x, edge_attr, edge_index, pe, batch):
        encoded_x, encoded_edge_attr = self.net(
            x, edge_index, batch=batch, edge_attr=edge_attr, pe=pe
        )
        aggregated_edge_info = scatter_mean(
            encoded_edge_attr, edge_index[1], dim=0, dim_size=x.size(0)
        )
        fused_node_features = torch.cat([encoded_x, aggregated_edge_info], dim=-1)
        final_node_features_for_vq = self.fusion_layer(fused_node_features)
        return final_node_features_for_vq, encoded_edge_attr
