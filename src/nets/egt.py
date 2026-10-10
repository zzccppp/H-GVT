import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, to_dense_adj

class EGTLayer(nn.Module):

    def __init__(
        self,
        feat_size,
        edge_feat_size,
        num_heads,
        num_virtual_nodes,
        dropout=0,
        attn_dropout=0,
        activation=nn.ELU(),
        edge_update=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_virtual_nodes = num_virtual_nodes
        self.edge_update = edge_update

        assert feat_size % num_heads == 0, "feat_size must be divisible by num_heads"
        self.dot_dim = feat_size // num_heads
        self.mha_ln_h = nn.LayerNorm(feat_size)
        self.mha_ln_e = nn.LayerNorm(edge_feat_size)
        self.edge_input = nn.Linear(edge_feat_size, num_heads)
        self.qkv_proj = nn.Linear(feat_size, feat_size * 3)
        self.gate = nn.Linear(edge_feat_size, num_heads)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.node_output = nn.Linear(feat_size, feat_size)
        self.mha_dropout_h = nn.Dropout(dropout)

        self.node_ffn = nn.Sequential(
            nn.LayerNorm(feat_size),
            nn.Linear(feat_size, feat_size),
            activation,
            nn.Linear(feat_size, feat_size),
            nn.Dropout(dropout),
        )

        if self.edge_update:
            self.edge_output = nn.Linear(num_heads, edge_feat_size)
            self.mha_dropout_e = nn.Dropout(dropout)
            self.edge_ffn = nn.Sequential(
                nn.LayerNorm(edge_feat_size),
                nn.Linear(edge_feat_size, edge_feat_size),
                activation,
                nn.Linear(edge_feat_size, edge_feat_size),
                nn.Dropout(dropout),
            )

    def forward(self, nfeat, efeat, mask=None):

        nfeat_r1 = nfeat
        efeat_r1 = efeat

        nfeat_ln = self.mha_ln_h(nfeat)
        efeat_ln = self.mha_ln_e(efeat)
        qkv = self.qkv_proj(nfeat_ln)
        e_bias = self.edge_input(efeat_ln)
        gates = self.gate(efeat_ln)
        bsz, N, _ = qkv.shape
        q_h, k_h, v_h = qkv.view(bsz, N, -1, self.num_heads).split(self.dot_dim, dim=2)
        attn_hat = torch.einsum("bldh,bmdh->blmh", q_h, k_h)
        attn_hat = attn_hat.clamp(-5, 5) + e_bias

        if mask is None:
            gates = torch.sigmoid(gates)
            attn_tild = F.softmax(attn_hat, dim=2) * gates
        else:
            gates = torch.sigmoid(gates + mask.unsqueeze(-1))
            attn_tild = F.softmax(attn_hat + mask.unsqueeze(-1), dim=2) * gates

        attn_tild = self.attn_dropout(attn_tild)
        v_attn = torch.einsum("blmh,bmkh->blkh", attn_tild, v_h)

        degrees = torch.sum(gates, dim=2, keepdim=True)
        degree_scalers = torch.log(1 + degrees)
        degree_scalers[:, : self.num_virtual_nodes] = 1.0
        v_attn = v_attn * degree_scalers

        v_attn = v_attn.reshape(bsz, N, self.num_heads * self.dot_dim)
        nfeat = self.node_output(v_attn)

        nfeat = self.mha_dropout_h(nfeat)
        nfeat.add_(nfeat_r1)
        nfeat_r2 = nfeat
        nfeat = self.node_ffn(nfeat)
        nfeat.add_(nfeat_r2)

        if self.edge_update:
            efeat = self.edge_output(attn_hat)
            efeat = self.mha_dropout_e(efeat)
            efeat.add_(efeat_r1)
            efeat_r2 = efeat
            efeat = self.edge_ffn(efeat)
            efeat.add_(efeat_r2)

            return nfeat, efeat

        return nfeat

class EGTNetwork(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        edge_dim,
        num_layers,
        dropout,
        beta=True,
        heads=1,
        edge_output_dim=None,
    ):
        super(EGTNetwork, self).__init__()

        self.num_layers = num_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim
        self.heads = heads
        if edge_output_dim is None:
            self.edge_output_dim = edge_dim
            edge_output_dim = edge_dim
        else:
            self.edge_output_dim = edge_output_dim

        self.node_input_proj = torch.nn.Linear(input_dim, hidden_dim)
        self.edge_input_proj = torch.nn.Linear(edge_dim, hidden_dim)

        conv_layers = []
        for _ in range(num_layers):
            conv_layers.append(
                EGTLayer(
                    feat_size=hidden_dim,
                    edge_feat_size=hidden_dim,
                    num_heads=heads,
                    num_virtual_nodes=0,
                    dropout=dropout,
                    edge_update=True,
                )
            )
        self.convs = torch.nn.ModuleList(conv_layers)

        self.norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(hidden_dim) for _ in range(num_layers - 1)]
        )

        self.node_output_proj = torch.nn.Linear(hidden_dim, output_dim)
        self.edge_output_proj = torch.nn.Linear(hidden_dim, edge_output_dim)

        self.dropout = dropout

    def reset_parameters(self):
        self.node_input_proj.reset_parameters()
        self.edge_input_proj.reset_parameters()

        for conv in self.convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()

        for norm in self.norms:
            norm.reset_parameters()

        self.node_output_proj.reset_parameters()
        self.edge_output_proj.reset_parameters()

    def prepare_inputs(self, x, edge_index, edge_attr, batch):
        x_dense, mask_nodes = to_dense_batch(x, batch)

        if edge_attr is not None:
            edge_dense = to_dense_adj(edge_index, batch, edge_attr)
        else:
            edge_dense = to_dense_adj(edge_index, batch)
            edge_dense = edge_dense.unsqueeze(-1)

        mask_2d = mask_nodes.unsqueeze(-1) & mask_nodes.unsqueeze(-2)
        attention_mask = torch.where(
            mask_2d,
            torch.zeros_like(mask_2d, dtype=torch.float),
            torch.full_like(mask_2d, -1e9, dtype=torch.float),
        )

        return x_dense, edge_dense, attention_mask, mask_nodes, mask_2d

    def forward(self, x, edge_index, batch=None, edge_attr=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.node_input_proj(x)
        if edge_attr is not None:
            edge_attr = self.edge_input_proj(edge_attr)

        x, edge_attr, attn_mask, mask_nodes, mask_adj = self.prepare_inputs(
            x, edge_index, edge_attr, batch
        )

        for i in range(self.num_layers - 1):
            x_new, edge_attr_new = self.convs[i](x, edge_attr, attn_mask)
            x = self.norms[i](x_new)
            x = torch.relu(x)
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            edge_attr = edge_attr_new

        x, edge_attr = self.convs[-1](x, edge_attr, attn_mask)

        x = self.node_output_proj(x)
        edge_attr = self.edge_output_proj(edge_attr)

        return x, edge_attr, mask_nodes
