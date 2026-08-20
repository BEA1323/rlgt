from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

class GNNGATEEncoder(nn.Module):
    """
    A GAT-based positional encoder that:
      1) Uses Laplacian PE when pe_dim > 0,
         otherwise assigns each node a random scalar.
      2) Projects the input to layer_embedding_dim.
      3) Runs a multi-layer GAT.
      4) Uses multi-head attention to aggregate neighbor information.
      5) Uses residual connections after each GAT layer.
      6) Aggregates intermediate layer outputs using:
         - "none"
         - "cat"
         - "mean"
         - "max"
      7) Projects the result to embedding_dim.
      8) Returns [B, K, embedding_dim].
    """

    def __init__(self, embedding_dim: int, num_layers: int = 4, pooling: str = "none", pe_dim: int = 0, num_heads: int = 4):
        super().__init__()

        self.pooling = pooling.lower()
        self.num_layers = num_layers
        self.pe_dim = pe_dim
        self.num_heads = num_heads

        # Dimension used inside every GAT layer
        self.layer_embedding_dim = embedding_dim // num_layers

        if self.pe_dim > 0:
            # Laplacian PE: [N, pe_dim]
            self.input_proj = nn.Linear(self.pe_dim,self.layer_embedding_dim)
        else:
            # Random scalar: [N, 1]
            self.input_proj = nn.Linear(1,self.layer_embedding_dim)

        # --------------------------------------------------
        # GAT layers
        # --------------------------------------------------

        self.conv = nn.ModuleList()

        for _ in range(num_layers):
            self.conv.append(
                GATConv(in_channels=self.layer_embedding_dim,out_channels=self.layer_embedding_dim // num_heads,heads=num_heads,concat=True,)
            )

        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            self.bns.append(nn.BatchNorm1d(self.layer_embedding_dim))

        # --------------------------------------------------
        # Dimension after pooling
        # --------------------------------------------------
        if self.pooling == "cat":
            final_input_dim = (self.layer_embedding_dim * num_layers)

        elif self.pooling in ["none", "mean", "max"]:
            final_input_dim = self.layer_embedding_dim

        else:
            raise ValueError(
                "Invalid pooling method. "
                "Choose from 'none', 'cat', 'mean', 'max'."
            )

        # --------------------------------------------------
        # Final projection
        # --------------------------------------------------

        self.final_transform = nn.Linear(final_input_dim,embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)

        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
        for conv in self.conv:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        nn.init.xavier_uniform_(self.final_transform.weight)

        if self.final_transform.bias is not None:
            nn.init.zeros_(self.final_transform.bias)

    def forward(self, edge_index, batch):
        device = edge_index.device
        total_nodes = batch.size(0)
        if self.pe_dim > 0:

            data = Data(edge_index=edge_index,num_nodes=total_nodes)

            transform = T.AddLaplacianEigenvectorPE(k=self.pe_dim)

            data = transform(data)
            x_input = data.laplacian_eigenvector_pe.to(device)
        else:
            x_input = torch.randn(total_nodes,1,device=device)

        x = self.input_proj(x_input)
        outputs = []
        for i, conv in enumerate(self.conv):
            x_res = x
            x_new = conv(x,edge_index)
            x_new = self.bns[i](x_new)
            x_new = F.relu(x_new)
            x = x_new + x_res
            # Store intermediate representation
            outputs.append(x)

        # --------------------------------------------------
        # Aggregate intermediate outputs
        # --------------------------------------------------

        if self.pooling == "none":
            x_final = outputs[-1]
        elif self.pooling == "cat":
            x_final = torch.cat(outputs,dim=-1)

        elif self.pooling == "mean":
            outputs_tensor = torch.stack(outputs,dim=-1)
            x_final = torch.mean(outputs_tensor,dim=-1)
        elif self.pooling == "max":
            outputs_tensor = torch.stack(outputs,dim=-1)
            x_final = torch.max(outputs_tensor,dim=-1)[0]

        x = self.final_transform(x_final)

        B = batch.max().item() + 1
        K = total_nodes // B
        out = x.view(B,K,-1)

        return out