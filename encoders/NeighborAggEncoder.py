from typing import Dict


import torch
import torch.nn as nn
import torch.nn.functional as F


class NeighborAggEncoder(nn.Module):
    """
    Encoder for neighbor aggregate-statistic vectors (precomputed featuretools features).

    Mirrors NeighborTfsEncoder's grouped-by-type, scatter-back pattern, but additionally
    tokenizes each scalar statistic by (primitive, hop, column) identity before pooling,
    since a raw Linear over the vector can't distinguish "max" from "mean" positionally.
    """
    def __init__(
        self,
        channels: int,
        node_type_map,
        agg_dim_dict: Dict[str, int],      # {node_type: F_type}
        feature_meta: Dict[str, Dict[str, torch.Tensor]],  # {node_type: {"primitive_ids", "hop_ids", "col_ids"}}
    ):
        super(NeighborAggEncoder, self).__init__()

        self.node_type_map = node_type_map
        self.inv_node_type_map = {idx: nt for nt, idx in node_type_map.items()}
        self.channels = channels
        self.agg_dim_dict = agg_dim_dict

        num_primitives = max(m["primitive_ids"].max().item() for m in feature_meta.values()) + 1
        num_hops = max(m["hop_ids"].max().item() for m in feature_meta.values()) + 1
        num_cols = max(m["col_ids"].max().item() for m in feature_meta.values()) + 1

        self.primitive_emb = nn.Embedding(num_primitives, channels)
        self.hop_emb = nn.Embedding(num_hops, channels)
        self.col_emb = nn.Embedding(num_cols, channels)

        self.value_w = nn.ParameterDict()
        self.value_b = nn.ParameterDict()
        for node_type, F_type in agg_dim_dict.items():
            self.value_w[node_type] = nn.Parameter(torch.randn(F_type, channels) * 0.02)
            self.value_b[node_type] = nn.Parameter(torch.zeros(F_type, channels))
            self.register_buffer(f"primitive_ids_{node_type}", feature_meta[node_type]["primitive_ids"])
            self.register_buffer(f"hop_ids_{node_type}", feature_meta[node_type]["hop_ids"])
            self.register_buffer(f"col_ids_{node_type}", feature_meta[node_type]["col_ids"])

        self.pool_norm = nn.LayerNorm(channels)

    def reset_parameters(self):
        self.primitive_emb.reset_parameters()
        self.hop_emb.reset_parameters()
        self.col_emb.reset_parameters()
        for w in self.value_w.values():
            nn.init.normal_(w, std=0.02)
        for b in self.value_b.values():
            nn.init.zeros_(b)

    def forward(self, batch_dict, neighbor_types):
        """
        Args:
            batch_dict (dict): A dictionary containing:
              - grouped_agg[t_int]: [N_t, F_type] tensor of raw aggregate values for
                                     all neighbors of type t_int in the batch.
              - grouped_indices[t_int]: flat positions for each row in grouped_agg[t_int]
                                         (same list already built for grouped_tfs).
              - flat_batch_idx, flat_nbr_idx: as in NeighborTfsEncoder.
            neighbor_types (Tensor): [B, K] node type indices.

        Returns:
            Tensor: [B, K, channels]
        """
        grouped_agg = batch_dict["grouped_agg"]
        grouped_indices = batch_dict["grouped_indices"]
        flat_batch_idx = batch_dict["flat_batch_idx"]
        flat_nbr_idx = batch_dict["flat_nbr_idx"]

        B, K = neighbor_types.shape
        N = len(flat_batch_idx)
        device = neighbor_types.device

        encoded_flat_tensor = torch.zeros((N, self.channels), device=device)

        for t_int, mat in grouped_agg.items():
            node_type_str = self.inv_node_type_map[t_int]
            mat = mat.to(device=device)
            mat = torch.nan_to_num(mat, nan=0.0, posinf=1e6, neginf=-1e6)  # [N_t, F_type]

            v = mat.unsqueeze(-1) * self.value_w[node_type_str] + self.value_b[node_type_str]  # [N_t, F_type, C]
            tok = (
                v
                + self.primitive_emb(getattr(self, f"primitive_ids_{node_type_str}"))
                + self.hop_emb(getattr(self, f"hop_ids_{node_type_str}"))
                + self.col_emb(getattr(self, f"col_ids_{node_type_str}")) #Be careful here think about it
                # to do +self.path_emb  beacuse we want to discriminate the paths
            )
            out_t = self.pool_norm(tok.sum(dim=1))  # [N_t, channels]

            idx_list = grouped_indices[t_int]
            idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
            encoded_flat_tensor[idx_tensor] = out_t

        output = torch.zeros((B, K, self.channels), device=device)
        indices_i = torch.tensor(flat_batch_idx, dtype=torch.long, device=device)
        indices_j = torch.tensor(flat_nbr_idx, dtype=torch.long, device=device)
        output[indices_i, indices_j] = encoded_flat_tensor

        return output