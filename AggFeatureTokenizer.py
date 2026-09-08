import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class AggFeatureTokenizer(nn.Module):
    def __init__(self, num_primitives, max_hop, num_raw_cols, embedding_dim):
        super().__init__()
        self.primitive_emb = nn.Embedding(num_primitives, embedding_dim)  # mean/max/min/skew/count/...
        self.hop_emb = nn.Embedding(max_hop + 1, embedding_dim)            # reuse same hop vocab as neighbor tokens
        self.col_emb = nn.Embedding(num_raw_cols, embedding_dim)           # which raw column was aggregated
        self.value_proj = nn.Linear(1, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    

    def forward(self, values, primitive_ids, hop_ids, col_ids):
        # values: [B, F] (F = num aggregate features, fixed layout)
        # primitive_ids/hop_ids/col_ids: [F], static — precomputed once, not per-sample
        v = self.value_proj(values.unsqueeze(-1))          # [B, F, d]
        tok = v + self.primitive_emb(primitive_ids)         # broadcasts [F,d] over batch
        tok = tok + self.hop_emb(hop_ids)
        tok = tok + self.col_emb(col_ids)
        return self.norm(tok)                                # [B, F, d]