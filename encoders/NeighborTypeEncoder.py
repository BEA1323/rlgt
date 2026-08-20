from typing import Dict


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_frame

import torch_geometric.transforms as T
from torch_geometric.data import Data

class NeighborTypeEncoder(nn.Module):
    '''
    Encoder for neighbor types. (Will remain same as relgt)
    Uses an embedding layer to convert integer type indices into dense vectors.
    '''
    def __init__(self, node_type_map, embedding_dim):
        '''
        Args:
            node_type_map (dict): A mapping from node type strings to integer indices.
            embedding_dim (int): Dimension of the embedding vectors.(size will be reduced)
        '''
        super(NeighborTypeEncoder, self).__init__()
        num_types = max(node_type_map.values()) + 1
        self.embedding=nn.Embedding(num_embeddings=num_types+1,embedding_dim=embedding_dim)


    def reset_parameters(self):
        self.embedding.reset_parameters()
    def forward(self,type_indices):
        """
        Args:
            type_indices (Tensor): Tensor of shape (...), containing integer indices for neighbor types.
        
        Returns:
            Tensor: Embedded representations of shape (..., embedding_dim).
        """
        return self.embedding(type_indices)