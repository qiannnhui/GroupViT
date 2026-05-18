import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

# Ensure we can import from the parent GroupViT repo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.group_vit import GroupViT

# Ensure we can import from GraphCL
sys.path.append('/home/qiannnhui/GraphCL/unsupervised_TU')
from gin import Encoder 

class NodeGroupViT(nn.Module):
    def __init__(self, in_channels, embed_dim=128, num_classes=0):
        super().__init__()
        # Initialize the actual GroupViT from the repository
        self.groupvit = GroupViT(
            embed_dim=embed_dim,
            embed_factors=[1, 1, 1],
            depths=[2, 2, 2],
            num_heads=[4, 4, 4],
            num_group_tokens=[16, 4, 0],
            num_output_groups=[16, 4],
            num_classes=num_classes
        )
        
        # Project our combined features into the GroupViT embed_dim
        self.proj = nn.Linear(in_channels, embed_dim, bias=False)
        
    def forward(self, x_combined, batch, max_nodes=1024):
        from torch_geometric.utils import to_dense_batch
        
        # x_dense: [B, max_nodes_in_batch, in_channels], mask: [B, max_nodes_in_batch]
        x_dense, mask = to_dense_batch(x_combined, batch)
        
        # Truncate each graph to at most max_nodes to prevent OOM
        if x_dense.size(1) > max_nodes:
            x_dense = x_dense[:, :max_nodes, :]
            mask = mask[:, :max_nodes]
        
        # Project to embed_dim
        x_emb = self.proj(x_dense)
        
        # Zero out padding tokens
        x_emb = x_emb * mask.unsqueeze(-1)
        
        # Pass through GroupViT layers
        group_token = None
        attn_list = []
        for layer in self.groupvit.layers:
            x_emb, group_token, attn_dict = layer(x_emb, group_token, return_attn=True)
            if attn_dict is not None:
                attn_list.append(attn_dict)
            
        components = self.groupvit.norm(x_emb)
        
        # forward_image_head applies avgpool over grouped tokens
        out = self.groupvit.forward_image_head(components)
        
        return out, components, attn_list, x_dense

class DualEncoder(nn.Module):
    def __init__(self, dataset_num_features, hidden_dim=128, num_gc_layers=3, rw_dim=16):
        super().__init__()
        # GCL branch
        self.graph_encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)
        
        # GroupViT branch (Now taking node features + RW features)
        combined_dim = dataset_num_features + rw_dim
        self.node_transformer = NodeGroupViT(in_channels=combined_dim, embed_dim=hidden_dim)
        
        # Projector for GCL branch
        self.graph_proj = nn.Linear(hidden_dim * num_gc_layers, hidden_dim)

    def forward(self, data, max_nodes=1024):
        x, edge_index, batch, rw_x = data.x, data.edge_index, data.batch, data.rw_x
        
        # 1. GCL branch
        z_graph, _ = self.graph_encoder(x, edge_index, batch)
        z_graph = self.graph_proj(z_graph)
        
        # 2. GroupViT branch
        # [FIXED] Concatenate node features (x) and random walk features (rw_x)
        # This allows the model to "see" both atom types and graph structure
        x_combined = torch.cat([x, rw_x], dim=-1)
        z_nodes, components, attn_list, rw_dense = self.node_transformer(x_combined, batch, max_nodes=max_nodes)
        
        return z_nodes, z_graph, components, attn_list, rw_dense
