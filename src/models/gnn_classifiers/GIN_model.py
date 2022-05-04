import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import BatchNorm, GINConv, Linear


class GIN(torch.nn.Module):
    """GIN"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, num_layers):
        super(GIN, self).__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        self.convs.append(GINConv(
            nn.Sequential(Linear(in_channels, hidden_channels),
                       BatchNorm(hidden_channels), nn.ReLU(), nn.Dropout(p=dropout),
                       Linear(hidden_channels, hidden_channels), nn.ReLU())))
        for _ in range(num_layers-2):
            self.convs.append(GINConv(
            nn.Sequential(Linear(hidden_channels, hidden_channels),
                       BatchNorm(hidden_channels), nn.ReLU(), nn.Dropout(p=dropout),
                       Linear(hidden_channels, hidden_channels), nn.ReLU())))
        self.convs.append(GINConv(
            nn.Sequential(Linear(hidden_channels, hidden_channels),
                       BatchNorm(hidden_channels), nn.ReLU(), nn.Dropout(p=dropout),
                       Linear(hidden_channels, out_channels), nn.ReLU())))

    def forward(self, x, edge_index):
        # Node embeddings 
        for conv in self.convs:
            x = conv(x, edge_index)

        return x