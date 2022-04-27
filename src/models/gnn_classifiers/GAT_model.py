import torch
from torch_geometric.nn import BatchNorm, Linear, GATConv, RGATConv
from torch import nn
import torch.nn.functional as F

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, heads, num_layers):
        super().__init__()
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_products_gat.py
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.skips = torch.nn.ModuleList()

        self.convs.append(GATConv(in_channels, hidden_channels, heads, dropout=0.2))
        self.batch_norms.append(BatchNorm(hidden_channels*heads))
        self.skips.append(Linear(in_channels, hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(heads * hidden_channels, hidden_channels, heads))
            self.batch_norms.append(BatchNorm(hidden_channels * heads))
            self.skips.append(Linear(hidden_channels * heads, hidden_channels * heads))
        self.convs.append(GATConv(heads * hidden_channels, out_channels, heads, concat=False))
        self.skips.append(Linear(hidden_channels * heads, out_channels))
        
        # for _ in range(num_layers - 2):
        #     self.skips.append(Linear(hidden_channels * heads, hidden_channels * heads))
        # self.skips.append(Linear(hidden_channels * heads, out_channels))

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        for conv, skip, batch_norm in zip(self.convs[:-1], self.skips[:-1], self.batch_norms):
            x = conv(x, edge_index) + skip(x)
            x = batch_norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index) + self.skips[-1](x)
        return x


class RGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, dropout, heads, num_bases, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.skips = torch.nn.ModuleList()

        self.convs.append(RGATConv(in_channels, hidden_channels, num_relations, num_bases, heads=heads, dropout=0.2))
        self.batch_norms.append(BatchNorm(hidden_channels * heads))
        self.skips.append(Linear(in_channels, hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.convs.append(RGATConv(heads * hidden_channels, hidden_channels, num_relations, num_bases, heads=heads))
            self.batch_norms.append(BatchNorm(hidden_channels * heads))
            self.skips.append(Linear(hidden_channels * heads, hidden_channels * heads))
        self.convs.append(RGATConv(heads * hidden_channels, out_channels, num_relations, num_bases, heads=heads, concat=False))
        self.skips.append(Linear(hidden_channels * heads, out_channels))
        self.dropout = nn.Dropout(dropout)            
        

    def forward(self, x, edge_index, edge_type):

        for conv, skip, batch_norm in zip(self.convs[:-1], self.skips[:-1], self.batch_norms):
            x = conv(x, edge_index, edge_type) + skip(x)
            x = batch_norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index, edge_type) + self.skips[-1](x)
        return x