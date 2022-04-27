import torch
from torch_geometric.nn import BatchNorm, GATConv, RGATConv
from torch import nn
import torch.nn.functional as F

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, heads, num_layers):
        super().__init__()
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_products_gat.py
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads, dropout=0.2))
        self.batch_norms.append(BatchNorm(hidden_channels*heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(heads * hidden_channels, hidden_channels, heads))
            self.batch_norms.append(BatchNorm(hidden_channels*heads))
        self.convs.append(GATConv(heads * hidden_channels, out_channels, heads, concat=False))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        for conv, batch_norm in zip(self.convs[:-1], self.batch_norms):
            x = conv(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        # x = F.relu(self.conv1(x, edge_index))
        # x = self.dropout(x)
        # x = self.conv2(x, edge_index)
        #x = F.relu(self.conv1(x, edge_index))
        #x = self.dropout(x)
        #x = F.relu(self.conv2(x, edge_index))
        #x = self.dropout(x)
        #x = self.conv3(x, edge_index)
        return x


class RGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, dropout, heads, num_bases, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.convs.append(RGATConv(in_channels, hidden_channels, num_relations, num_bases, heads=heads, dropout=0.2))
        self.batch_norms.append(BatchNorm(hidden_channels*heads))
        for _ in range(num_layers - 2):
            self.convs.append(RGATConv(heads * hidden_channels, hidden_channels, num_relations, num_bases, heads=heads))
            self.batch_norms.append(BatchNorm(hidden_channels*heads))
        self.convs.append(RGATConv(heads * hidden_channels, out_channels, num_relations, num_bases, heads=heads, concat=False))
        self.dropout = nn.Dropout(dropout)

        # self.conv1 = RGATConv(in_channels, hidden_channels, num_relations, num_bases, heads=heads)
        # self.conv2 = RGATConv(hidden_channels * heads, hidden_channels * heads, num_relations, num_bases, heads=1)
        # self.conv3 = RGATConv(hidden_channels * heads, out_channels, num_relations, num_bases, heads=1, concat=False)
        #self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_type):

        for conv, batch_norm in zip(self.convs[:-1], self.batch_norms):
            x = conv(x, edge_index, edge_type)
            x = batch_norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index, edge_type)
        #x = self.conv1(x, edge_index, edge_type).relu()
        #x = self.dropout(x)
        #x = self.conv2(x, edge_index, edge_type).relu()
        #x = self.dropout(x)
        #x = self.conv3(x, edge_index, edge_type)
        #x = self.lin(x)
        return x