import torch
from torch_geometric.nn import GATConv, RGATConv
from torch import nn
import torch.nn.functional as F

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, heads):
        super().__init__()

        # self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=dropout)
        # self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False,
        #                      dropout=dropout)
        # self.dropout = nn.Dropout(dropout)
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels * heads, heads=1,
                             dropout=dropout)
        self.conv3 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False,
                             dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        # x = F.relu(self.conv1(x, edge_index))
        # x = self.dropout(x)
        # x = self.conv2(x, edge_index)
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        return x


class RGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, dropout, heads, num_bases):
        super().__init__()
        self.conv1 = RGATConv(in_channels, hidden_channels, num_relations, num_bases, heads=heads)
        self.conv2 = RGATConv(hidden_channels * heads, hidden_channels * heads, num_relations, num_bases, heads=1)
        self.conv3 = RGATConv(hidden_channels * heads, out_channels, num_relations, num_bases, heads=1, concat=False)
        self.dropout = nn.Dropout(dropout)
        #self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_type).relu()
        x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_type)
        #x = self.lin(x)
        return x