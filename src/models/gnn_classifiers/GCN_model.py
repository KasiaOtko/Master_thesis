import torch
import torch.nn.functional as F
from torch import batch_norm, nn
from torch_geometric.nn import (BatchNorm, GCNConv, Linear,  # type: ignore
                                RGCNConv)


class GCN(nn.Module):
    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float, num_layers: int, linear_l: bool
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.skips = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.linear_l = linear_l

        self.lin1 = Linear(in_channels, hidden_channels)
        if self.linear_l:
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=False))
            self.skips.append(Linear(hidden_channels, hidden_channels))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels, cached=False))
            self.skips.append(Linear(in_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=False))
            self.skips.append(Linear(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))

        if self.linear_l:
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.skips.append(Linear(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))
            self.lin2 = Linear(hidden_channels, out_channels)
        else:
            self.convs.append(GCNConv(hidden_channels, out_channels))
            self.skips.append(Linear(hidden_channels, out_channels))
        self.dropout = nn.Dropout(dropout)
        


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("Expected input is not a 2D tensor,"
                             f"instead it is a {x.ndim}D tensor.")

        if self.linear_l:
            x = self.lin1(x).relu()
            for i, (conv, skip, batch_norm) in enumerate(zip(self.convs, self.skips, self.batch_norms)):
                x = conv(x, edge_index) + skip(x)
                if i == self.num_layers - 1:
                    h = x.clone()
                x = batch_norm(x)
                x = F.relu(x)
                x = self.dropout(x)
            x = self.lin2(x)

        else:

            for i, (conv, skip, batch_norm) in enumerate(zip(self.convs[:-1], self.skips[:-1], self.batch_norms)):
                x = conv(x, edge_index) + skip(x)
                if i == self.num_layers - 2:
                    h = x.clone()
                x = batch_norm(x)
                x = F.relu(x)
                x = self.dropout(x)
            x = self.convs[-1](x, edge_index) + self.skips[-1](x)

        return h, x


class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, dropout, num_bases, num_layers, linear_l):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.skips = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.linear_l = linear_l

        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations, num_bases=num_bases))
        self.skips.append(Linear(in_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases))
            self.skips.append(Linear(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))
        self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases))
        self.skips.append(Linear(hidden_channels, hidden_channels))
        self.dropout = nn.Dropout(dropout)
        self.linear = Linear(hidden_channels, out_channels)
        

    def forward(self, x, edge_index, edge_type):
        for i, (conv, skip, batch_norm) in enumerate(zip(self.convs[:-1], self.skips[:-1], self.batch_norms)):
            x = conv(x, edge_index, edge_type) + skip(x)
            if i == self.num_layers - 1:
                    h = x.copy()
            x = batch_norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index, edge_type) + self.skips[-1](x)


        return h, x


class RGCN_no_skips(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, dropout, num_bases, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.skips = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations, num_bases=num_bases))
        #self.skips.append(Linear(in_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases))
            #self.skips.append(Linear(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))
        self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases))
        #self.skips.append(Linear(hidden_channels, hidden_channels))
        self.dropout = nn.Dropout(dropout)
        self.linear = Linear(hidden_channels, out_channels)
        

    def forward(self, x, edge_index, edge_type):
        for conv, batch_norm in zip(self.convs[:-1], self.batch_norms):
            x = conv(x, edge_index, edge_type)
            x = batch_norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index, edge_type)

        return x


class GCN_no_skips(nn.Module):
    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float, num_layers: int
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.skips = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False))
        #self.skips.append(Linear(in_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=False))
            #self.skips.append(Linear(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        #self.skips.append(Linear(hidden_channels, hidden_channels))
        self.dropout = nn.Dropout(dropout)
        self.linear = Linear(hidden_channels, out_channels)


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("Expected input is not a 2D tensor,"
                             f"instead it is a {x.ndim}D tensor.")

        for conv, batch_norm in zip(self.convs[:-1], self.batch_norms):
            x = conv(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)

        return x