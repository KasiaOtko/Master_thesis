import torch
import torch.nn.functional as F
from torch import batch_norm, nn
from torch_geometric.nn import BatchNorm, GCNConv, RGCNConv  # type: ignore


class GCN(nn.Module):
    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float, num_layers: int
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False))
        self.batch_norms.append(BatchNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=False))
            self.batch_norms.append(BatchNorm(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = nn.Dropout(dropout)
        # self.conv1 = GCNConv(in_channels, hidden_channels, cached=False)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels, cached=False)
        # self.conv3 = GCNConv(hidden_channels, out_channels)
        # self.linear = nn.Linear(hidden_channels, out_channels)
        
        # self.batch_norm1 = nn.BatchNorm1d(hidden_channels)

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

        # x = self.batch_norm1(self.conv1(x, edge_index))
        # x = self.dropout(F.relu(x))
        # x = self.batch_norm1(self.conv2(x, edge_index))
        # x = self.dropout(F.relu(x))
        # x = self.conv3(x, edge_index)
        # x = self.dropout(F.relu(x))
        # x = self.linear(x)
        return x

    def inference(self, x_all, subgraph_loader, device):

        #for i, conv in enumerate(self.convs):
        xs = []
        for batch_size, n_id, adj in subgraph_loader:
            edge_index, _, size = adj.to(device)
            x = x_all[n_id].to(device)
            x_target = x[:size[1]]
            x = self.batch_norm1(self.conv1((x, x_target), edge_index))
            x = self.dropout(F.relu(x))
            x = self.conv2((x, x_target), edge_index)
            x = F.log_softmax(x, dim=-1)
            xs.append(x.cpu())

        x_all = torch.cat(xs, dim=0)
        return x_all


class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, dropout, num_bases, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations, num_bases=num_bases))
        self.batch_norms.append(BatchNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases))
            self.batch_norms.append(BatchNorm(hidden_channels))
        self.convs.append(RGCNConv(hidden_channels, out_channels, num_relations, num_bases=num_bases))

        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations, num_bases=num_bases)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases) #30
        self.conv3 = RGCNConv(hidden_channels, out_channels, num_relations, num_bases=num_bases) 
        self.linear = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_channels)
        self.batch_norm2 = nn.BatchNorm1d(hidden_channels)

    def forward(self, x, edge_index, edge_type):
        # for i, conv in enumerate(self.convs):
        #     x = conv(x, edge_index, edge_type)
        #     if i < len(self.convs) - 1:
        #         x = x.relu()
        #         x = F.dropout(x)
        for conv, batch_norm in zip(self.convs[:-1], self.batch_norms):
            x = conv(x, edge_index, edge_type)
            x = batch_norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index, edge_type)

        # x = self.batch_norm1(self.conv1(x, edge_index, edge_type))
        # x = self.dropout(F.relu(x))
        # x = self.batch_norm2(self.conv2(x, edge_index, edge_type))
        # x = self.dropout(F.relu(x))
        # x = self.conv3(x, edge_index, edge_type)
        # x = self.dropout(F.relu(x))
        # x = self.linear(x)
        return x