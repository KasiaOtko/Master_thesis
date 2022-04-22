import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, RGCNConv  # type: ignore


class GCN(nn.Module):
    def __init__(
        self, hidden_channels: int, num_features: int, num_classes: int, dropout: float
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("Expected input is not a 2D tensor,"
                             f"instead it is a {x.ndim}D tensor.")

        x = self.conv1(x, edge_index)
        x = self.dropout(F.relu(x))
        x = self.conv2(x, edge_index)
        return x


class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super().__init__()
        self.conv1 = RGCNConv(in_channels=in_channels, out_channels=hidden_channels, num_relations=num_relations,
                              num_bases=None)
        self.conv2 = RGCNConv(in_channels=hidden_channels, out_channels=out_channels, num_relations=num_relations,
                              num_bases=None) #30

    def forward(self, edge_index, edge_type):
        x = F.relu(self.conv1(None, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return x