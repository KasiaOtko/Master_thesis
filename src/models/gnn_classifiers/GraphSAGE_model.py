import torch
import torch.nn.functional as F
from torch import batch_norm, nn
from torch_geometric.nn import BatchNorm, Linear, SAGEConv  # type: ignore


class SAGE(nn.Module):
    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float, num_layers: int, aggr: str, linear_l: bool
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.linear_l - linear_l
        self.convs = torch.nn.ModuleList()
        self.skips = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        self.lin1 = Linear(in_channels, hidden_channels)
        if self.linear_l:
            self.convs.append(SAGEConv(hidden_channels, hidden_channels,  aggr = aggr, root_weight = False))
            self.skips.append(Linear(hidden_channels, hidden_channels))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels, aggr = aggr, root_weight = False))
            self.skips.append(Linear(in_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr = aggr, root_weight = False))
            self.skips.append(Linear(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))

        if self.linear_l:
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr = aggr, root_weight = False))
            self.skips.append(Linear(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))
            self.lin2 = Linear(hidden_channels, out_channels)
        else:
            self.convs.append(SAGEConv(hidden_channels, out_channels, aggr = aggr, root_weight = False))
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