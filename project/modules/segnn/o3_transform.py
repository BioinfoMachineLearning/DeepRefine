import torch
from e3nn.o3 import Irreps, spherical_harmonics
from torch_geometric.data import Data
from torch_scatter import scatter


class O3Transform:
    """Graph transformation respecting O(3) equivariance."""
    def __init__(self, lmax_attr: int):
        self.attr_irreps = Irreps.spherical_harmonics(lmax_attr)

    def __call__(self, graph: Data) -> Data:
        """Transform a given graph by appending to it new equivariant node and edge features."""
        pos = graph.pos

        rel_pos = pos[graph.edge_index[0]] - pos[graph.edge_index[1]]
        edge_dist = torch.sqrt(rel_pos.pow(2).sum(1, keepdims=True))

        graph.edge_attr = spherical_harmonics(self.attr_irreps, rel_pos, normalize=True, normalization='integral')
        graph.node_attr = scatter(graph.edge_attr, graph.edge_index[1], dim=0, reduce='mean')

        mean_pos = pos.mean(0, keepdims=True)

        graph.x = torch.cat((pos - mean_pos, graph.ndata), dim=1).float()  # Must make `x` a float here for SEGR ops
        graph.additional_message_features = torch.cat((edge_dist, graph.edata), dim=-1)
        return graph
