import random

import numpy as np
import torch
from e3nn.o3 import Irreps

from project.datasets.RG.rg_dgl_dataset import get_rgraph
from project.modules.segnn.balanced_irreps import WeightBalancedIrreps
from project.modules.segnn.o3_transform import O3Transform
from project.modules.segnn.segnn import SteerableEquivariantGraphRefinementModel
from project.utils.egr.utils import rotate


def test_segnn():
    # Seed everything
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Declare hyperparameters for testing
    ca_only = True
    num_nodes = 200
    knn = 20
    num_layers = 7
    num_hidden_channels = 128
    num_node_input_feats = 31
    num_edge_input_feats = 14
    layer_norm = 'IN'
    lmax_attr = 3
    lmax_h = 2

    # Initialize SEGNN model for testing
    input_irreps = Irreps(f'1x1o + {num_node_input_feats}x0e')
    edge_attr_irreps = Irreps.spherical_harmonics(lmax_attr)
    node_attr_irreps = Irreps.spherical_harmonics(lmax_attr)
    hidden_irreps = WeightBalancedIrreps(
        Irreps(f'{num_hidden_channels}x0e'),
        node_attr_irreps,
        sh=True,
        lmax=lmax_h
    )
    output_irreps = Irreps('1x1o + 1x0e')
    additional_message_irreps = Irreps(f'{1 + num_edge_input_feats}x0e')
    pos_skip_connection = True  # Pos. skip connections must be enabled to guarantee E(3) equivariance for type-1 feats
    use_attention = True
    num_attention_heads = 4

    segnn_model = SteerableEquivariantGraphRefinementModel(
        input_irreps=input_irreps,
        hidden_irreps=hidden_irreps,
        output_irreps=output_irreps,
        edge_attr_irreps=edge_attr_irreps,
        node_attr_irreps=node_attr_irreps,
        num_layers=num_layers,
        norm=layer_norm,
        additional_message_irreps=additional_message_irreps,
        pos_skip_connection=pos_skip_connection,
        use_attention=use_attention,
        num_attention_heads=num_attention_heads
    )

    # Create random transformation tensors
    R = rotate(*torch.rand(3))
    t = torch.randn(1, 3)

    # Generate random graph
    o3_transform = O3Transform(lmax_attr=lmax_attr)
    rgraph = get_rgraph(
        num_nodes=num_nodes,
        knn=knn,
        self_loops=False,
        dtype=torch.FloatTensor,
        using_segnn=True,
        o3_transform=o3_transform,
        ca_only=ca_only
    )

    # Assemble node features for propagation #
    node_feats = rgraph.ndata
    node_coords = rgraph.pos

    # Cache first two nodes' features
    node1 = node_feats[0, :]
    node2 = node_feats[1, :]

    # Switch first and second nodes' positions
    node_feats_permuted_row_wise = node_feats.clone().detach()
    node_feats_permuted_row_wise[0, :] = node2
    node_feats_permuted_row_wise[1, :] = node1

    # Store latest node and edge features in base random graphs
    rgraph.ndata = node_feats
    rgraph1, rgraph2, rgraph3 = rgraph.clone(), rgraph.clone(), rgraph.clone()

    # Pass messages over graph (Stage 1)
    rgraph1.pos = (node_coords @ R + t).clone().detach()
    rgraph1 = o3_transform(rgraph1)  # Explicitly update spherical harmonics embeddings now that positions have changed
    rgraph1 = segnn_model(rgraph1)
    nf1, nc1 = rgraph1.ndata, rgraph1.pos

    # Pass messages over graph (Stage 2)
    rgraph2 = rgraph.clone()  # Make a forward pass with the original input graph, to make it into a baseline
    rgraph2 = segnn_model(rgraph2)
    nf2, nc2 = rgraph2.ndata, rgraph2.pos

    # Pass messages over graph (Stage 3)
    rgraph3 = rgraph.clone()
    rgraph3.x[:, 3:] = node_feats_permuted_row_wise.clone().detach()  # Simply update ndata in-place
    rgraph3 = segnn_model(rgraph3)
    nf3, n_coords3 = rgraph3.ndata, rgraph3.pos

    # Assess model's equivariance across all three stages
    assert torch.allclose(nf1, nf2, atol=1e-6), 'Type 0 feats must be E(3) invariant'
    assert not torch.allclose(nf1, nf3, atol=1e-6), 'Type 0 feats must be equivariant to node permutations'
    assert torch.allclose(nc1, (nc2 @ R + t), atol=1e-6), 'Type 1 feats must be E(3) equivariant'
    print(f'The SEGNN model passes all tests for equivariance')


if __name__ == '__main__':
    test_segnn()
