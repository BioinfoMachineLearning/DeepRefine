import random

import numpy as np
import torch

from project.datasets.RG.rg_dgl_dataset import get_rgraph
from project.modules.egr.egr import IterativeEquivariantGraphRefinementModel
from project.utils.deeprefine_constants import BASE_AMINO_ACIDS, PROT_ATOM_NAMES
from project.utils.egr.utils import rotate


def test_egr():
    # Seed everything
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Declare hyperparameters for testing
    ca_only = False
    num_nodes = 200
    knn = 20
    num_iegr_layers = 5
    num_iegr_iter = 1
    num_hidden_channels = 128
    num_edge_input_feats = 14 if ca_only else 15
    num_attn_heads = 8
    num_atom_types = len(BASE_AMINO_ACIDS) if ca_only else len(PROT_ATOM_NAMES)
    atom_emb_dim = num_hidden_channels
    use_surf_prox = True
    use_dihedral_angles = True
    use_edge_feats_in_iegr_layers = True
    use_dist_in_iegr_layers = True
    use_local_attention = False
    use_global_attention = False
    attn_depth = 1
    update_pos_with_ca_atoms = False
    batched_input = False
    pooling = 'max'
    pred_struct_qual = True
    dropout = 0.0
    nonlin = 'lkyrelu'
    layer_norm = 'LN'
    layer_norm_coords = '0'
    skip_weight_h = 0.0
    x_connection_init = 0.0
    leakyrelu_neg_slope = 0.2
    shared_layers = False
    norm_coord_updates = True
    coord_norm_const = 1.0
    learn_skip_weights = False
    use_fairscale = False
    use_deepspeed = False
    manually_init_weights = False
    verbose = False
    pdb_filepaths = ['']

    # Initialize EGR model for testing
    egr_model = IterativeEquivariantGraphRefinementModel(
        num_iegr_layers=num_iegr_layers,
        num_iegr_iter=num_iegr_iter,
        num_hidden_feats=num_hidden_channels,
        num_edge_input_feats=num_edge_input_feats,
        num_attn_heads=num_attn_heads,
        num_atom_types=num_atom_types,
        atom_emb_dim=atom_emb_dim,
        use_surf_prox=use_surf_prox,
        use_dihedral_angles=use_dihedral_angles,
        use_edge_feats_in_iegr_layers=use_edge_feats_in_iegr_layers,
        use_dist_in_iegr_layers=use_dist_in_iegr_layers,
        use_local_attention=use_local_attention,
        use_global_attention=use_global_attention,
        attn_depth=attn_depth,
        update_pos_with_ca_atoms=update_pos_with_ca_atoms,
        knn=knn,
        batched_input=batched_input,
        pooling=pooling,
        pred_struct_qual=pred_struct_qual,
        dropout=dropout,
        nonlin=nonlin,
        layer_norm=layer_norm,
        layer_norm_coords=layer_norm_coords,
        skip_weight_h=skip_weight_h,
        x_connection_init=x_connection_init,
        leakyrelu_neg_slope=leakyrelu_neg_slope,
        shared_layers=shared_layers,
        norm_coord_updates=norm_coord_updates,
        coord_norm_const=coord_norm_const,
        learn_skip_weights=learn_skip_weights,
        use_fairscale=use_fairscale,
        use_deepspeed=use_deepspeed,
        manually_init_weights=manually_init_weights,
        ca_only=ca_only,
        verbose=verbose
    )

    # Create random transformation tensors
    R = rotate(*torch.rand(3))
    t = torch.randn(1, 3)

    # Generate random graph
    rgraph = get_rgraph(
        num_nodes=num_nodes,
        knn=knn,
        self_loops=False,
        dtype=torch.FloatTensor,
        ca_only=ca_only
    )

    # Assemble node features for propagation
    node_feats = rgraph.ndata['atom_type']  # Use atom types as a proxy during testing for all node features
    node_coords = rgraph.ndata['x_pred']

    # Cache first two nodes' features
    node1 = node_feats[0, :]
    node2 = node_feats[1, :]

    # Switch first and second nodes' positions
    node_feats_permuted_row_wise = node_feats.clone().detach()
    node_feats_permuted_row_wise[0, :] = node2
    node_feats_permuted_row_wise[1, :] = node1

    # Store latest node and edge features in base random graphs
    rgraph.ndata['atom_type'] = node_feats
    rgraph1, rgraph2, rgraph3 = rgraph.clone(), rgraph.clone(), rgraph.clone()

    # Pass messages over graph (Stage 1)
    rgraph1.ndata['x_pred'] = (node_coords @ R + t).clone().detach()
    rgraph1 = egr_model(rgraph1, pdb_filepaths)
    nf1, nc1 = rgraph1.ndata['f'], rgraph1.ndata['x_pred']

    # Pass messages over graph (Stage 2)
    rgraph2 = rgraph.clone()
    rgraph2 = egr_model(rgraph2, pdb_filepaths)
    nf2, nc2 = rgraph2.ndata['f'], rgraph2.ndata['x_pred']

    # Pass messages over graph (Stage 3)
    rgraph3 = rgraph.clone()
    rgraph3.ndata['atom_type'] = node_feats_permuted_row_wise.clone().detach()
    rgraph3 = egr_model(rgraph3, pdb_filepaths)
    nf3, n_coords3 = rgraph3.ndata['f'], rgraph3.ndata['x_pred']

    # Assess model's equivariance across all three stages
    assert torch.allclose(nf1, nf2, atol=1e-5 if nonlin == 'lkyrelu' else 1e-6), 'Type 0 feats must be E(3) invariant'
    assert not torch.allclose(nf1, nf3, atol=1e-6), 'Type 0 feats must be equivariant to node permutations'
    assert torch.allclose(nc1, (nc2 @ R + t), atol=1e-6), 'Type 1 feats must be E(3) equivariant'
    print(f'The EGR model passes all tests for equivariance')


if __name__ == '__main__':
    test_egr()
