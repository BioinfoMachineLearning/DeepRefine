import logging
from typing import Dict, Tuple, Optional, Literal, List, Union

import deepspeed
import dgl
import fairscale.nn
import torch
import torch.nn as nn
from dgl import function as fn

from project.modules.set.layers.pooling import GPooling
from project.utils.deeprefine_utils import is_divisible_by
from project.utils.egr.utils import get_nonlin, get_layer_norm, LocalMultiHeadAttentionModule
from project.utils.egr.utils import iegr_copy_without_weak_connections
from project.utils.set.runtime.utils import update_relative_positions, update_potential_values
from project.utils.segnn.utils import LinearAttentionTransformer


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for DeepRefine (https://github.com/BioinfoMachineLearning/DeepRefine):
# -------------------------------------------------------------------------------------------------------------------------------------
class IEGRLayer(nn.Module):
    """Iterative E(n)-equivariant graph structure refinement layer as a DGL module."""

    def __init__(
            self,
            num_node_input_feats: int,
            num_hidden_feats: int,
            num_edge_input_feats: int,
            num_output_feats: int,
            num_attn_heads: int,
            use_dist_in_iegr_layers: bool,
            use_local_attention: bool,
            use_global_attention: bool,
            attn_depth: int,
            update_pos_with_ca_atoms: bool,
            dropout=0.0,
            nonlin='lkyrelu',
            layer_norm='LN',
            layer_norm_coords='0',
            skip_weight_h=0.75,
            x_connection_init=0.0,
            leakyrelu_neg_slope=1e-2,
            update_coords=True,
            norm_coord_updates=True,
            coord_norm_const=1.0,
            learn_skip_weights=False,
            use_fairscale=False,
            use_deepspeed=False,
            verbose=False
    ):
        """Iterative E(n)-Equivariant Graph Structure Refinement Layer
        Parameters
        ----------
        num_node_input_feats: int
            Node input representation size.
        num_hidden_feats: int
            Hidden representation size.
        num_edge_input_feats: int
            Edge input representation size.
        num_output_feats: int
            Output representation size.
        num_attn_heads: int
            Number of attention heads to employ in multi-head attention computations.
        use_dist_in_iegr_layers:
            Whether to include distance-based edge features in each iteration of message-passing.
        use_local_attention: bool
            Whether to compute node-wise attention scores node-locally within the IEGR model.
        use_global_attention: bool
            Whether to compute node-wise attention scores node-globally within the IEGR model.
        attn_depth: int
            How many global attention layers to apply within the IEGR model.
        update_pos_with_ca_atoms: bool
            Whether to update the position of non-Ca atoms using their Ca atom displacements.
        dropout: float
            Rate of dropout to apply within IEGR layers.
        nonlin: str
            Which nonlinearity to apply within IEGR layers.
        layer_norm: str
            Which version of normalization to apply to each layer's learned representations.
        layer_norm_coords: str
            Which version of normalization to apply to each set of updated coordinates.
        skip_weight_h: float
            Skip weight to apply to node representations.
        x_connection_init: float
            Initial connection to node coordinates.
        leakyrelu_neg_slope: float
            Value for the LeakyReLU function's negative slope.
        update_coords: bool
            Whether to update coordinates in an equivariant manner.
        norm_coord_updates: bool
            Whether to normalize coordinate updates by their L2-norm plus a constant.
        coord_norm_const: float
            A normalizing constant for coordinate updates.
        learn_skip_weights: bool
            Whether to learn skip connection weights.
        use_fairscale: bool
            Whether to perform activation checkpointing with FairScale.
        use_deepspeed: bool
            Whether to perform activation checkpointing with DeepSpeed.
        verbose: bool
            Whether to log node coordinates and representations for manual inspection.
        """
        super().__init__()
        assert coord_norm_const >= 0.0, 'Coordinate update normalizing constant must be non-negative'

        # Initialize model parameters
        self.num_node_input_feats = num_node_input_feats
        self.num_hidden_feats = num_hidden_feats
        self.num_edge_input_feats = num_edge_input_feats
        self.num_output_feats = num_output_feats
        self.num_attn_heads = num_attn_heads
        self.use_dist_in_iegr_layers = use_dist_in_iegr_layers
        self.use_local_attention = use_local_attention
        self.use_global_attention = use_global_attention
        self.use_attention = use_local_attention or use_global_attention
        self.update_pos_with_ca_atoms = update_pos_with_ca_atoms

        self.skip_weight_h = skip_weight_h
        self.x_connection_init = x_connection_init

        self.update_coords = update_coords
        self.norm_coord_updates = norm_coord_updates
        self.coord_norm_const = coord_norm_const
        self.learn_skip_weights = learn_skip_weights
        self.use_fairscale = use_fairscale
        self.use_deepspeed = use_deepspeed
        self.verbose = verbose

        self.all_sigmas_dist = [1.5 ** x for x in range(15)]

        """Define layers and auxiliary modules for edge representations."""

        # Define the multi-layer perceptron (MLP) for edge representations (i.e., the edge MLP)
        edges_mlp_input_dim = (num_hidden_feats * 2) + num_edge_input_feats + len(self.all_sigmas_dist)
        self.edges_mlp = nn.Sequential(
            nn.Linear(edges_mlp_input_dim, num_output_feats),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            get_nonlin(nonlin, leakyrelu_neg_slope),
            get_layer_norm(layer_norm, num_output_feats),
            nn.Linear(num_output_feats, num_output_feats),
            get_nonlin(nonlin, leakyrelu_neg_slope)
        )

        # Optionally use FairScale activation checkpointing for message passing since, in our graphs, E >> N
        if self.use_fairscale:
            self.edges_mlp = fairscale.nn.checkpoint_wrapper(self.edges_mlp)

        """Define layers and auxiliary modules for node representations."""

        # Establish a normalization technique for node representations
        self.nodes_norm = nn.LayerNorm(num_hidden_feats)  # Note: Originally nn.Identity()

        # Set up queries, keys, and values for node-wise multi-head attention (MHA) mechanism
        if use_local_attention:
            self.attn_module = LocalMultiHeadAttentionModule(num_hidden_feats,
                                                             num_hidden_feats // num_attn_heads,
                                                             num_attn_heads,
                                                             use_bias=False)
        elif use_global_attention:
            self.attn_module = LinearAttentionTransformer(dim=num_hidden_feats,
                                                          heads=num_attn_heads,
                                                          depth=attn_depth,
                                                          max_seq_len=12288)

        # Define the MLP for node representations (i.e., the node MLP)
        nodes_mlp_input_dim = num_hidden_feats + num_output_feats + num_hidden_feats + num_node_input_feats
        self.nodes_mlp = nn.Sequential(
            nn.Linear(nodes_mlp_input_dim, num_hidden_feats),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            get_nonlin(nonlin, leakyrelu_neg_slope),
            get_layer_norm(layer_norm, num_hidden_feats),
            nn.Linear(num_hidden_feats, num_output_feats),
        )

        """Define layers and auxiliary modules for node coordinates."""
        if update_coords:
            # Define the MLP for node coordinates (i.e., the coordinates MLP)
            final_coords_layer = nn.Linear(num_output_feats, 1, bias=False)
            torch.nn.init.xavier_uniform_(final_coords_layer.weight, gain=0.001)  # Ensure initial transforms are small
            self.coords_mlp = nn.Sequential(
                # Note: Scalar weight to be multiplied by (x_i - x_j)
                nn.Linear(num_output_feats, num_output_feats),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                get_nonlin(nonlin, leakyrelu_neg_slope),
                get_layer_norm(layer_norm_coords, num_output_feats),
                final_coords_layer,
                # nn.Tanh() if use_tanh_in_iegr_layers else nn.Identity()
            )

        self.x_connection_init = x_connection_init
        self.skip_weight_h: Union[float, torch.Tensor] = skip_weight_h
        if self.learn_skip_weights and self.num_hidden_feats == self.num_output_feats:
            self.skip_weight_h = nn.Parameter(data=torch.Tensor(1), requires_grad=True)

    def build_edge_dists(self, graph: dgl.DGLGraph):
        """
        Construct all distance-related edge features required for E(n)-equivariant message-passing.

        Parameters
        ----------
        graph: dgl.DGLGraph
            DGL input graph.
        """
        # Compute relative position of each source node from its respective destination node
        graph.apply_edges(fn.u_sub_v('x_latest_to_update', 'x_latest_to_update', 'x_rel'))  # x_i-x_j -> (num_edges, 3)

        # Normalize relative positions using their L2-norm and a stabilizing constant
        if self.norm_coord_updates:
            norm_term = torch.sqrt(torch.sum(graph.edata['x_rel'] ** 2, dim=1, keepdim=True)) + self.coord_norm_const
            graph.edata['x_rel'] = graph.edata['x_rel'] / norm_term

        # Calculate the relative magnitude of the weight between each connected pair of atoms using multiple RBFs
        if self.use_dist_in_iegr_layers:
            x_rel_mag = graph.edata['x_rel'] ** 2
            x_rel_mag = torch.sum(x_rel_mag, dim=1, keepdim=True)  # Note: ||x_i - x_j|| ^ 2 -> (num_edges, 1)
            x_rel_mag = torch.cat([torch.exp(-x_rel_mag / sigma) for sigma in self.all_sigmas_dist], dim=-1)
        else:
            # Otherwise, "zero-out" the magnitude-based edge distance features that would have been derived above
            x_rel_mag = torch.zeros(
                (graph.edata['x_rel'].size(dim=0), len(self.all_sigmas_dist)),
                device=graph.device,
                dtype=graph.edata['x_rel'].dtype
            ).detach()

        # Install relative magnitudes as a new edge feature
        graph.edata['x_rel_mag'] = x_rel_mag

    def message_func(self, edges: dgl.udf.EdgeBatch) -> Dict[str, torch.Tensor]:
        """
        In an E(n)-equivariant manner, compute the messages for an EdgeBatch of edges.
        This function is set up as a User Defined Function in DGL.

        Parameters
        ----------
        edges: EdgeBatch
            A batch of edges for which to compute messages.

        Returns
        ----------
        Dict[str, torch.Tensor]
            New edge messages.
        """
        # Assemble inter-node messages
        edge_mlp_input = torch.cat([
            edges.src['feats'],
            edges.dst['feats'],
            edges.data['orig_feats'],
            edges.data['x_rel_mag']
        ], dim=1)
        if self.use_deepspeed:
            m_ij = deepspeed.checkpointing.checkpoint(self.edges_mlp, edge_mlp_input)
        else:
            m_ij = self.edges_mlp(edge_mlp_input)

        # Preserve edge messages in the event that a DGL reduce_func() is called
        edges.data['m_ij'] = m_ij
        return {'m_ij': m_ij}

    def compute_node_attn(self, graph: dgl.DGLGraph):
        """
        In an E(n)-invariant manner, compute MHA representations for each node in the input graph.

        Parameters
        ----------
        graph: dgl.DGLGraph
            DGL input graph.
        """
        # Apply attention to get new node representations as requested
        if self.use_attention:
            attn_node_feats = graph.ndata['feats'].unsqueeze(0)
            attn_module_args = [graph, attn_node_feats] if self.use_local_attention else [attn_node_feats]
            if self.use_deepspeed:
                graph.ndata['aggr_mha_msg'] = deepspeed.checkpointing.checkpoint(
                    self.attn_module, *attn_module_args
                ).reshape(-1, self.num_hidden_feats)
            else:
                graph.ndata['aggr_mha_msg'] = self.attn_module(
                    *attn_module_args
                ).reshape(-1, self.num_hidden_feats)
        else:
            graph.ndata['aggr_mha_msg'] = (graph.ndata['feats'].detach() * 0.0).detach()  # Zero-out attention features

        # Log node-wise attention outputs for manual inspection
        if self.verbose:
            logging.info('aggr_mha_msg(i): sum_j a_{i,j} * h_j = ' + str(torch.max(graph.ndata['aggr_mha_msg'])))

    def update_node_coords(self, graph: dgl.DGLGraph, ca_atom_node_idx: torch.Tensor):
        """
        In an E(n)-equivariant manner, update the node coordinates in the input graph.

        Parameters
        ----------
        graph: dgl.DGLGraph
            DGL input graph.
        ca_atom_node_idx: torch.Tensor
            Node indices of Ca atoms.
        """
        # Compute edge-wise update coefficients and positional moments
        if self.use_deepspeed:
            # Note: \phi ^ x(m_{i->j})
            edge_coef = deepspeed.checkpointing.checkpoint(self.coords_mlp, graph.edata['m_ij'])
        else:
            edge_coef = self.coords_mlp(graph.edata['m_ij'])
        graph.edata['x_moment'] = graph.edata['x_rel'] * edge_coef  # (x_i - x_j) * \phi ^ x(m_{i->j})

        # Log edge-wise update coefficients and positional moments for manual inspection
        if self.verbose:
            logging.info('edge_coef: \phi ^ x(m_{i->j}) = ' + str(torch.max(edge_coef)))
            logging.info('x_moment: (x_i - x_j) * \phi ^ x(m_{i->j}) = ' + str(torch.max(graph.edata['x_moment'])))

        # Compute the equivariant update to each node's coordinates via efficient DGL message-passing routines
        graph.update_all(fn.copy_edge('x_moment', 'm'), fn.mean('m', 'x_update'))  # Can also use a `sum` update here

        # Determine how to update coordinates (i.e., whether to use Ca centroids or individual positional updates)
        if self.update_pos_with_ca_atoms:
            # Use updates to atoms' displacements to update their coordinates, using residues as "centroids" for atoms
            x_latest = torch.zeros(
                (graph.num_nodes(), graph.ndata['x_update'].size(dim=1)),
                device=graph.device,
                dtype=graph.ndata['x_update'].dtype
            )
            for _, i in enumerate(ca_atom_node_idx):
                # Get the latest locations of Ca atoms along with their learned (displacement) coordinate updates
                ca_atom_x_orig = graph.ndata['x_orig'][i].unsqueeze(0)
                ca_atom_x_latest = graph.ndata['x_latest'][i].unsqueeze(0)
                ca_atom_x_disp_update = graph.ndata['x_update'][i].unsqueeze(0)

                # Get the indices of non-Ca atoms along with their learned (displacement) coordinate updates
                res_non_ca_atoms_idx = torch.where((graph.ndata['residue_number'] == graph.ndata['residue_number'][i])
                                                   & (graph.nodes() != i))
                res_atoms_x_disp_update = graph.ndata['x_update'][res_non_ca_atoms_idx]

                # Derive coordinates-wise skip connections for a residue, and use them to update its atoms' coordinates
                if self.x_connection_init > 0.0:
                    ca_x_upd = self.x_connection_init * ca_atom_x_orig + \
                               (1.0 - self.x_connection_init) * ca_atom_x_latest + \
                               ca_atom_x_disp_update
                else:
                    ca_x_upd = ca_atom_x_latest + ca_atom_x_disp_update
                x_latest[i] = ca_x_upd

                # Update non-Ca atoms using (displaced) Ca atom coords. and non-Ca atom (displaced) coord. updates
                x_latest[res_non_ca_atoms_idx] = x_latest[i] + res_atoms_x_disp_update

                # Install the latest atom coordinates as a Tensor
                graph.ndata['x_latest'] = x_latest
        else:
            # Construct new coordinates using each atom's positions as well as coordinates-wise skip connections
            if self.x_connection_init > 0.0:
                x_upd = self.x_connection_init * graph.ndata['x_orig'] + \
                        (1.0 - self.x_connection_init) * graph.ndata['x_latest'] + \
                        graph.ndata['x_update']
            else:
                x_upd = graph.ndata['x_latest'] + graph.ndata['x_update']
            graph.ndata['x_latest'] = x_upd

        # Log new node coordinates for manual inspection
        if self.verbose:
            logging.info('x_update: \sum_j (x_i - x_j) * \phi ^ x(m_{i->j}) = '
                         + str(torch.max(graph.ndata['x_update'])))
            logging.info('x_i new: x_latest : x_i + x_update = '
                         + str(torch.max(graph.ndata['x_latest'])))

    def update_node_feats(self, graph: dgl.DGLGraph):
        """
        In an E(n)-invariant manner, update the node representations in the input graph.

        Parameters
        ----------
        graph: dgl.DGLGraph
            DGL input graph.
        """
        # Compute new node representation updates via message-passing
        graph.update_all(fn.copy_edge('m_ij', 'm'), fn.mean('m', 'aggr_msg'))

        # Log new node representation message-passing updates for manual inspection
        if self.verbose:
            logging.info('aggr_msg: \sum_j m_{i->j} = ' + str(torch.max(graph.ndata['aggr_msg'])))

        # Assemble relevant features for node information update
        init_n_upd = torch.cat((self.nodes_norm(graph.ndata['feats']),
                                graph.ndata['aggr_msg'],
                                graph.ndata['aggr_mha_msg'],
                                graph.ndata['orig_feats']), dim=1)

        # Construct new node representations via node-wise skip connection
        skip_weight_h_is_learnable = self.learn_skip_weights and self.num_hidden_feats == self.num_output_feats
        skip_weight_h = torch.sigmoid(self.skip_weight_h) if skip_weight_h_is_learnable else self.skip_weight_h
        if self.use_deepspeed:
            nodes_mlp_out = deepspeed.checkpointing.checkpoint(self.nodes_mlp, init_n_upd)
        else:
            nodes_mlp_out = self.nodes_mlp(init_n_upd)
        if self.num_hidden_feats == self.num_output_feats:
            if 0.0 < self.skip_weight_h < 1.0:
                n_upd = skip_weight_h * nodes_mlp_out + (1.0 - skip_weight_h) * graph.ndata['feats']
            else:
                n_upd = graph.ndata['feats'] + nodes_mlp_out
        else:
            n_upd = nodes_mlp_out

        # Log new node representations for manual inspection
        if self.verbose:
            logging.info('nodes_mlp parameters:')
            for p in self.nodes_mlp.parameters():
                logging.info(p)
            logging.info('concat(h_i, aggr_msg, aggr_mha_msg) = ' + str(torch.max(init_n_upd)))
            logging.info('h_i new = h_i + MLP(h_i, aggr_msg, aggr_mha_msg) = ' + str(torch.max(n_upd)))
            logging.info('skip_weight_h: self.skip_weight_h = ' + str(self.skip_weight_h))

        # Install new node representations in the input graph
        graph.ndata['feats'] = n_upd

    def forward(self,
                graph: dgl.DGLGraph,
                h_feats: torch.Tensor,
                coords: torch.Tensor,
                orig_node_feats: torch.Tensor,
                orig_edge_feats: torch.Tensor,
                orig_coords: torch.Tensor,
                ca_atom_node_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network.

        Parameters
        ----------
        graph: DGLGraph
            DGL input graph.
        h_feats: torch.Tensor:
            Latest node representations.
        coords: torch.Tensor
            Latest node coordinates.
        orig_node_feats: torch.Tensor
            Original node representations.
        orig_edge_feats: torch.Tensor
            Original edge representations.
        orig_coords: torch.Tensor
            Original node coordinates.
        ca_atom_node_idx: torch.Tensor
            Node indices of Ca atoms.

        Returns
        ----------
        Tuple[torch.Tensor, torch.Tensor]
            Output node representations and coordinates.
        """
        # Avoid preserving artifacts from the current iteration of message-passing
        with graph.local_scope():
            # Store original and latest representations and coordinates in the input graph as temporary data
            graph.ndata['feats'] = h_feats
            graph.ndata['x_latest'] = coords
            graph.ndata['orig_feats'] = orig_node_feats
            graph.edata['orig_feats'] = orig_edge_feats
            graph.ndata['x_orig'] = orig_coords

            # Log for manual inspection node representations and coordinates at the entrance of the current layer
            if self.verbose:
                logging.info('feats: h_i at layer entrance = ' + str(torch.max(graph.ndata['feats'])))
                logging.info('x_latest: x_i at layer entrance = ' + str(torch.max(graph.ndata['x_latest'])))

            # Determine the latest version of each node's coordinates with which to update
            if self.update_pos_with_ca_atoms:
                # Convert atom coordinates into displacements w.r.t. their respective Ca atoms
                x_latest_disp = torch.zeros(
                    (graph.num_nodes(), graph.ndata['x_latest'].size(dim=1)),
                    device=graph.device,
                    dtype=graph.ndata['x_latest'].dtype
                )
                for _, i in enumerate(ca_atom_node_idx):
                    # Get the latest locations of Ca atoms
                    ca_atom_x = graph.ndata['x_latest'][i].unsqueeze(0)

                    # For a Ca atom, get indices of its corresponding non-Ca atoms along with their latest coordinates
                    res_atoms_idx = torch.where(graph.ndata['residue_number'] == graph.ndata['residue_number'][i])[0]
                    res_atoms_x = graph.ndata['x_latest'][res_atoms_idx]

                    # Derive each non-Ca atom's displacement w.r.t. its corresponding Ca atom
                    x_latest_disp[res_atoms_idx] = res_atoms_x - ca_atom_x

                # Install the latest atom coordinate displacements as a Tensor
                graph.ndata['x_latest_to_update'] = x_latest_disp
            else:
                # Directly use the latest version of each node's coordinates for the next round of coordinate updates
                graph.ndata['x_latest_to_update'] = graph.ndata['x_latest']

            # Derive all distance-based edge features required for E(n)-equivariant message-passing
            self.build_edge_dists(graph)

            # Log edge-wise relative positions and node-wise magnitudes for manual inspection
            if self.verbose:
                logging.info('x_rel: x_i - x_j = ' + str(torch.max(graph.edata['x_rel'])))
                logging.info('x_rel_mag: [exp(-||x_i - x_j||^2 / sigma) for sigma = 1.5 ** x, x = [0, 15]] = '
                             + str(torch.max(torch.max(graph.edata['x_rel_mag'], dim=0).values)))

            # Craft all edge messages (i.e., i->j => m_ij)
            graph.apply_edges(self.message_func)

            # Log edge-wise messages for manual inspection
            if self.verbose:
                logging.info('m_ij = m_{i->j} = phi^e(h_i, h_j, f_{i,j}, x_rel_mag) = '
                             + str(torch.max(graph.edata['m_ij'])))

            # Compute node-wise attention representations
            self.compute_node_attn(graph)

            # Update node coordinates, upon request
            if self.update_coords:
                self.update_node_coords(graph, ca_atom_node_idx)

            # Update node representations
            self.update_node_feats(graph)

            # Return updated node representations and coordinates
            return graph.ndata['feats'], graph.ndata['x_latest']

    def __repr__(self):
        return f'IEGR Layer {str(self.__dict__)}'


class IEGR(nn.Module):
    """Iterative E(n)-equivariant graph structure refinement as a DGL module.
    IEGR stands for an E(n)-equivariant iterative graph refiner. It is the
    equivalent of a series of (non-)linear layers in an MLP or convolution
    layers in a CNN.
    """

    def __init__(
            self,
            num_iegr_layers: int,
            num_iegr_iter: int,
            num_hidden_feats: int,
            num_edge_input_feats: int,
            num_attn_heads: int,
            num_atom_types: int,
            atom_emb_dim: int,
            use_surf_prox: bool,
            use_dihedral_angles: bool,
            use_edge_feats_in_iegr_layers: bool,
            use_dist_in_iegr_layers: bool,
            use_local_attention: bool,
            use_global_attention: bool,
            attn_depth: int,
            update_pos_with_ca_atoms: bool,
            knn: int,
            batched_input: bool,
            pooling: Optional[Literal['max', 'avg']] = None,
            pred_struct_qual=False,
            dropout=0.0,
            nonlin='lkyrelu',
            layer_norm='LN',
            layer_norm_coords='0',
            skip_weight_h=0.75,
            x_connection_init=0.0,
            leakyrelu_neg_slope=1e-2,
            shared_layers=False,
            norm_coord_updates=True,
            coord_norm_const=1.0,
            learn_skip_weights=False,
            use_fairscale=False,
            use_deepspeed=False,
            ca_only=False,
            verbose=False
    ):
        """Iterative E(n)-Equivariant Graph Structure Refinement Module
        Parameters
        ----------
        num_iegr_layers: int
            Number of layers within IEGR module.
        num_iegr_iter: int
            Number of times to restructure input graph topology based on IEGR module coordinate outputs.
        num_hidden_feats: int
            Hidden representation size.
        num_edge_input_feats: int
            Edge input representation size.
        num_attn_heads: int
            Number of attention heads to employ in multi-head attention computations.
        num_atom_types: int
            Number of types of atoms to support within an initial type embedding.
        atom_emb_dim: int
            Dimensionality to which to embed initial node (i.e., atom) types.
        use_surf_prox: bool
            Whether to concatenate atoms' surface proximities to their initial set of atom-type embeddings.
        use_dihedral_angles: bool
            Whether to concatenate atoms' dihedral angles to their initial set of atom-type embeddings.
        use_edge_feats_in_iegr_layers: bool
            Whether to initialize edge representations with precomputed features such as distances and bonds.
        use_dist_in_iegr_layers:
            Whether to include distance-based edge features in each iteration of message-passing.
        use_local_attention: bool
            Whether to compute node-wise attention scores node-locally within the IEGR model.
        use_global_attention: bool
            Whether to compute node-wise attention scores node-globally within the IEGR model.
        attn_depth: int
            How many global attention layers to apply within the IEGR model.
        update_pos_with_ca_atoms: bool
            Whether to update the position of non-Ca atoms using their Ca atom displacements.
        knn: int
            Number of nearest neighbors with which to construct new graph edges dynamically.
        batched_input: bool
            Whether the input graph is a batched graph, comprised of multiple subgraphs.
        pooling: Optional[Literal['max', 'avg']]
            Whether to apply 'max', 'avg', or None (i.e., no) graph pooling after all iterations of message-passing.
        pred_struct_qual: bool
            Whether to predict the quality of a predicted structure.
        dropout: float
            Rate of dropout to apply within IEGR layers.
        nonlin: str
            Which nonlinearity to apply within IEGR layers.
        layer_norm: str
            Which version of normalization to apply to each layer's learned representations.
        layer_norm_coords: str
            Which version of normalization to apply to each set of updated coordinates.
        skip_weight_h: float
            Skip weight to apply to node representations.
        x_connection_init: float
            Initial connection to node coordinates.
        leakyrelu_neg_slope: float
            Value for the LeakyReLU function's negative slope.
        shared_layers: bool
            Whether to use the same set of weights for all intermediate network layers.
        norm_coord_updates: bool
            Whether to normalize coordinate updates by their L2-norm plus a constant.
        coord_norm_const: float
            A normalizing constant for coordinate updates.
        learn_skip_weights: bool
            Whether to learn skip connection weights.
        use_fairscale: bool
            Whether to perform activation checkpointing with FairScale.
        use_deepspeed: bool
            Whether to perform activation checkpointing with DeepSpeed.
        ca_only: bool
            Whether the input graphs contains Ca atoms as nodes to represent each residue in the input protein.
        verbose: bool
            Whether to log node coordinates and representations for manual inspection.
        """
        super().__init__()

        # Establish values for each module state variable
        self.num_iegr_iter = num_iegr_iter
        self.use_surf_prox = use_surf_prox
        self.use_dihedral_angles = use_dihedral_angles
        self.use_edge_feats_in_iegr_layers = use_edge_feats_in_iegr_layers
        self.use_dist_in_iegr_layers = use_dist_in_iegr_layers
        self.update_pos_with_ca_atoms = update_pos_with_ca_atoms
        self.knn = knn
        self.batched_input = batched_input
        self.pooling = pooling
        self.pred_struct_qual = pred_struct_qual
        self.ca_only = ca_only
        self.verbose = verbose

        # Ascertain true input dimensionality of nodes
        num_node_input_feats = atom_emb_dim
        surf_prox_incr = 5 if ca_only else 1
        d_a_incr = 6 if use_dihedral_angles and ca_only else 0
        num_n_input_feats_incr = surf_prox_incr if self.use_surf_prox else 0
        num_n_input_feats_incr = num_n_input_feats_incr + d_a_incr if use_dihedral_angles else num_n_input_feats_incr
        num_node_input_feats += num_n_input_feats_incr

        # Adjust atom type embedding size according to whether or not the current configuration supports self-attention
        if not is_divisible_by(num_node_input_feats, num_attn_heads):
            for _ in range(atom_emb_dim - 1):
                atom_emb_dim -= 1
                is_div_by_heads = is_divisible_by(atom_emb_dim + num_n_input_feats_incr, num_attn_heads)
                is_power_of_two = is_divisible_by((atom_emb_dim + num_n_input_feats_incr) // num_attn_heads, 2)
                if is_div_by_heads and is_power_of_two:
                    break

            # Ensure shape compatibility for subsequent attention computations
            assert is_divisible_by(atom_emb_dim + num_n_input_feats_incr, num_attn_heads), \
                'Attention embedding dimensionality must be evenly divisible by the number of attention heads to use'

        # Adjust node input feature count
        num_node_input_feats = atom_emb_dim + num_n_input_feats_incr

        # Support ``num_atom_types`` types of atoms for embedding projection
        self.atom_emb_layer = nn.Embedding(num_embeddings=num_atom_types, embedding_dim=atom_emb_dim)

        # Assemble all layers for the current IEGR module and all its iterations
        inner_iegr_layers_list = []
        for iter_idx in range(num_iegr_iter):
            is_intermediate_iter = iter_idx != (num_iegr_iter - 1)

            # Add first layer
            if num_iegr_layers == 1:
                # Support IEGR message-passing with a single layer each iteration
                num_effective_output_feats = num_node_input_feats if is_intermediate_iter else num_hidden_feats
                inner_iegr_layers_list.append(nn.ModuleList([IEGRLayer(
                    num_node_input_feats=num_node_input_feats,
                    num_hidden_feats=num_node_input_feats,
                    num_edge_input_feats=num_edge_input_feats,
                    num_output_feats=num_effective_output_feats,
                    num_attn_heads=num_attn_heads,
                    use_dist_in_iegr_layers=use_dist_in_iegr_layers,
                    use_local_attention=use_local_attention,
                    use_global_attention=use_global_attention,
                    attn_depth=attn_depth,
                    update_pos_with_ca_atoms=update_pos_with_ca_atoms,
                    dropout=dropout,
                    nonlin=nonlin,
                    layer_norm=layer_norm,
                    layer_norm_coords=layer_norm_coords,
                    skip_weight_h=skip_weight_h,
                    x_connection_init=x_connection_init,
                    leakyrelu_neg_slope=leakyrelu_neg_slope,
                    update_coords=True,
                    norm_coord_updates=norm_coord_updates,
                    coord_norm_const=coord_norm_const,
                    learn_skip_weights=learn_skip_weights,
                    use_fairscale=use_fairscale,
                    use_deepspeed=use_deepspeed,
                    verbose=verbose
                )]))

                # Exit loop early since we have already added all requested layers
                break
            else:
                inner_iegr_layers = [IEGRLayer(
                    num_node_input_feats=num_node_input_feats,
                    num_hidden_feats=num_node_input_feats,
                    num_edge_input_feats=num_edge_input_feats,
                    num_output_feats=num_hidden_feats,
                    num_attn_heads=num_attn_heads,
                    use_dist_in_iegr_layers=use_dist_in_iegr_layers,
                    use_local_attention=use_local_attention,
                    use_global_attention=use_global_attention,
                    attn_depth=attn_depth,
                    update_pos_with_ca_atoms=update_pos_with_ca_atoms,
                    dropout=dropout,
                    nonlin=nonlin,
                    layer_norm=layer_norm,
                    layer_norm_coords=layer_norm_coords,
                    skip_weight_h=skip_weight_h,
                    x_connection_init=x_connection_init,
                    leakyrelu_neg_slope=leakyrelu_neg_slope,
                    update_coords=True,
                    norm_coord_updates=norm_coord_updates,
                    coord_norm_const=coord_norm_const,
                    learn_skip_weights=learn_skip_weights,
                    use_fairscale=use_fairscale,
                    use_deepspeed=use_deepspeed,
                    verbose=verbose
                )]

            # Aggregate subsequent layers
            if shared_layers:
                # Use the same set of weights for all intermediate layers
                shared_layer = IEGRLayer(
                    num_node_input_feats=num_node_input_feats,
                    num_hidden_feats=num_hidden_feats,
                    num_edge_input_feats=num_edge_input_feats,
                    num_output_feats=num_hidden_feats,
                    num_attn_heads=num_attn_heads,
                    use_dist_in_iegr_layers=use_dist_in_iegr_layers,
                    use_local_attention=use_local_attention,
                    use_global_attention=use_global_attention,
                    attn_depth=attn_depth,
                    update_pos_with_ca_atoms=update_pos_with_ca_atoms,
                    dropout=dropout,
                    nonlin=nonlin,
                    layer_norm=layer_norm,
                    layer_norm_coords=layer_norm_coords,
                    skip_weight_h=skip_weight_h,
                    x_connection_init=x_connection_init,
                    leakyrelu_neg_slope=leakyrelu_neg_slope,
                    update_coords=True,
                    norm_coord_updates=norm_coord_updates,
                    coord_norm_const=coord_norm_const,
                    learn_skip_weights=learn_skip_weights,
                    use_fairscale=use_fairscale,
                    use_deepspeed=use_deepspeed,
                    verbose=verbose
                )
                # Note: Final layer must contain a unique set of weights for the sake of node dimensionality reduction
                shared_update_node_coords = not pred_struct_qual
                num_effective_output_feats = num_node_input_feats if is_intermediate_iter else num_hidden_feats
                final_shared_layer = IEGRLayer(
                    num_node_input_feats=num_node_input_feats,
                    num_hidden_feats=num_hidden_feats,
                    num_edge_input_feats=num_edge_input_feats,
                    num_output_feats=num_effective_output_feats,
                    num_attn_heads=num_attn_heads,
                    use_dist_in_iegr_layers=use_dist_in_iegr_layers,
                    use_local_attention=use_local_attention,
                    use_global_attention=use_global_attention,
                    attn_depth=attn_depth,
                    update_pos_with_ca_atoms=update_pos_with_ca_atoms,
                    dropout=dropout,
                    nonlin=nonlin,
                    layer_norm=layer_norm,
                    layer_norm_coords=layer_norm_coords,
                    skip_weight_h=skip_weight_h,
                    x_connection_init=x_connection_init,
                    leakyrelu_neg_slope=leakyrelu_neg_slope,
                    update_coords=shared_update_node_coords,
                    norm_coord_updates=norm_coord_updates,
                    coord_norm_const=coord_norm_const,
                    learn_skip_weights=learn_skip_weights,
                    use_fairscale=use_fairscale,
                    use_deepspeed=use_deepspeed,
                    verbose=verbose
                )
                # Add each layer successively
                for _ in range(1, num_iegr_layers - 1):
                    inner_iegr_layers.append(shared_layer)
                inner_iegr_layers.append(final_shared_layer)
            else:
                # Use a unique set of weights for each intermediate layer as well as the final layer
                for layer_idx in range(1, num_iegr_layers):
                    is_final_layer = layer_idx == (num_iegr_layers - 1)
                    is_mid_iter_layer = is_final_layer and is_intermediate_iter
                    is_final_iter_final_layer = is_final_layer and not is_intermediate_iter and pred_struct_qual
                    num_effective_output_feats = num_node_input_feats if is_mid_iter_layer else num_hidden_feats
                    update_node_coords = False if is_final_iter_final_layer else True
                    inner_iegr_layers.append(
                        IEGRLayer(
                            num_node_input_feats=num_node_input_feats,
                            num_hidden_feats=num_hidden_feats,
                            num_edge_input_feats=num_edge_input_feats,
                            num_output_feats=num_effective_output_feats,
                            num_attn_heads=num_attn_heads,
                            use_dist_in_iegr_layers=use_dist_in_iegr_layers,
                            use_local_attention=use_local_attention,
                            use_global_attention=use_global_attention,
                            attn_depth=attn_depth,
                            update_pos_with_ca_atoms=update_pos_with_ca_atoms,
                            dropout=dropout,
                            nonlin=nonlin,
                            layer_norm=layer_norm,
                            layer_norm_coords=layer_norm_coords,
                            skip_weight_h=skip_weight_h,
                            x_connection_init=x_connection_init,
                            leakyrelu_neg_slope=leakyrelu_neg_slope,
                            # Skip updating coordinates in last layer to get graph structure quality predictions
                            update_coords=update_node_coords,
                            norm_coord_updates=norm_coord_updates,
                            coord_norm_const=coord_norm_const,
                            learn_skip_weights=learn_skip_weights,
                            use_fairscale=use_fairscale,
                            use_deepspeed=use_deepspeed,
                            verbose=verbose
                        )
                    )

            # Organize all layers in the current iteration as a single module
            inner_iegr_layers_list.append(nn.ModuleList(inner_iegr_layers))

        # Organize the modules across all iterations as a single module
        self.outer_iegr_module = nn.ModuleList(inner_iegr_layers_list)

        if self.pred_struct_qual:
            # Set up pooling module or per-node prediction layers for producing graph structure quality predictions
            n_out_feats = num_hidden_feats
            if self.pooling is not None:
                # Prepare to predict node-pooled structure quality scores
                self.pooling_module = GPooling(feat_type=0, pool=self.pooling)
                self.pooling_mlp = nn.Sequential(
                    nn.Linear(n_out_feats, n_out_feats),
                    nn.ReLU(),
                    nn.Linear(n_out_feats, 1)
                )
            else:
                # Prepare to predict per-node LDDT scores
                self.norm_lddt = nn.LayerNorm(n_out_feats)
                self.pred_lddt = nn.Linear(n_out_feats, 1)

    def forward(self, graph: dgl.DGLGraph, pdb_filepaths: List[str]) -> dgl.DGLGraph:
        """
        Forward pass of the network.

        Parameters
        ----------
        graph: dgl.DGLGraph
            DGL input graph.
        pdb_filepaths: List[str]
            List of paths to input PDB files.

        Returns
        ----------
        dgl.DGLGraph
            DGL output graph.
        """
        # Embed atom types using a lookup table
        atom_type = torch.where(graph.ndata['atom_type'] == 1.0)[1].long()  # (num_nodes, num_atom_types)
        h_feats = self.atom_emb_layer(atom_type)  # (num_nodes, node_dim=atom_emb_dim)

        # Log initial hidden node representations for manual inspection
        if self.verbose:
            logging.info('h_feats before any IEGR layers = ' + str(torch.max(h_feats)))

        # Add atom-wise surface proximities to the initial set of node features
        if self.use_surf_prox:
            h_feats = torch.cat([h_feats, graph.ndata['surf_prox']], dim=1)  # (num_nodes, node_dim=node_dim + 5)

        # Add atom-wise dihedral angles to the initial set of node features, for Ca atom graphs
        if self.use_dihedral_angles and self.ca_only:
            h_feats = torch.cat([h_feats, graph.ndata['dihedral_angles']], dim=1)  # (num_nodes, node_dim=node_dim + 6)

        # Log initial hidden node representations for manual inspection
        if self.verbose:
            logging.info('h_feats before any IEGR layers but after concatenating surf_prox and dihedral_angles = '
                         + str(torch.max(h_feats)), str(torch.norm(h_feats)))

        # Type coerce node features
        h_feats = h_feats.type(graph.ndata['x_pred'].dtype)

        # Cache original versions of node and edge representations
        orig_node_feats = h_feats.detach()
        orig_edge_feats = graph.edata['f'].detach() * self.use_edge_feats_in_iegr_layers  # (num_edges, num_edge_feats)

        # Cache current and original versions of node features and coordinates
        graph.ndata['f'] = h_feats
        orig_coords = graph.ndata['x_pred'].detach()  # (num_nodes, 3)

        # If using Ca atoms as residue centroids for coordinate updates, precompute the node indices of Ca atoms
        ca_atom_node_idx = torch.where(graph.ndata['is_ca_atom'])[0] if self.update_pos_with_ca_atoms else None

        # Perform each requested iteration of graph rewiring
        for iter_idx, inner_iegr_layers in enumerate(self.outer_iegr_module):
            # Retrieve the latest versions of node representations and coordinates for message-passing
            h_feats, coords = graph.ndata['f'], graph.ndata['x_pred']

            # Perform message-passing with a series of IEGR layers
            for layer in inner_iegr_layers:
                h_feats, coords = layer(
                    graph,
                    h_feats,
                    coords,
                    orig_node_feats,
                    orig_edge_feats,
                    orig_coords,
                    ca_atom_node_idx
                )

            # Log final node representations and coordinates for manual inspection
            if self.verbose:
                logging.info('h_feats after all IEGR layers = ' + str(torch.max(h_feats)))
                logging.info('coords after all IEGR layers = ' + str(torch.max(coords)))

            # Record learned node representations and coordinates inside the input graph
            graph.ndata['f'] = h_feats
            graph.ndata['x_pred'] = coords

            # In every iteration except the last, rewire the input graph using the latest predicted coordinates
            if iter_idx != (self.num_iegr_iter - 1):
                num_nodes = graph.batch_num_nodes()
                knn_lt_num_nodes = ((num_nodes > self.knn).float() == 1).nonzero().squeeze()
                # Copy new strong-link subgraphs corresponding to weak-link subgraphs where KNN < num_nodes
                if torch.numel(knn_lt_num_nodes) > 0:
                    # Obtain the latest relative positions and potentials to guide graph rewiring
                    update_relative_positions(graph)
                    update_potential_values(graph, r=graph.edata['r'])
                    # Rewire the input graph
                    graph = iegr_copy_without_weak_connections(graph,
                                                               graph_idx=knn_lt_num_nodes,
                                                               edges_per_node=self.knn,
                                                               batched_input=self.batched_input,
                                                               pdb_filepaths=pdb_filepaths)
                    # Update original edge features, since they may have changed after graph rewiring
                    orig_edge_feats = graph.edata['orig_feats']

        if self.pred_struct_qual:
            if self.pooling is not None:
                # Predict node-pooled structure quality scores
                pooled_feats = self.pooling_module({'0': graph.ndata['f']}, graph=graph)
                graph_output = self.pooling_mlp(pooled_feats).squeeze(-1)
                graph_output = graph_output.type(graph.ndata['f'].dtype)
            else:
                # Predict per-node LDDT scores
                pred_node_feats = self.pred_lddt(self.norm_lddt(graph.ndata['f']))
                pred_lddt = torch.clamp(pred_node_feats, 0.0, 1.0)
                graph_output = pred_lddt.type(graph.ndata['f'].dtype)
            graph.gdata = {'q': graph_output}  # Add quality prediction by installing it as a global graph attribute

        # Return updated graph
        return graph

    def __repr__(self):
        return f'IEGR {str(self.__dict__)}'


class IterativeEquivariantGraphRefinementModel(nn.Module):
    """Iterative E(n)-equivariant graph structure refinement as a DGL module."""

    def __init__(
            self,
            num_iegr_layers: int,
            num_iegr_iter: int,
            num_hidden_feats: int,
            num_edge_input_feats: int,
            num_attn_heads: int,
            num_atom_types: int,
            atom_emb_dim: int,
            use_surf_prox: bool,
            use_dihedral_angles: bool,
            use_edge_feats_in_iegr_layers: bool,
            use_dist_in_iegr_layers: bool,
            use_local_attention: bool,
            use_global_attention: bool,
            attn_depth: int,
            update_pos_with_ca_atoms: bool,
            knn: int,
            batched_input: bool,
            pooling: Optional[Literal['max', 'avg']] = None,
            pred_struct_qual=False,
            dropout=0.0,
            nonlin='lkyrelu',
            layer_norm='LN',
            layer_norm_coords='0',
            skip_weight_h=0.75,
            x_connection_init=0.0,
            leakyrelu_neg_slope=1e-2,
            shared_layers=False,
            norm_coord_updates=True,
            coord_norm_const=1.0,
            learn_skip_weights=False,
            use_fairscale=False,
            use_deepspeed=False,
            manually_init_weights=False,
            ca_only=False,
            verbose=False
    ):
        """Iterative E(n)-Equivariant Graph Structure Refinement Model

        Parameters
        ----------
        num_iegr_layers: int
            Number of layers within IEGR module.
        num_iegr_iter: int
            Number of times to restructure input graph topology based on IEGR module coordinate outputs.
        num_hidden_feats: int
            Hidden representation size.
        num_edge_input_feats: int
            Edge input representation size.
        num_attn_heads: int
            Number of attention heads to employ in multi-head attention computations.
        num_atom_types: int
            Number of types of atoms to support within an initial type embedding.
        atom_emb_dim: int
            Dimensionality to which to embed initial node (i.e., atom) types.
        use_surf_prox: bool
            Whether to concatenate atoms' surface proximities to their initial set of atom-type embeddings.
        use_dihedral_angles: bool
            Whether to concatenate atoms' dihedral angles to their initial set of atom-type embeddings.
        use_edge_feats_in_iegr_layers: bool
            Whether to initialize edge representations with precomputed features such as distances and bonds.
        use_dist_in_iegr_layers: bool
            Whether to include distance-based edge features in each iteration of message-passing.
        use_local_attention: bool
            Whether to compute node-wise attention scores node-locally within the IEGR model.
        use_global_attention: bool
            Whether to compute node-wise attention scores node-globally within the IEGR model.
        attn_depth: int
            How many global attention layers to apply within the IEGR model.
        update_pos_with_ca_atoms: bool
            Whether to update the position of non-Ca atoms using their Ca atom displacements.
        knn: int
            Number of nearest neighbors with which to construct new graph edges dynamically.
        batched_input: bool
            Whether the input graph is a batched graph, comprised of multiple subgraphs.
        pooling: Optional[Literal['max', 'avg']]
            Whether to apply 'max', 'avg', or None (i.e., no) graph pooling after all iterations of message-passing.
        pred_struct_qual: bool
            Whether to predict the quality of a predicted structure.
        dropout: float
            Rate of dropout to apply within IEGR layers.
        nonlin: str
            Which nonlinearity to apply within IEGR layers.
        layer_norm: str
            Which version of normalization to apply to each layer's learned representations.
        layer_norm_coords: str
            Which version of normalization to apply to each set of updated coordinates.
        skip_weight_h: float
            Skip weight to apply to node representations.
        x_connection_init: float
            Initial connection to node coordinates.
        leakyrelu_neg_slope: float
            Value for the LeakyReLU function's negative slope.
        shared_layers: bool
            Whether to use the same set of weights for all intermediate network layers.
        norm_coord_updates: bool
            Whether to normalize coordinate updates by their L2-norm plus a constant.
        coord_norm_const: float
            A normalizing constant for coordinate updates.
        learn_skip_weights: bool
            Whether to learn skip connection weights.
        use_fairscale: bool
            Whether to perform activation checkpointing with FairScale.
        use_deepspeed: bool
            Whether to perform activation checkpointing with DeepSpeed.
        manually_init_weights: bool
            Whether to manually initialize the weights and biases in the requested IEGR module.
        ca_only: bool
            Whether the input graphs contains Ca atoms as nodes to represent each residue in the input protein.
        verbose: bool
            Whether to log node coordinates and representations for manual inspection.
        """
        assert num_iegr_iter > 0, 'You must request at least one iteration for IEGR message-passing.'
        super().__init__()

        # Alert user to expected behavior of the IEGR module when providing certain arguments
        if dropout > 0.0:
            logging.info('IEGR Warning: Applying dropout > 0.0 will break E(n)-invariance w.r.t. node feature updates '
                         'and E(n)-equivariance w.r.t node coordinate updates.')
        if num_iegr_layers == 1:
            logging.info('IEGR Warning: Performing IEGR message-passing with a single layer each iteration '
                         'will induce unused coordinate MLP parameters in the final iteration.')

        self.iegr = IEGR(
            num_iegr_layers=num_iegr_layers,
            num_iegr_iter=num_iegr_iter,
            num_hidden_feats=num_hidden_feats,
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
            ca_only=ca_only,
            verbose=verbose
        )

        if manually_init_weights:
            self.reset_parameters()

    def reset_parameters(self):
        """Reset learnable parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p, gain=1.0)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, graph: dgl.DGLGraph, pdb_filepaths: List[str]) -> dgl.DGLGraph:
        """
        Forward pass of the network.

        Parameters
        ----------
        graph: dgl.DGLGraph
            DGL input graph.
        pdb_filepaths: List[str]
            List of paths to input PDB files.

        Returns
        ----------
        dgl.DGLGraph
            DGL output graph.
        """
        # Perform message-passing with an IEGR module
        graph = self.iegr(graph, pdb_filepaths)

        # Return updated batch graph
        return graph

    def __repr__(self):
        return f'IterativeEquivariantGraphRefinementModel {str(self.__dict__)}'
