import logging
import math
from typing import List, Tuple

import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import torch
from biopandas.pdb import PandasPdb
from numpy import linalg as LA
from torch import nn

from project.utils.set.runtime.utils import compute_bond_edge_feats, compute_chain_matches


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code derived from egnn-pytorch (https://github.com/lucidrains/egnn-pytorch/blob/main/egnn_pytorch/utils.py):
# -------------------------------------------------------------------------------------------------------------------------------------
def rot_z(gamma):
    return torch.tensor([
        [torch.cos(gamma), -torch.sin(gamma), 0],
        [torch.sin(gamma), torch.cos(gamma), 0],
        [0, 0, 1]
    ], dtype=gamma.dtype)


def rot_y(beta):
    return torch.tensor([
        [torch.cos(beta), 0, torch.sin(beta)],
        [0, 1, 0],
        [-torch.sin(beta), 0, torch.cos(beta)]
    ], dtype=beta.dtype)


def rotate(alpha, beta, gamma):
    return rot_z(alpha) @ rot_y(beta) @ rot_z(gamma)


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from equidock_public (https://github.com/octavian-ganea/equidock_public/):
# -------------------------------------------------------------------------------------------------------------------------------------
class GraphNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, is_node=True):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.affine = affine
        self.is_node = is_node

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.num_features))
            self.beta = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def norm(self, x):
        mean = x.mean(dim=0, keepdim=True)
        var = x.std(dim=0, keepdim=True)
        return (x - mean) / (var + self.eps)

    def forward(self, g, h):
        graph_size = g.batch_num_nodes() if self.is_node else g.batch_num_edges()
        x_list = torch.split(h, graph_size.tolist())
        norm_list = []
        for x in x_list:
            norm_list.append(self.norm(x))
        norm_x = torch.cat(norm_list, 0)

        if self.affine:
            return self.gamma * norm_x + self.beta
        else:
            return norm_x


def G_fn(protein_coords: torch.Tensor, x: torch.Tensor, sigma: float):
    # protein_coords: (n,3) ,  x: (m,3), output: (m,)
    e = torch.exp(- torch.sum((protein_coords.view(1, -1, 3) - x.view(-1, 1, 3)) ** 2, dim=2) / sigma)  # (m, n)
    return - sigma * torch.log(1e-3 + e.sum(dim=1))


def compute_body_intersection_loss(graph_batch: dgl.DGLGraph,
                                   chain_combination_list: List[torch.Tensor],
                                   sigma: float,
                                   surface_ct: float):
    bil_loss = 0.0
    for chain_combinations, graph in zip(chain_combination_list, dgl.unbatch(graph_batch)):
        for chain_combination in chain_combinations:
            receptor_node_idx = torch.where(graph.ndata['chain_id'] == chain_combination[0])[0]
            ligand_node_idx = torch.where(graph.ndata['chain_id'] == chain_combination[1])[0]
            receptor_coords = graph.ndata['x_pred'][receptor_node_idx, :]
            ligand_coords = graph.ndata['x_pred'][ligand_node_idx, :]
            first_term = torch.clamp(surface_ct - G_fn(receptor_coords, ligand_coords, sigma), min=0)
            second_term = torch.clamp(surface_ct - G_fn(ligand_coords, receptor_coords, sigma), min=0)
            loss = torch.mean(first_term.type(receptor_coords.dtype)) + \
                   torch.mean(second_term.type(ligand_coords.dtype))
            bil_loss = bil_loss + loss
    return bil_loss


def get_nonlin(type, negative_slope):
    if type == 'swish':
        return nn.SiLU()
    elif type == 'tanh':
        return nn.Tanh()
    elif type == 'prelu':
        return nn.PReLU()
    elif type == 'elu':
        return nn.ELU()
    else:
        assert type == 'lkyrelu'
        return nn.LeakyReLU(negative_slope=negative_slope)


def get_layer_norm(layer_norm_type, dim, groups=8):
    if layer_norm_type == 'BN':
        return nn.BatchNorm1d(dim)
    elif layer_norm_type == 'LN':
        return nn.LayerNorm(dim)
    elif layer_norm_type == 'GN':
        return nn.GroupNorm(groups, dim)
    else:
        return nn.Identity()


def validate_residues(residues: List[pd.DataFrame], input_file: str) -> List[pd.DataFrame]:
    """Ensure each residue has a valid nitrogen (N), carbon-alpha (Ca), and carbon (C) atom."""
    residues_filtered = []
    for residue_idx, residue in enumerate(residues):
        df = residue[1]
        n_atom = df[df['atom_name'] == 'N']
        ca_atom = df[df['atom_name'] == 'CA']
        c_atom = df[df['atom_name'] == 'C']
        if n_atom.shape[0] == 1 and ca_atom.shape[0] == 1 and c_atom.shape[0] == 1:
            residues_filtered.append(residue)
        else:
            raise Exception(f'Residue {residue_idx} in {input_file} did not contain at least one valid backbone atom.')
    return residues_filtered


def build_geometric_vectors(residues: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive all residue-local geometric vector quantities."""
    n_i_list = []
    u_i_list = []
    v_i_list = []
    residue_representatives_loc_list = []

    for residue in residues:
        df = residue[1]

        n_atom = df[df['atom_name'] == 'N']
        ca_atom = df[df['atom_name'] == 'CA']
        c_atom = df[df['atom_name'] == 'C']

        if n_atom.shape[0] != 1 or ca_atom.shape[0] != 1 or c_atom.shape[0] != 1:
            logging.info(df.iloc[0, :])
            raise ValueError('In compute_rel_geom_feats(), no N/CA/C exists')

        n_loc = n_atom[['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32)
        ca_loc = ca_atom[['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32)
        c_loc = c_atom[['x_coord', 'y_coord', 'z_coord']].to_numpy().squeeze().astype(np.float32)

        u_i = (n_loc - ca_loc) / LA.norm(n_loc - ca_loc)
        t_i = (c_loc - ca_loc) / LA.norm(c_loc - ca_loc)
        n_i = np.cross(u_i, t_i) / LA.norm(np.cross(u_i, t_i))
        v_i = np.cross(n_i, u_i)
        assert (math.fabs(LA.norm(v_i) - 1.) < 1e-5), 'In compute_rel_geom_feats(), v_i norm was larger than 1'

        n_i_list.append(n_i)
        u_i_list.append(u_i)
        v_i_list.append(v_i)

        residue_representatives_loc_list.append(ca_loc)

    residue_representatives_loc_feat = np.stack(residue_representatives_loc_list, axis=0)  # (num_res, 3)
    n_i_feat = np.stack(n_i_list, axis=0)
    u_i_feat = np.stack(u_i_list, axis=0)
    v_i_feat = np.stack(v_i_list, axis=0)

    return residue_representatives_loc_feat, n_i_feat, u_i_feat, v_i_feat


def compute_rel_geom_feats(graph: dgl.DGLGraph,
                           atoms_df: pd.DataFrame,
                           input_file='Structure.pdb',
                           check_residues=False) -> torch.Tensor:
    """Calculate the relative geometric features for each residue's local coordinate system."""
    # Collect all residues along with their constituent atoms
    residues = list(atoms_df.groupby(['chain_id', 'residue_number', 'residue_name']))  # Note: Not the sequence order!

    # Validate the atom-wise composition of each residue
    if check_residues:
        residues = validate_residues(residues, input_file)

    # Derive zero-based node-wise residue numbers
    residue_numbers = graph.ndata['residue_number'] - 1

    # Derive all geometric vector quantities specific to each residue
    residue_representatives_loc_feat, n_i_feat, u_i_feat, v_i_feat = build_geometric_vectors(residues)

    # Loop over all edges of the graph, and build the various p_ij, q_ij, k_ij, t_ij pairs
    edges = graph.edges()
    edge_feat_ori_list = []
    for edge in zip(edges[0], edges[1]):
        # Get edge metadata
        src = edge[0]
        dst = edge[1]
        res_src = residue_numbers[src]
        res_dst = residue_numbers[dst]

        # Place n_i, u_i, v_i as lines in a 3 x 3 basis matrix
        basis_matrix = np.stack((n_i_feat[res_dst, :], u_i_feat[res_dst, :], v_i_feat[res_dst, :]), axis=0)
        p_ij = np.matmul(basis_matrix,
                         residue_representatives_loc_feat[res_src, :] -
                         residue_representatives_loc_feat[res_dst, :])
        q_ij = np.matmul(basis_matrix, n_i_feat[res_src, :])  # Shape: (3,)
        k_ij = np.matmul(basis_matrix, u_i_feat[res_src, :])
        t_ij = np.matmul(basis_matrix, v_i_feat[res_src, :])
        s_ij = np.concatenate((p_ij, q_ij, k_ij, t_ij), axis=0)  # Shape: (12,)
        edge_feat_ori_list.append(s_ij)

    # Return our resulting relative geometric features, local to each residue
    edge_feat_ori_feat = torch.from_numpy(np.stack(edge_feat_ori_list, axis=0)).float().to(graph.device)  # (n_edge, 12)
    return edge_feat_ori_feat


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from GraphTransformer (https://github.com/graphdeeplearning/graphtransformer/):
# -------------------------------------------------------------------------------------------------------------------------------------
def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        """Compute the dot product between source nodes' and destination nodes' representations."""
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}

    return func


def scaled_exp(field, scale_constant, clip_constant):
    def func(edges):
        """Clamp edge representations for softmax numerical stability."""
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-clip_constant, clip_constant))}

    return func


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from DeepInteract (https://github.com/BioinfoMachineLearning/DeepInteract):
# -------------------------------------------------------------------------------------------------------------------------------------
class LocalMultiHeadAttentionModule(nn.Module):
    """A graph-based local multi-head attention (MHA) mechanism as a DGL module."""

    def __init__(self, num_input_feats: int, num_output_feats: int, num_heads: int, use_bias: bool):
        super().__init__()

        # Define shared variables
        self.num_output_feats = num_output_feats
        self.num_heads = num_heads
        self.use_bias = use_bias

        # Define node features' query, key, and value tensors
        self.Q = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=use_bias)
        self.K = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=use_bias)
        self.V = nn.Linear(num_input_feats, self.num_output_feats * self.num_heads, bias=use_bias)

    def propagate_attention(self, graph: dgl.DGLGraph):
        """
        Compute all scores for MHA.

        Parameters
        ----------
        graph: dgl.DGLGraph
            DGL input graph.
        """
        # Compute attention scores
        graph.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'))  # , edges)

        # Scale, softmax, and clip attention scores
        graph.apply_edges(scaled_exp('score', np.sqrt(self.num_output_feats), 5.0))

        # Send weighted values to target nodes
        e_ids = graph.edges()
        graph.send_and_recv(e_ids, fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        graph.send_and_recv(e_ids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))

    def forward(self, graph: dgl.DGLGraph, node_feats: torch.Tensor) -> torch.Tensor:
        """
        Compute attention scores with a DGLGraph's node features.

        Parameters
        ----------
        graph: dgl.DGLGraph
            DGL input graph.
        node_feats: torch.Tensor
            Input node representations.

        Returns
        ----------
        torch.Tensor
            MHA node representations.
        """
        with graph.local_scope():
            node_feats_Q = self.Q(node_feats)
            node_feats_K = self.K(node_feats)
            node_feats_V = self.V(node_feats)

            # Reshape tensors into [num_nodes, num_heads, feat_dim] to get projections for MHA
            graph.ndata['Q_h'] = node_feats_Q.view(-1, self.num_heads, self.num_output_feats)
            graph.ndata['K_h'] = node_feats_K.view(-1, self.num_heads, self.num_output_feats)
            graph.ndata['V_h'] = node_feats_V.view(-1, self.num_heads, self.num_output_feats)

            # Disperse attention information
            self.propagate_attention(graph)

            # Normalize new node representations after applying MHA
            h_feats = graph.ndata['wV'] / graph.ndata['z']

            # Return MHA node representations
            return h_feats


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for DeepRefine (https://github.com/BioinfoMachineLearning/DeepRefine):
# -------------------------------------------------------------------------------------------------------------------------------------
def min_max_normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize provided tensor to have values be in range [0, 1]."""
    # Credit: https://discuss.pytorch.org/t/how-to-efficiently-normalize-a-batch-of-tensor-to-0-1/65122
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    return tensor


def iegr_copy_without_weak_connections(graph_to_rewire: dgl.DGLGraph,
                                       graph_idx: torch.Tensor,
                                       edges_per_node: int,
                                       batched_input: bool,
                                       pdb_filepaths: List[str]) -> dgl.DGLGraph:
    """
    Make a copy of a graph, preserving only the edges_per_node-strongest incoming edges for each node.

    Parameters
    ----------
    graph_to_rewire: dgl.DGLGraph
        Input graph to copy.
    graph_idx: torch.Tensor
        Indices of subgraphs (in a batched graph) to copy.
    edges_per_node: int
        Number of connections to preserve for each node in the rewired output graph.
    batched_input: bool
        Whether the input graph is a batched graph, comprised of multiple subgraphs.
    pdb_filepaths: List[str]
        List of paths to input PDB files.

    Returns
    -------
    dgl.DGLGraph
        Rewired DGL output graph.
    """
    # Cache the original batch number of nodes and edges
    batch_num_nodes, batch_num_edges = None, None
    if batched_input:
        batch_num_nodes = graph_to_rewire.batch_num_nodes()
        batch_num_edges = graph_to_rewire.batch_num_edges()

    # Iterate through all batched subgraphs to be rewired, since not all subgraphs have to have the same number of nodes
    graphs = dgl.unbatch(graph_to_rewire) if batched_input else [graph_to_rewire]
    for i in graph_idx.reshape(-1, 1):
        # Gather graph data
        graph = graphs[i.squeeze()]
        num_nodes = graph.num_nodes()
        w = graph.edata['w']
        srcs, dsts = graph.all_edges()

        # Load input PDB as a DataFrame
        atoms_df = PandasPdb().read_pdb(pdb_filepaths[i]).df['ATOM']

        # Sort srcs, dsts and w by dsts_node_id, then srcs_node_id
        w = w.view(num_nodes, edges_per_node)  # [dsts, srcs]
        dsts = dsts.view(num_nodes, edges_per_node)
        srcs = srcs.view(num_nodes, edges_per_node)

        # Sort edges according to their weight
        w, indices = torch.sort(w, descending=True)
        dsts = torch.gather(dsts, dim=-1, index=indices)
        srcs = torch.gather(srcs, dim=-1, index=indices)

        # Take the top-edges_per_node edges
        num_edges = graph.num_nodes() * edges_per_node
        dsts = dsts[:, :num_edges]
        srcs = srcs[:, :num_edges]

        # Reshape into 1D
        dsts = torch.reshape(dsts, (num_nodes * edges_per_node,)).detach()
        srcs = torch.reshape(srcs, (num_nodes * edges_per_node,)).detach()

        # Create new graph with fewer edges and fill with data
        rewired_graph = dgl.graph(data=(srcs, dsts), num_nodes=graph.num_nodes())
        # Fill in node data
        rewired_graph.ndata['f'] = graph.ndata['f']
        rewired_graph.ndata['chain_id'] = graph.ndata['chain_id']
        rewired_graph.ndata['x_pred'] = graph.ndata['x_pred']
        rewired_graph.ndata['x_true'] = graph.ndata['x_true']
        rewired_graph.ndata['labeled'] = graph.ndata['labeled']
        rewired_graph.ndata['interfacing'] = graph.ndata['interfacing']
        rewired_graph.ndata['covalent_radius'] = graph.ndata['covalent_radius']
        rewired_graph.ndata['residue_number'] = graph.ndata['residue_number']
        rewired_graph.ndata['is_ca_atom'] = graph.ndata['is_ca_atom']
        # Fill in edge data
        edge_dtype = graph.edata['edge_dist'].dtype
        edge_dist = torch.norm(rewired_graph.ndata['x_pred'][dsts] - rewired_graph.ndata['x_pred'][srcs], dim=1, p=2)
        rewired_graph.edata['edge_dist'] = min_max_normalize_tensor(edge_dist).reshape(-1, 1)
        rewired_graph.edata['edge_dist'] = rewired_graph.edata['edge_dist'].type(edge_dtype)
        rewired_graph.edata['bond_type'] = compute_bond_edge_feats(rewired_graph)
        rewired_graph.edata['bond_type'] = rewired_graph.edata['bond_type'].type(edge_dtype)
        rewired_graph.apply_edges(compute_chain_matches)  # Install edata['in_same_chain']
        rewired_graph.edata['in_same_chain'] = rewired_graph.edata['in_same_chain'].type(edge_dtype)
        rewired_graph.edata['rel_geom_feats'] = compute_rel_geom_feats(rewired_graph, atoms_df)
        rewired_graph.edata['rel_geom_feats'] = rewired_graph.edata['rel_geom_feats'].type(edge_dtype)
        # Combine individual edge features into a single edge feature tensor
        rewired_graph.edata['orig_feats'] = torch.cat((
            rewired_graph.edata['edge_dist'],
            rewired_graph.edata['bond_type'],
            rewired_graph.edata['in_same_chain'],
            rewired_graph.edata['rel_geom_feats']
        ), dim=1)
        rewired_graph.edata['orig_feats'] = rewired_graph.edata['orig_feats'].type(edge_dtype)

        # Update subgraph in original list of batched graphs
        graphs[i] = rewired_graph

    # Re-batch graphs after in-place updating those that were rewired
    graph_to_rewire = dgl.batch(graphs) if batched_input else graphs[0]

    # Restore the original batch number of nodes and edges
    if batched_input:
        graph_to_rewire.set_batch_num_nodes(batch_num_nodes)
        graph_to_rewire.set_batch_num_edges(batch_num_edges)

    return graph_to_rewire
