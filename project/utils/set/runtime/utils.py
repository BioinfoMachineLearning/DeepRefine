# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

import argparse
import ctypes
import logging
import math
import os
import random
from functools import wraps
from typing import Union, List, Dict, Tuple

import dgl
import numpy as np
import torch
import pandas as pd
import torch.distributed as dist
from biopandas.pdb import PandasPdb
from torch import Tensor, FloatTensor
from numpy import linalg as LA

from project.utils.deeprefine_constants import COVALENT_RADIUS_TOLERANCE


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


def min_max_normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize provided tensor to have values be in range [0, 1]."""
    # Credit: https://discuss.pytorch.org/t/how-to-efficiently-normalize-a-batch-of-tensor-to-0-1/65122
    tensor -= tensor.min()
    tensor /= tensor.max()
    return tensor


def compute_euclidean_distance_matrix(points: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise Euclidean distances between every input point (i.e., row).

    Parameters
    ----------
    points: torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    euclidean_dists = torch.norm(points[:, None] - points, dim=2, p=2)
    return euclidean_dists


def find_intersection_indices_2D(first: torch.Tensor, second: torch.Tensor, device: str) -> torch.Tensor:
    """
    Return the row indices (in the first input Tensor) also present in the second input Tensor using NumPy.

    Parameters
    ----------
    first: torch.Tensor
        Primary input tensor.
    second:
        Secondary input tensor.
    device:
        Device on which to put index Tensor indicating rows in first that were also in second.

    Returns
    -------
    torch.Tensor
        A Tensor of row indices shared by both input Tensors.
    """
    first_np = first.cpu().numpy()
    second_np = second.cpu().numpy()
    first_np_view = first_np.view([('', first_np.dtype)] * first_np.shape[1])
    second_np_view = second_np.view([('', second_np.dtype)] * second_np.shape[1])
    intersect_idx = torch.tensor(np.intersect1d(first_np_view, second_np_view, return_indices=True)[1], device=device)
    return intersect_idx


def compute_bond_edge_feats(graph: dgl.DGLGraph, first_iter=False) -> torch.Tensor:
    """
    Compute edge feature indicating whether a covalent bond exists between a pair of atoms.

    Parameters
    ----------
    graph: dgl.DGLGraph
    first_iter: bool

    Returns
    -------
    torch.Tensor
    """
    # Compute all atom-atom Euclidean distances as a single distance matrix
    coords_distance_matrix = compute_euclidean_distance_matrix(graph.ndata['x_pred'])

    # Create a covalent 'distance' matrix by adding the radius array with its transpose
    orig_covalent_radius_distance_matrix = torch.add(
        graph.ndata['covalent_radius'].reshape(-1, 1),
        graph.ndata['covalent_radius'].reshape(1, -1)
    )

    # Add the covalent bond distance tolerance to the original covalent radius distance matrix
    covalent_radius_distance_matrix = (orig_covalent_radius_distance_matrix + COVALENT_RADIUS_TOLERANCE)

    # Sanity-check values in both distance matrices, only when first computing covalent bonds
    if first_iter:
        assert not torch.isnan(coords_distance_matrix).any(), 'No NaNs are allowed as coordinate pair distances'
        assert not torch.isnan(covalent_radius_distance_matrix).any(), 'No NaNs are allowed as covalent distances'

    # Threshold distance matrix to entries where Euclidean distance D > 0.4 and D < (covalent radius + tolerance)
    coords_distance_matrix[coords_distance_matrix <= 0.4] = torch.nan
    coords_distance_matrix[coords_distance_matrix >= covalent_radius_distance_matrix] = torch.nan
    covalent_bond_matrix = torch.nan_to_num(coords_distance_matrix)
    covalent_bond_matrix[covalent_bond_matrix > 0] = 1

    # Derive relevant covalent bonds based on the binary covalent bond matrix computed previously
    graph_edges_with_eids = graph.edges(form='all')
    graph_edges = torch.cat(
        (graph_edges_with_eids[0].reshape(-1, 1),
         graph_edges_with_eids[1].reshape(-1, 1)),
        dim=1
    )
    covalent_bond_edge_indices = covalent_bond_matrix.nonzero()
    combined_edges = torch.cat((graph_edges, covalent_bond_edge_indices))
    unique_edges, edge_counts = combined_edges.unique(dim=0, return_counts=True)
    covalently_bonded_edges = unique_edges[edge_counts > 1]

    # Find edges in the original graph for which a covalent bond was discovered
    covalently_bonded_eids = find_intersection_indices_2D(graph_edges, covalently_bonded_edges, graph_edges.device)

    # Craft new bond features based on the covalent bonds discovered above
    covalent_bond_feats = torch.zeros((len(graph_edges), 1), device=graph_edges.device)
    covalent_bond_feats[covalently_bonded_eids] = 1.0

    return covalent_bond_feats


def compute_chain_matches(edges: dgl.udf.EdgeBatch) -> Dict:
    """
    Compute edge feature indicating whether a pair of atoms belong to the same chain.

    Parameters
    ----------
    edges: dgl.udf.EdgeBatch

    Returns
    -------
    dict
    """
    # Collect edges' source and destination node IDs, as well as all nodes' chain IDs
    in_same_chain = torch.eq(edges.src['chain_id'], edges.dst['chain_id']).long().float()
    return {'in_same_chain': in_same_chain}


def get_r(graph: dgl.DGLGraph):
    """Compute inter-nodal distances"""
    cloned_rel_pos = torch.clone(graph.edata['rel_pos'])
    if graph.edata['rel_pos'].requires_grad:
        cloned_rel_pos.requires_grad_()
    return torch.sqrt(torch.sum(cloned_rel_pos ** 2, -1, keepdim=True))


def apply_potential_function(edge: dgl.udf.EdgeBatch):
    potential_parameters = torch.cosine_similarity(edge.src['surf_prox'], edge.dst['surf_prox']).float().reshape(-1, 1)
    return {'w': potential_function(edge.data['r'], potential_parameters)}


def potential_function(r: Tensor, potential_parameters: FloatTensor):
    x = r - potential_parameters - 1
    potential_function_global_min = -0.321919
    return x ** 4 - x ** 2 + 0.1 * x - potential_function_global_min


def potential_gradient(r: Tensor, potential_parameters: FloatTensor):
    x = r - potential_parameters - 1
    return 4 * x ** 3 - 2 * x + 0.1


def iset_copy_without_weak_connections(orig_graph: dgl.DGLGraph,
                                       graph_idx: Tensor,
                                       edges_per_node: int,
                                       batched_input: bool,
                                       pdb_filepaths: List[str]):
    """Make a copy of a graph, preserving only the edges_per_node-strongest incoming edges for each node.

    Args
        orig_graph: dgl.DGLGraph, the graph to copy
        graph_idx: Tensor, the indices of subgraphs (in a batched graph) to copy
        edges_per_node: int, the number of connections to preserve for each node in the output graph
        batched_input: bool, whether the input graph is a batched graph, comprised of multiple subgraphs
        pdb_filepaths: List[str], a collection of paths to input PDB files
    """
    # Cache the original batch number of nodes and edges
    batch_num_nodes, batch_num_edges = None, None
    if batched_input:
        batch_num_nodes = orig_graph.batch_num_nodes()
        batch_num_edges = orig_graph.batch_num_edges()

    # Iterate through all batched subgraphs to be rewired, since not all subgraphs have to have the same number of nodes
    graphs = dgl.unbatch(orig_graph) if batched_input else [orig_graph]
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
        rewired_graph.edata['f'] = torch.cat((
            rewired_graph.edata['edge_dist'],
            rewired_graph.edata['bond_type'],
            rewired_graph.edata['in_same_chain'],
            rewired_graph.edata['rel_geom_feats']
        ), dim=1)
        rewired_graph.edata['f'] = rewired_graph.edata['f'].type(edge_dtype)

        # Use newly-selected edges to update relative positions and potentials between nodes j and nodes i
        update_relative_positions(rewired_graph)
        update_potential_values(rewired_graph, r=rewired_graph.edata['r'])

        # Update subgraph in original list of batched graphs
        graphs[i] = rewired_graph

    # Re-batch graphs after in-place updating those that were rewired
    orig_graph = dgl.batch(graphs) if batched_input else graphs[0]

    # Restore the original batch number of nodes and edges
    if batched_input:
        orig_graph.set_batch_num_nodes(batch_num_nodes)
        orig_graph.set_batch_num_edges(batch_num_edges)

    return orig_graph


def update_absolute_positions(graph: dgl.DGLGraph, pos_updates: Tensor, absolute_position_key='x_pred'):
    """For each node in the graph, update the absolute position of the corresponding node.
     Write the updated absolute positions to the graph as node data."""
    graph.ndata[absolute_position_key] = graph.ndata[absolute_position_key] + pos_updates


def update_relative_positions(graph: dgl.DGLGraph, relative_position_key='rel_pos', absolute_position_key='x_pred'):
    """For each directed edge in the graph, calculate the relative position of the destination node with respect
    to the source node. Write the relative positions to the graph as edge data."""
    srcs, dsts = graph.all_edges()
    absolute_positions = graph.ndata[absolute_position_key]
    graph.edata[relative_position_key] = absolute_positions[dsts] - absolute_positions[srcs]
    graph.edata['r'] = graph.edata[relative_position_key].norm(dim=-1, keepdim=True)


def update_potential_values(graph: dgl.DGLGraph, r=None):
    """For each directed edge in the graph, compute the value of the potential between source and destination nodes.
    Write the computed potential values to the graph as edge data."""
    if r is None:
        r = get_r(graph)
    graph.edata['r'] = r
    graph.apply_edges(func=apply_potential_function)


def aggregate_residual(feats1, feats2, method: str):
    """ Add or concatenate two fiber features together. If degrees don't match, will use the ones of feats2. """
    if method in ['add', 'sum']:
        return {k: (v + feats1[k]) if k in feats1 else v for k, v in feats2.items()}
    elif method in ['cat', 'concat']:
        return {k: torch.cat([v, feats1[k]], dim=1) if k in feats1 else v for k, v in feats2.items()}
    else:
        raise ValueError('Method must be add/sum or cat/concat')


def degree_to_dim(degree: int) -> int:
    return 2 * degree + 1


def unfuse_features(features: Tensor, degrees: List[int]) -> Dict[str, Tensor]:
    return dict(zip(map(str, degrees), features.split([degree_to_dim(deg) for deg in degrees], dim=-1)))


def str2bool(v: Union[bool, str]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def to_cuda(x):
    """ Try to convert a Tensor, a collection of Tensors or a DGLGraph to CUDA """
    if isinstance(x, Tensor):
        return x.cuda(non_blocking=True)
    elif isinstance(x, tuple):
        return (to_cuda(v) for v in x)
    elif isinstance(x, list):
        return [to_cuda(v) for v in x]
    elif isinstance(x, dict):
        return {k: to_cuda(v) for k, v in x.items()}
    else:
        # DGLGraph or other objects
        return x.to(device=torch.cuda.current_device())


def get_local_rank() -> int:
    return int(os.environ.get('LOCAL_RANK', 0))


def init_distributed() -> bool:
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    distributed = world_size > 1
    if distributed:
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend, init_method='env://')
        if backend == 'nccl':
            torch.cuda.set_device(get_local_rank())
        else:
            logging.warning('Running on CPU only!')
        assert torch.distributed.is_initialized()
    return distributed


def increase_l2_fetch_granularity():
    # maximum fetch granularity of L2: 128 bytes
    _libcudart = ctypes.CDLL('libcudart.so')
    # set device limit on the current device
    # cudaLimitMaxL2FetchGranularity = 0x05
    pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
    _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
    _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
    assert pValue.contents.value == 128


def seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rank_zero_only(fn):
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if not dist.is_initialized() or dist.get_rank() == 0:
            return fn(*args, **kwargs)

    return wrapped_fn


def using_tensor_cores(amp: bool) -> bool:
    major_cc, minor_cc = torch.cuda.get_device_capability()
    return (amp and major_cc >= 7) or major_cc >= 8
