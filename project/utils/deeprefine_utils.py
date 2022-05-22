import argparse
import copy
import getpass
import logging
import math
import os
import pickle
import random
import re
import shutil
import signal
import subprocess
import sys
from functools import partial
from itertools import count, combinations, chain
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, Union, Iterator, Set
from typing import List, Optional, Tuple

import atom3.database as db
import dgl
import dgl.backend as dgl_F
import dill
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import pytorch_lightning as pl
import scipy.spatial as spatial
import torch
import torch.nn.functional as F
import torch_geometric
from Bio import SeqIO
from Bio.PDB import PDBParser, get_surface, internal_coords
from Bio.PDB.ResidueDepth import min_dist
from biopandas.pdb import PandasPdb
from dgl import DGLError
from dgl.base import dgl_warning
from numpy import linalg as LA
from pandas import Index
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin, DDPShardedPlugin, DeepSpeedPlugin
from scipy.spatial.distance import pdist, squareform
from scipy.special import softmax
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer, AdamW, Adam, SGD, NAdam, RAdam
from torch.optim.lr_scheduler import LambdaLR, CyclicLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts, StepLR
from torch_geometric.data import Batch, Data

from project.modules.segnn.o3_transform import O3Transform
from project.utils.deeprefine_constants import DEFAULT_BOND_STATE, \
    RESIDUE_ATOM_BOND_STATE, COVALENT_RADII, COVALENT_RADIUS_TOLERANCE, get_allowable_feats, RESI_THREE_TO_1, \
    DIHEDRAL_ANGLE_ID_TO_ATOM_NAMES
from project.utils.modeller.ca2all import ca2all
from project.utils.set.runtime.utils import update_relative_positions, \
    update_potential_values, find_intersection_indices_2D
from project.utils.segnn.utils import convert_dgl_graph_to_pyg_graph


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from DGL (https://github.com/DMCL/DGL/):
# -------------------------------------------------------------------------------------------------------------------------------------
def pairwise_squared_distance(x):
    """
    x : (n_samples, n_points, dims)
    return : (n_samples, n_points, n_points)
    """
    x2s = dgl_F.sum(x * x, -1, True)
    # assuming that __matmul__ is always implemented (true for PyTorch, MXNet and Chainer)
    return x2s + dgl_F.swapaxes(x2s, -1, -2) - 2 * x @ dgl_F.swapaxes(x, -1, -2)


def _knn_blas(x, k, dist='euclidean'):
    r"""Construct a graph from a set of points according to k-nearest-neighbor (KNN).

    This function first compute the distance matrix using BLAS matrix multiplication
    operation provided by backend frameworks. Then use topk algorithm to get
    k-nearest neighbors.

    Parameters
    ----------
    x : Tensor
        The point coordinates. It can be either on CPU or GPU.

        * If is 2D, ``x[i]`` corresponds to the i-th node in the KNN graph.

        * If is 3D, ``x[i]`` corresponds to the i-th KNN graph and
          ``x[i][j]`` corresponds to the j-th node in the i-th KNN graph.
    k : int
        The number of nearest neighbors per node.
    dist : str, optional
        The distance metric used to compute distance between points. It can be the following
        metrics:
        * 'euclidean': Use Euclidean distance (L2 norm) :math:`\sqrt{\sum_{i} (x_{i} - y_{i})^{2}}`.
        * 'cosine': Use cosine distance.
        (default: 'euclidean')
    """
    if dgl_F.ndim(x) == 2:
        x = dgl_F.unsqueeze(x, 0)
    n_samples, n_points, _ = dgl_F.shape(x)

    if k > n_points:
        dgl_warning("'k' should be less than or equal to the number of points in 'x'" \
                    "expect k <= {0}, got k = {1}, use k = {0}".format(n_points, k))
        k = n_points

    # if use cosine distance, normalize input points first
    # thus we can use euclidean distance to find knn equivalently.
    if dist == 'cosine':
        l2_norm = lambda v: dgl_F.sqrt(dgl_F.sum(v * v, dim=2, keepdims=True))
        x = x / (l2_norm(x) + 1e-5)

    ctx = dgl_F.context(x)
    dist = pairwise_squared_distance(x)
    k_indices = dgl_F.astype(dgl_F.argtopk(dist, k, 2, descending=False), dgl_F.int64)
    # index offset for each sample
    offset = dgl_F.arange(0, n_samples, ctx=ctx) * n_points
    offset = dgl_F.unsqueeze(offset, 1)
    src = dgl_F.reshape(k_indices, (n_samples, n_points * k))
    src = dgl_F.unsqueeze(src, 0) + offset
    dst = dgl_F.repeat(dgl_F.arange(0, n_points, ctx=ctx), k, dim=0)
    dst = dgl_F.unsqueeze(dst, 0) + offset
    return dgl_F.reshape(src, (-1,)), dgl_F.reshape(dst, (-1,))


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from Atom3D (https://github.com/drorlab/atom3d/blob/master/atom3d/util/graph.py):
# -------------------------------------------------------------------------------------------------------------------------------------
def prot_df_to_dgl_graph_with_feats(df: pd.DataFrame, feats: List, allowable_feats: List[List], knn: int) \
        -> Tuple[dgl.DGLGraph, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert protein in DataFrame representation to a graph compatible with DGL, where each node is an atom."""
    # Aggregate one-hot encodings of each atom's type to serve as the primary source of atom features
    atom_type_feat_vecs = [one_of_k_encoding_unk(feat, allowable_feats[0]) for feat in feats]
    atom_types = torch.FloatTensor(atom_type_feat_vecs)
    assert not torch.isnan(atom_types).any(), 'Atom types must be valid float values, not NaN'

    # Gather chain IDs to serve as an additional node feature
    chain_ids = torch.FloatTensor([c for c, (k, g) in enumerate(groupby(df['chain_id'].values.tolist()), 0) for _ in g])

    # Organize atom coordinates into a FloatTensor
    node_coords = torch.tensor(df[['x_coord', 'y_coord', 'z_coord']].values, dtype=torch.float32)

    # Define edges - KNN argument determines whether an atom-atom edge gets created in the resulting graph
    knn_graph = dgl.knn_graph(node_coords, knn)

    return knn_graph, node_coords, atom_types, chain_ids


def one_of_k_encoding_unk(feat, allowable_set):
    """Convert input to 1-hot encoding given a set of (or sets of) allowable values.
     Additionally, map inputs not in the allowable set to the last element."""
    if feat not in allowable_set:
        feat = allowable_set[-1]
    return list(map(lambda s: feat == s, allowable_set))


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from EquiDock (https://github.com/octavian-ganea/equidock_public/):
# -------------------------------------------------------------------------------------------------------------------------------------
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


def build_geometric_vectors(residues: List[pd.DataFrame]) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive all residue-local geometric vector quantities."""
    n_i_list = []
    u_i_list = []
    v_i_list = []
    ca_atom_loc_list = []

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

        ca_atom_loc_list.append(ca_loc)

    ca_atom_loc_list = np.stack(ca_atom_loc_list, axis=0)  # (num_res, 3)
    n_i_feat = np.stack(n_i_list, axis=0)
    u_i_feat = np.stack(u_i_list, axis=0)
    v_i_feat = np.stack(v_i_list, axis=0)

    return ca_atom_loc_list, n_i_feat, u_i_feat, v_i_feat


def compute_rel_geom_feats(graph: dgl.DGLGraph,
                           atoms_df: pd.DataFrame,
                           input_file: str,
                           check_residues=False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the relative geometric features for each residue's local coordinate system."""
    # Collect all residues along with their constituent atoms
    residues = list(atoms_df.groupby(['chain_id', 'residue_number', 'residue_name']))  # Note: Not the sequence order!

    # Validate the atom-wise composition of each residue
    if check_residues:
        residues = validate_residues(residues, input_file)

    # Derive zero-based node-wise residue numbers
    residue_numbers = graph.ndata['residue_number'] - 1

    # Derive all geometric vector quantities specific to each residue
    ca_atom_loc_list, n_i_feat, u_i_feat, v_i_feat = build_geometric_vectors(residues)

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
                         ca_atom_loc_list[res_src, :] -
                         ca_atom_loc_list[res_dst, :])
        q_ij = np.matmul(basis_matrix, n_i_feat[res_src, :])  # Shape: (3,)
        k_ij = np.matmul(basis_matrix, u_i_feat[res_src, :])
        t_ij = np.matmul(basis_matrix, v_i_feat[res_src, :])
        s_ij = np.concatenate((p_ij, q_ij, k_ij, t_ij), axis=0)  # Shape: (12,)
        edge_feat_ori_list.append(s_ij)

    # Return our resulting relative geometric features, local to each residue
    edge_feat_ori_feat = torch.from_numpy(np.stack(edge_feat_ori_list, axis=0)).float().to(graph.device)  # (n_edge, 12)
    return edge_feat_ori_feat


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from Graphein (https://www.github.com/a-r-j/graphein):
# -------------------------------------------------------------------------------------------------------------------------------------
def assign_bond_states_to_atom_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take a PandasPDB atom dataframe and assign bond states to each atom based on:
    Atomic Structures of all the Twenty Essential Amino Acids and a Tripeptide, with Bond Lengths as Sums of Atomic Covalent Radii
    Heyrovska, 2008

    :param df: Pandas PDB dataframe
    :type df: pd.DataFrame
    :return: Dataframe with added atom_bond_state column
    :rtype: pd.DataFrame
    """
    # Map atoms to their standard bond states
    naive_bond_states = pd.Series(df["atom_name"].map(DEFAULT_BOND_STATE))

    # Create series of bond states for non-standard states
    ss = (
        pd.DataFrame(RESIDUE_ATOM_BOND_STATE)
            .unstack()
            .rename_axis(("residue_name", "atom_name"))
            .rename("atom_bond_state")
    )

    # Map non-standard states to the dataframe based on residue name-atom name pairs
    df = df.join(ss, on=["residue_name", "atom_name"])

    # Fill all NaNs with standard states
    df = df.fillna(value={"atom_bond_state": naive_bond_states})

    # Note: For example, in the case of ligand input, replace remaining NaN values with the most common bond state
    if df["atom_bond_state"].isna().sum() > 1:
        most_common_bond_state = df["atom_bond_state"].value_counts().index[0]
        df = df.fillna(value={"atom_bond_state": most_common_bond_state})

    return df


def assign_covalent_radii_to_atom_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign covalent radius to each atom based on its bond state using values from:
    Atomic Structures of all the Twenty Essential Amino Acids and a Tripeptide, with Bond Lengths as Sums of Atomic Covalent Radii
    Heyrovska, 2008

    :param df: Pandas PDB dataframe with a bond_states_column
    :type df: pd.DataFrame
    :return: Pandas PDB dataframe with added covalent_radius column
    :rtype: pd.DataFrame
    """
    # Assign covalent radius to each atom
    df["covalent_radius"] = df["atom_bond_state"].map(COVALENT_RADII)
    return df


def compute_distmat(pdb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise euclidean distances between every atom.

    Design choice: passed in a DataFrame to enable easier testing on
    dummy data.

    :param pdb_df: pd.Dataframe containing protein structure. Must contain columns ["x_coord", "y_coord", "z_coord"]
    :type pdb_df: pd.DataFrame
    :return: pd.Dataframe of euclidean distance matrix
    :rtype: pd.DataFrame
    """
    eucl_dists = pdist(
        pdb_df[["x_coord", "y_coord", "z_coord"]], metric="euclidean"
    )
    eucl_dists = pd.DataFrame(squareform(eucl_dists))
    eucl_dists.index = pdb_df.index
    eucl_dists.columns = pdb_df.index

    return eucl_dists


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


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for DeepRefine (https://github.com/BioinfoMachineLearning/DeepRefine):
# -------------------------------------------------------------------------------------------------------------------------------------
def tidy_up_and_update_pdb_file(pdb_filepath: str):
    """Run 'pdb-tools' to tidy-up and replace an existing erroneous PDB file."""
    # Make a copy of the original PDB filepath to circumvent race conditions with 'pdb-tools'
    tmp_pdb_filepath = f'{pdb_filepath}.tmp'
    shutil.copyfile(pdb_filepath, tmp_pdb_filepath)

    # Clean temporary PDB file and then save its cleaned version as the original PDB file
    args = ['pdb_tidy', tmp_pdb_filepath]
    p1 = subprocess.Popen(args=args, stdout=subprocess.PIPE)
    with open(pdb_filepath, 'w') as outfile:
        _ = subprocess.run(args=['pdb_tidy'], stdin=p1.stdout, stdout=outfile)

    # Clean up from using temporary PDB file for tidying
    os.remove(tmp_pdb_filepath)


def make_pdb_from_coords(coords: Union[torch.Tensor, np.ndarray], filename: str,
                         metric: str, ca_only: bool, reconstruct_all_atoms=True) -> str:
    """Create a temporary PDB file that houses predicted atom-wise coordinates for a given protein."""
    # Use the input filename to establish the input and output directory and filename
    tmp_dir = os.path.split(filename)[0]
    pdb_path = filename
    tmp_filename = f'{os.path.splitext(os.path.basename(filename))[0]}_refined.pdb'
    if not reconstruct_all_atoms:
        tmp_filename = f'{os.path.splitext(tmp_filename)[0]}_refined_ca_only.pdb'
    tmp_filepath = os.path.join(tmp_dir, tmp_filename)

    # Load in e.g., AlphaFold's (original) predicted PDB structure
    pdb = PandasPdb().read_pdb(pdb_path)
    # Remove duplicate atoms, just as we do when constructing our cross-validation datasets
    subset = ['atom_name', 'residue_name', 'chain_id', 'residue_number']
    pdb.df['ATOM'] = pdb.df['ATOM'].drop_duplicates(subset=subset)
    # Reindex atom and residue numbers to start from 1
    pdb.df['ATOM'] = reindex_df_field_values(pdb.df['ATOM'], field_name='atom_number', start_index=1)
    pdb.df['ATOM'] = reindex_df_field_values(pdb.df['ATOM'], field_name='residue_number', start_index=1)

    # Replace the predicted PDB structure's coordinates with those predicted by our model
    coords = coords if type(coords) == np.ndarray else coords.detach().cpu().numpy()
    if ca_only:
        pdb.df['ATOM'].loc[pdb.df['ATOM']['atom_name'] == 'CA', ['x_coord', 'y_coord', 'z_coord']] = coords
        if reconstruct_all_atoms:
            # For modeller, save a temporary PDB file consisting of only Ca atoms
            pdb.df['ATOM'] = pdb.df['ATOM'][pdb.df['ATOM']['atom_name'] == 'CA']
            pdb.df['ATOM'] = pdb.df['ATOM'].reset_index(drop=True)
            pdb.df['ATOM'] = reindex_df_field_values(pdb.df['ATOM'], field_name='atom_number', start_index=1)
            pdb.df['ATOM'] = reindex_df_field_values(pdb.df['ATOM'], field_name='line_idx', start_index=1)
            pdb.to_pdb(tmp_filepath)
            # Derive an all-atom PDB
            ca2all(filename=tmp_filepath, output=tmp_filepath, iterations=1, verbose=False)
            # Load in latest PDB file
            pdb = PandasPdb().read_pdb(tmp_filepath)
    else:
        pdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']] = coords

    relabel_pdb_chains(pdb, filepath=tmp_filepath, metric=metric)

    return tmp_filepath


def dgl_psr_collate(protein_dicts: List[Dict]) -> Tuple[dgl.DGLGraph, List[str], List[torch.Tensor]]:
    """Assemble a list of protein dictionaries into a single DGLGraph batch."""
    batched_graph = dgl.batch([protein_dict['graph'] for protein_dict in protein_dicts])
    protein_dict_filepaths = [protein_dict['filepath'] for protein_dict in protein_dicts]
    chain_combination_list = [protein_dict['chain_combinations'] for protein_dict in protein_dicts]
    return batched_graph, protein_dict_filepaths, chain_combination_list


def pyg_psr_collate(protein_dicts: List[Dict]) -> Tuple[Batch, List[str], List[torch.Tensor]]:
    """Assemble a list of protein dictionaries into a single PyTorch Geometric Data batch."""
    batched_graph = torch_geometric.data.Batch.from_data_list([protein_dict['graph'] for protein_dict in protein_dicts])
    protein_dict_filepaths = [protein_dict['filepath'] for protein_dict in protein_dicts]
    chain_combination_list = [protein_dict['chain_combinations'] for protein_dict in protein_dicts]
    return batched_graph, protein_dict_filepaths, chain_combination_list


def dgl_collate(samples) -> Tuple[dgl.DGLGraph, torch.Tensor]:
    """Assemble a batched graph along with its respective labels (Note: For equivariant tests)."""
    graphs, y = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(y)


def calculate_and_store_dists_in_graph(graph: dgl.DGLGraph) -> dgl.DGLGraph:
    """Derive all inter-node distance features from a given batch of DGLGraphs."""
    graphs = dgl.unbatch(graph)
    for graph in graphs:
        edges = graph.edges()
        graph.edata['c'] = graph.ndata['x_pred'][edges[0]] - graph.ndata['x_pred'][edges[1]]
        graph.edata['r'] = torch.sum(graph.edata['c'] ** 2, 1).reshape(-1, 1)
    graph = dgl.batch(graphs)
    return graph


def get_rgraph(num_nodes: int,
               knn: int,
               self_loops: bool,
               dtype: torch.Type,
               using_segnn=False,
               o3_transform=O3Transform(3),
               ca_only=True) -> Union[dgl.DGLGraph, Data]:
    """Populate a randomly-generated KNN graph for the user."""
    pred_pos = torch.rand((num_nodes, 3))  # [num_nodes, 3]
    true_pos = torch.rand((num_nodes, 3))  # [num_nodes, 3]
    graph_knn = knn if self_loops else knn + 1
    G = dgl.knn_graph(pred_pos, graph_knn)

    """Remove self-loops from the randomly-generated graph"""
    if not self_loops:
        G = dgl.remove_self_loop(G)  # (e.g., By removing self-loops w/ k=11, we are effectively left w/ edges for k=10)

    """Encode node features in graph"""
    num_classes = 20 if ca_only else 38
    num_surf_prox_feats = 5 if ca_only else 1
    G.ndata['atom_type'] = F.one_hot(torch.arange(0, num_nodes) % 3, num_classes=num_classes)  # [num_nodes, 20 or 38]
    G.ndata['x_pred'] = pred_pos.type(dtype)  # [num_nodes, 3]
    G.ndata['x_true'] = true_pos.type(dtype)  # [num_nodes, 3]
    G.ndata['labeled'] = torch.randint(high=1, size=(num_nodes, 1))
    G.ndata['interfacing'] = torch.randint(high=1, size=(num_nodes, 1))
    G.ndata['covalent_radius'] = torch.rand((num_nodes, 1))
    G.ndata['chain_id'] = torch.randint(high=3, size=(num_nodes, 1))
    G.ndata['surf_prox'] = torch.rand((num_nodes, num_surf_prox_feats))  # [num_nodes, 5 or 1]
    G.ndata['residue_number'] = torch.IntTensor(
        list(chain(*[[i for _ in range(5)] for i in range(1, (num_nodes // 5) + 1)]))
    )  # [num_nodes, 1]
    G.ndata['is_ca_atom'] = torch.IntTensor([
        1 if i % 5 == 0 else 0 for i in range(G.ndata['residue_number'].size(dim=0))
    ])  # [num_nodes, 1]
    G.ndata['f'] = torch.cat((G.ndata['atom_type'],
                              G.ndata['surf_prox']), dim=1)
    if ca_only:
        G.ndata['dihedral_angles'] = torch.rand((num_nodes, 6))  # [num_nodes, 6]
        G.ndata['f'] = torch.cat((G.ndata['f'],
                                  G.ndata['dihedral_angles']), dim=1)

    """Encode edge features in graph"""
    num_edges = G.num_edges()
    G.edata['pos_enc'] = torch.rand((num_edges, 1))  # [num_edges, 1]
    G.edata['rel_pos'] = torch.rand((num_edges, 3))  # [num_edges, 3]
    G.edata['r'] = torch.rand((num_edges, 1))  # [num_edges, 1]
    G.edata['w'] = torch.rand((num_edges, 1))  # [num_edges, 1]
    G.edata['in_same_chain'] = torch.randint(high=2, size=(num_edges, 1))  # [num_edges, 1]
    G.edata['rel_geom_feats'] = torch.rand((num_edges, 12))  # [num_edges, 12]
    G.edata['f'] = torch.cat((G.edata['pos_enc'],
                              G.edata['in_same_chain'],
                              G.edata['rel_geom_feats']), dim=1)
    if not ca_only:
        G.edata['bond_type'] = torch.randint(high=1, size=(num_edges, 1))
        G.edata['f'] = torch.cat((G.edata['f'],
                                  G.edata['bond_type']), dim=1)

    if using_segnn:
        # Convert the randomly-generated KNN graph into its PyTorch Geometric equivalent
        G = convert_dgl_graph_to_pyg_graph(G, ca_only=ca_only)
        G = o3_transform(G)

    return G


def rotate_randomly(x, dtype=np.float32) -> torch.Tensor:
    """Apply a random rotation to the tensor given."""
    s = np.random.randn(3, 3)
    r, __ = np.linalg.qr(s)
    r = r.astype(dtype)
    return x @ r


def max_normalize_array(array: np.ndarray, max_value: float = None) -> np.ndarray:
    """Normalize values in provided array using its maximum value."""
    if array.size > 0:
        array = array / (array.max() if max_value is None else max_value)
    return array


def max_normalize_tensor(tensor: torch.Tensor, max_value: float = None) -> torch.Tensor:
    """Normalize values in provided tensor using its maximum value."""
    # Credit: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/distance.html
    if tensor.numel() > 0:
        tensor = tensor / (tensor.max() if max_value is None else max_value)
    return tensor


def construct_filenames_frame_txt_filenames(mode: str, root: str, root_postfix: str) -> Tuple[str, str, str]:
    """Build the file path of the requested filename DataFrame text file."""
    base_txt_filename = f'proteins' if mode == 'full' else f'proteins-{mode}'
    filenames_frame_txt_filename = base_txt_filename + '.txt'
    filenames_frame_txt_filepath = os.path.join(root, root_postfix, filenames_frame_txt_filename)
    return base_txt_filename, filenames_frame_txt_filename, filenames_frame_txt_filepath


def build_filenames_frame_error_message(dataset: str, task: str, filenames_frame_txt_filepath: str) -> str:
    """Assemble the standard error message for a corrupt or missing filenames DataFrame text file."""
    return f'Unable to {task} {dataset} filenames text file' \
           f' (i.e. {filenames_frame_txt_filepath}).' \
           f' Please make sure it is downloaded and not corrupted.'


def normalize_protein_features(protein_dict: Dict[str, Any], ca_only: bool, const: int) -> Dict[str, Tuple[Any]]:
    """Normalize the input features for a given protein dictionary using a dataset-global maximum constant."""
    # # Normalize node features
    # protein_dict['graph'].ndata['atom_type'] = protein_dict['graph'].ndata['atom_type']
    # protein_dict['graph'].ndata['surf_prox'] = protein_dict['graph'].ndata['surf_prox'] / const
    # if ca_only:
    #     dihedral_angles = protein_dict['graph'].ndata['dihedral_angles'] / const
    #     protein_dict['graph'].ndata['dihedral_angles'] = dihedral_angles
    #
    # # Rebuild node features
    # protein_dict['graph'].ndata['f'] = torch.cat((
    #     protein_dict['graph'].ndata['atom_type'],
    #     protein_dict['graph'].ndata['surf_prox']
    # ), dim=1)
    # if ca_only:
    #     protein_dict['graph'].ndata['f'] = torch.cat((
    #         protein_dict['graph'].ndata['f'],
    #         protein_dict['graph'].ndata['dihedral_angles']
    #     ), dim=1)

    # # Normalize edge features
    # protein_dict['graph'].edata['pos_enc'] = protein_dict['graph'].edata['pos_enc'] / const
    # protein_dict['graph'].edata['in_same_chain'] = protein_dict['graph'].edata['in_same_chain']
    # protein_dict['graph'].edata['rel_geom_feats'] = protein_dict['graph'].edata['rel_geom_feats'] / const
    # if not ca_only:
    #     bond_types = protein_dict['graph'].edata['bond_type']
    #     protein_dict['graph'].edata['bond_type'] = bond_types

    # Rebuild edge features
    # protein_dict['graph'].edata['f'] = torch.cat((
    #     protein_dict['graph'].edata['pos_enc'],
    #     protein_dict['graph'].edata['in_same_chain'],
    #     protein_dict['graph'].edata['rel_geom_feats']
    # ), dim=1)
    # if not ca_only:
    #     protein_dict['graph'].edata['f'] = torch.cat((
    #         protein_dict['graph'].edata['f'],
    #         protein_dict['graph'].edata['bond_type']
    #     ), dim=1)

    # Return feature-normalized protein dictionary
    return protein_dict


def retrieve_indices_in_orig_df(merged_df: pd.DataFrame, orig_df: pd.DataFrame) -> pd.Index:
    """Get the original indices corresponding to the merged DataFrame's rows."""
    on_cols = ['atom_name', 'chain_id', 'residue_number']
    unique_merged_cols_array = merged_df.drop(labels=merged_df.columns.difference(on_cols), axis=1)
    matched_df = orig_df.reset_index().merge(right=unique_merged_cols_array, how='inner', on=on_cols).set_index('index')
    return matched_df.index


def get_shared_df_coords(left_df: pd.DataFrame, right_df: pd.DataFrame, merge_order: str, return_np_coords=False) \
        -> Union[Tuple[Union[torch.Tensor, np.ndarray], pd.Index], pd.Index]:
    """Reconcile two dataframes to get the 3D coordinates corresponding to each atom shared by both input DataFrames."""
    on_cols = ['atom_name', 'chain_id', 'residue_number']
    if merge_order == 'left':
        merged_df = left_df.merge(right_df, how='inner', on=on_cols)
        orig_left_df_indices = retrieve_indices_in_orig_df(merged_df, left_df)  # Retrieve orig. IDs for merged rows
        if return_np_coords:
            true_coords = merged_df[['x_coord_y', 'y_coord_y', 'z_coord_y']].to_numpy()
        else:
            true_coords = torch.tensor(merged_df[['x_coord_y', 'y_coord_y', 'z_coord_y']].values, dtype=torch.float32)
        return true_coords, orig_left_df_indices
    elif merge_order == 'right':
        merged_df = right_df.merge(left_df, how='inner', on=on_cols)
        orig_right_df_indices = retrieve_indices_in_orig_df(merged_df, right_df)  # Retrieve orig. IDs for merged rows
        return orig_right_df_indices
    else:
        raise NotImplementedError(f'Merge order {merge_order} is not currently supported')


def get_interfacing_atom_indices(true_atom_df: pd.DataFrame,
                                 orig_true_df_indices: pd.Index,
                                 idt: float,
                                 return_partners=False) -> Tuple[Index, Optional[Dict[Any, Any]]]:
    """
    For a given true PDB DataFrame, return the indices of atoms found in any interface between at least two chains.

    Parameters
    ----------
    true_atom_df: pd.DataFrame
    orig_true_df_indices: pd.Index
    idt: float
    return_partners: bool

    Returns
    -------
    Tuple[Index, Union[Dict[int, Set[Any]], Index]]
    """
    # Filter down to only the atoms contained within the predicted PDB structure, and clean up these atoms' chain IDs
    atoms = true_atom_df.loc[orig_true_df_indices, :]
    atoms.reset_index(drop=True,
                      inplace=True)  # Make selected atoms' indices start from zero for following computations
    atoms = reindex_df_field_values(atoms, field_name='chain_id', start_index=0)
    unique_chain_ids = atoms['chain_id'].unique().tolist()
    unique_chain_ids.sort()
    # Pre-compute all pairwise atom distances
    all_atom_coords = atoms[['x_coord', 'y_coord', 'z_coord']].to_numpy()
    atom_coord_tree = spatial.cKDTree(all_atom_coords)
    # Find the index of all inter-chain atoms satisfying the specified interface distance (IDT) threshold
    inter_chain_atom_index_mapping = {
        i: atoms[atoms['chain_id'] != chain_id].index.values.tolist()
        for i, chain_id in enumerate(unique_chain_ids)
    }
    interfacing_atom_indices = set()
    interfacing_atom_mapping = {} if return_partners else None
    for i, atom in enumerate(atoms.itertuples(index=False)):
        inter_chain_atom_indices = inter_chain_atom_index_mapping[atom.chain_id]
        atom_neighbor_indices = atom_coord_tree.query_ball_point([atom.x_coord, atom.y_coord, atom.z_coord], idt)
        inter_chain_atom_neighbor_indices = set(atom_neighbor_indices) & set(inter_chain_atom_indices)
        interfacing_atom_indices = interfacing_atom_indices.union(inter_chain_atom_neighbor_indices)
        if return_partners:
            interfacing_atom_mapping[i] = list(inter_chain_atom_neighbor_indices)
            interfacing_atom_mapping[i].sort()
    # Sort collected atom indices and return them as a Pandas Index
    interfacing_atom_indices = list(interfacing_atom_indices)
    interfacing_atom_indices.sort()
    interfacing_atom_indices = pd.Index(interfacing_atom_indices)
    return interfacing_atom_indices, interfacing_atom_mapping


def map_ca_atom_ids_to_all_atom_ids(orig_node_ids: torch.Tensor, all_atom_id_mapping: Dict[int, int]) -> torch.Tensor:
    mapped_ca_atom_ids = torch.tensor([all_atom_id_mapping[n_id.item()] for n_id in orig_node_ids])
    return mapped_ca_atom_ids


def create_interfacing_atoms_visualization_pdb(input_file: str, pred_pdb: PandasPdb, true_pdb: PandasPdb,
                                               pred_atom_df: pd.DataFrame, true_atom_df: pd.DataFrame,
                                               orig_pred_df_indices: pd.Index, orig_true_df_indices: pd.Index,
                                               interfacing_atom_indices: pd.Index):
    """
    Reindex chain IDs in a given PDB file to visualize all non-interfacing atoms as being in the first of two chains.

    Parameters
    ----------
    input_file: str
    pred_pdb: PandasPdb
    true_pdb: PandasPdb
    pred_atom_df: pd.DataFrame
    true_atom_df: pd.DataFrame
    orig_pred_df_indices: pd.Index
    orig_true_df_indices: pd.Index
    interfacing_atom_indices: pd.Index
    """
    viz_pred = True
    viz_filename_parts = input_file.split(os.sep)[-2:]
    tmp_dir = os.path.join('/dev', 'shm', getpass.getuser(), 'tmp', 'Viz', viz_filename_parts[0])
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_viz_filepath = os.path.join(tmp_dir, os.path.splitext(viz_filename_parts[1])[0] + '.pdb')
    if viz_pred:
        viz_atoms_df = pred_atom_df.loc[orig_pred_df_indices, :]
    else:
        viz_atoms_df = true_atom_df.loc[orig_true_df_indices, :]
    one_hot_viz_chain_ids = np.zeros((len(viz_atoms_df), 1), dtype=np.int32)
    one_hot_viz_chain_ids[interfacing_atom_indices] = 1
    viz_atoms_df['chain_id'] = one_hot_viz_chain_ids
    if viz_pred:
        pred_pdb.df['ATOM'] = viz_atoms_df
        pred_pdb.to_pdb(tmp_viz_filepath)
    else:
        true_pdb.df['ATOM'] = viz_atoms_df
        true_pdb.to_pdb(tmp_viz_filepath)


def compute_mean_vector_ratio_norm(graph: dgl.DGLGraph, pdb_filepath: str, verbose=False) -> torch.Tensor:
    """
    For all nodes (i.e., atoms), find their respective mean vector ratio norms w.r.t their neighbors.
    Such scalars can describe how close to the chain surface a given atom is, which, as such,
    can serve as a proxy for an atom's "buriedness" (an important descriptor for tasks involving multiple
    biological entities interacting with one another).
    Parameters
    ----------
    graph: dgl.DGLGraph
    pdb_filepath: str
    verbose: bool
    Returns
    -------
    torch.Tensor
    """
    mean_norm_list = []
    for node_id in graph.nodes():
        # Find intra-chain source nodes directed towards the current node
        nbr_ids = graph.predecessors(node_id)
        node_chain_id = graph.ndata['chain_id'][node_id, :]
        nbr_chain_ids = graph.ndata['chain_id'][nbr_ids, :]
        nbr_ids = nbr_ids[(nbr_chain_ids == node_chain_id).nonzero()[:, 0]]

        # Upon finding that an atom has no intra-chain neighbors (via KNN), assume the atom is on its chain's surface
        if len(nbr_ids) == 0:
            if verbose:
                logging.info(f'Node {node_id} had no intra-chain KNN neighbors;'
                             f' Assuming unknown surface proximity temporarily...')
            mean_norm_list.append(np.array([np.nan for _ in range(5)], dtype=np.float32))
            continue

        # Find inter-residue source nodes directed towards the current node
        node_res_num = graph.ndata['residue_number'].reshape(-1, 1)[node_id, :]
        nbr_res_nums = graph.ndata['residue_number'].reshape(-1, 1)[nbr_ids, :]
        nbr_ids = nbr_ids[(nbr_res_nums != node_res_num).nonzero()[:, 0]]

        # Upon finding that an atom has no inter-residue neighbors (via KNN), assume atom is not on its chain's surface
        if len(nbr_ids) == 0:
            if verbose:
                logging.info(f'Node {node_id} had no intra-chain, inter-residue KNN neighbors;'
                             f' Assuming minimal surface proximity temporarily...')
            mean_norm_list.append(np.array([0.0 for _ in range(5)], dtype=np.float32))
            continue

        # Compute the distances between an atom and its predecessors
        node_x = graph.ndata['x_pred'][node_id, :]
        nbr_x = graph.ndata['x_pred'][nbr_ids, :]
        node_nbr_diff_vecs = node_x - nbr_x  # (num_nbr, 3)
        node_nbr_dist = np.linalg.norm(node_nbr_diff_vecs, axis=1)

        # Compute radial basis functions (i.e., edge weights) describing the distance between an atom and its neighbors
        sigma = np.array([1.0, 2.0, 5.0, 10.0, 30.0]).reshape(-1, 1)
        weights = softmax(- node_nbr_dist.reshape(1, -1) ** 2 / sigma, axis=1)  # (num_sigma, num_nbr)
        assert (1 - 1e-2) < weights[0].sum() < 1.01, f'Atom-atom weights in {pdb_filepath} must sum to 1.0'

        # Construct the mean vector ratio norm for a given atom
        mean_vec = weights.dot(node_nbr_diff_vecs)  # (num_sigma, 3)
        denominator = weights.dot(node_nbr_dist)  # (num_sigma,)
        mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (num_sigma,)
        mean_norm_list.append(mean_vec_ratio_norm)

    # Compile all collected norms together as a 2D NumPy array
    mean_norms = np.array(mean_norm_list).astype(np.float32)

    # Impute mean norms for atoms that did not container either any intra-chain neighbors or any inter-residue neighbors
    intra_chain_row_idx = np.where(np.isnan(mean_norms).all(axis=1))[0]
    inter_residue_row_idx = np.where(~mean_norms.any(axis=1))[0]

    filled_row_idx = np.ones(mean_norms.shape[0], dtype=bool)
    filled_row_idx[intra_chain_row_idx] = False
    filled_row_idx[inter_residue_row_idx] = False

    norms_for_imputation = mean_norms[filled_row_idx, 4]
    min_mean_norm, max_mean_norm = np.min(norms_for_imputation), np.max(norms_for_imputation)
    min_mean_norm, max_mean_norm = max(min_mean_norm, 0.0), min(max_mean_norm, 1.0)

    if len(intra_chain_row_idx) > 0:
        mean_norms[intra_chain_row_idx, :] = np.array([
            [max_mean_norm for _ in range(5)] for _ in range(len(intra_chain_row_idx))
        ])
    if len(inter_residue_row_idx) > 0:
        mean_norms[inter_residue_row_idx, :] = np.array([
            [min_mean_norm for _ in range(5)] for _ in range(len(inter_residue_row_idx))
        ])

    return torch.from_numpy(mean_norms)


def compute_surface_proximities(graph: dgl.DGLGraph, pdb_filepath: str, ca_only: bool) -> torch.Tensor:
    """
    For all nodes (i.e., atoms), find their proximity to their respective solvent-accessible surface area.
    Such a scalar can describe how close to the chain surface a given atom is, which, as such,
    can serve as a measure for an atom's "buriedness" (an important descriptor for tasks involving
    multiple biological entities interacting with one another).

    Parameters
    ----------
    graph: dgl.DGLGraph
    pdb_filepath: str
    ca_only: bool

    Returns
    -------
    torch.Tensor
    """
    # If requested, compute residue buriedness using average vector ratio norms for each residue
    if ca_only:
        return compute_mean_vector_ratio_norm(graph, pdb_filepath)

    # Extract from MSMS the vectors describing each chain's surface
    parser = PDBParser()
    pdb_code = db.get_pdb_code(pdb_filepath)
    structure = parser.get_structure(pdb_code, pdb_filepath)
    num_struct_models = len(structure)
    assert num_struct_models == 1, f'Input PDB {pdb_filepath} must consist of only a single model'
    model = structure[0]
    chain_id_map = {k: c for c, (k, g) in enumerate(groupby([chain.id for chain in model]), 0) for _ in g}
    surfaces = {chain_id_map[chain.id]: get_surface(chain) for chain in model}

    # Derive the depth of each atom in the given graph
    atom_depth_map = {v: [] for v in chain_id_map.values()}
    for node_id in graph.nodes():
        node_x = tuple(graph.ndata['x_pred'][node_id, :].numpy())
        chain_id = graph.ndata['chain_id'][node_id].item()
        atom_depth_map[chain_id].append(min_dist(node_x, surfaces[chain_id]))

    # Normalize each chain's atom depths in batch yet separately from all other chain batches
    surf_prox_list = []
    for k in atom_depth_map.keys():
        # Compile all atom depths together as a 2D NumPy array
        atom_depths = np.array(atom_depth_map[k]).astype(np.float32).reshape(-1, 1)
        # Normalize atom depths using a specified maximum value
        atom_depths_scaled = max_normalize_array(atom_depths, max_value=100.0)
        # Take the elementwise complement of atom depths to get the normalized surface proximity of each atom
        surf_prox = np.ones_like(atom_depths_scaled) - atom_depths_scaled
        # Ensure minimum surface proximity is zero, given that atoms may be more than 100 Angstrom away from a surface
        clipped_surf_prox = surf_prox.clip(0.0, surf_prox.max())
        surf_prox_list.append(clipped_surf_prox)

    surf_prox = np.concatenate(surf_prox_list)
    return torch.from_numpy(surf_prox)


def determine_is_ca_atom(graph: dgl.DGLGraph, atom_df: pd.DataFrame) -> torch.Tensor:
    """
    Determine, for each atom, whether it is a carbon-alpha (Ca) atom or not.

    Parameters
    ----------
    graph: dgl.DGLGraph
    atom_df: pd.DataFrame

    Returns
    -------
    torch.Tensor
    """
    is_ca_atom = torch.zeros((graph.num_nodes(), 1), dtype=torch.bool)
    is_ca_atom_df_indices = atom_df[atom_df['atom_name'] == 'CA'].index.values
    is_ca_atom[is_ca_atom_df_indices] = True
    return is_ca_atom.int()


def derive_dihedral_angles(pdb_filepath: str) -> pd.DataFrame:
    """Find all dihedral angles for the residues in an input PDB file."""
    # Increase BioPython's MaxPeptideBond to capture all dihedral angles
    internal_coords.IC_Chain.MaxPeptideBond = 100.0

    # Parse our input PDB structure
    parser = PDBParser(PERMISSIVE=1, QUIET=1)
    structure = parser.get_structure('', pdb_filepath)

    # Generate internal coordinates for the input structure
    structure.atom_to_internal_coordinates()

    # Collect backbone dihedral angles for each residue
    num_residues = int(sum([len(record.seq) for record in SeqIO.parse(pdb_filepath, "pdb-atom")]))
    dihedral_angles_dict = {i + 1: np.zeros(3) for i in range(num_residues)}

    # Note: Each structure has at least two chains
    residue_num = 1
    latest_residue_num = 0
    structure_chains = list(structure.get_chains())
    for structure_chain in structure_chains:
        ic_chain = structure_chain.internal_coord
        for key in ic_chain.dihedra.keys():
            dihedral = ic_chain.dihedra[key]

            dihedral_id_tokens = dihedral.id.split(':')
            residue_num = int(dihedral_id_tokens[1].split('_')[0]) + latest_residue_num
            dihedral_angle_atoms = [s.split('_')[-1] for s in dihedral_id_tokens]

            if dihedral_angle_atoms in DIHEDRAL_ANGLE_ID_TO_ATOM_NAMES.values():
                angle_idx = [k for k, v in DIHEDRAL_ANGLE_ID_TO_ATOM_NAMES.items() if v == dihedral_angle_atoms][0]
                dihedral_angles_dict[residue_num][angle_idx] = ic_chain.dihedra[key].angle

        # Track the latest residue number in the most-recent chain
        latest_residue_num = residue_num

    # Assemble and return resulting dihedral angles
    dihedral_angles = np.stack(list(dihedral_angles_dict.values()), axis=0)
    assert dihedral_angles.any(), 'Must have found at least one valid dihedral angle for the input protein'
    return pd.DataFrame(dihedral_angles, columns=['phi', 'psi', 'omega'])


def compute_dihedral_angles(pdb_filepath: str) -> torch.Tensor:
    """
    Derive the phi and psi backbone dihedral angles for each residue in the input PDB file.

    Parameters
    ----------
    pdb_filepath: str

    Returns
    -------
    torch.Tensor
    """
    angles_df = derive_dihedral_angles(pdb_filepath)

    phi_angles = angles_df['phi'].values
    psi_angles = angles_df['psi'].values
    omega_angles = angles_df['omega'].values

    cos_encoded_angles = np.cos(phi_angles), np.cos(psi_angles), np.cos(omega_angles)
    sin_encoded_angles = np.sin(phi_angles), np.sin(psi_angles), np.sin(omega_angles)
    dihedral_angles = torch.from_numpy(np.stack((*cos_encoded_angles, *sin_encoded_angles), axis=1))
    return dihedral_angles


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
    covalent_bond_feats = torch.zeros((len(graph_edges), 1))
    covalent_bond_feats[covalently_bonded_eids] = 1

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


def convert_dfs_to_dgl_graph(pred_pdb: PandasPdb, true_pdb: PandasPdb, input_file: str,
                             atom_selection_type: str, knn: int, idt: float, output_iviz: bool,
                             tmp_file_dataset='PSR') -> \
        Tuple[dgl.DGLGraph, pd.Index]:
    r""" Transform a given set of predicted and true atom DataFrames into a corresponding DGL graph.

    Parameters
    ----------
    pred_pdb: PandasPdb
    true_pdb: PandasPdb
    input_file: str
    atom_selection_type: str
    knn: int
    idt: float
    output_iviz: bool
    tmp_file_dataset: str

    Returns
    -------
    :class:`typing.Tuple[:class:`dgl.DGLGraph`, :class:`np.ndarray`]`
        Index 1. Graph structure, feature tensors for each node and edge.
            ...     predicted_node_coords = graph.ndata['x_pred']
            ...     true_node_coords = graph.ndata['x_true']
        - ``ndata['atom_type']``: one-hot type of each node
        - ``ndata['x_pred']:`` predicted Cartesian coordinate tensors of the nodes
        - ``ndata['x_true']:`` true Cartesian coordinate tensors of the nodes
        - ``ndata['labeled']``: one-hot indication of whether a node has a ground-truth coordinates label available
        - ``ndata['interfacing']``: one-hot indication of whether node lies within 'idt' Angstrom of an inter-chain node
        - ``ndata['covalent_radius']``: scalar descriptor of the hypothesized covalent radius of a node
        - ``ndata['chain_id']``: integer descriptor of the unique ID of the chain to which a node belongs (e.g., 3)
        - ``ndata['surf_prox']``: scalar descriptor of how close to the surface of a molecular chain a node is
        - ``edata['rel_pos']``: vector descriptor of the relative position of each destination node to its source node
        - ``edata['r']``: scalar descriptor of the normalized distance of each destination node to its source node
        - ``edata['w']``: scalar descriptor of the hypothetical weight between each destination node and source node
        - ``edata['edge_dist']``: zero-to-one normalized Euclidean distance between pairs of atoms
        - ``edata['bond_type']``: one-hot description of whether a hypothetical covalent bond exists between a node pair
        - ``edata['in_same_chain']``: one-hot description of whether a node pair belongs to the same molecular chain
        Index 2. DataFrame indices corresponding to atoms shared by both the predicted structure and true structure.
    """
    """Build the input graph"""
    # Determine atom selection type requested
    ca_only = atom_selection_type == 'ca_atom'

    # Ascertain atoms in DataFrames
    pred_atom_df = pred_pdb.df['ATOM']
    true_atom_df = true_pdb.df['ATOM']
    orig_pred_atom_df = copy.deepcopy(pred_atom_df)

    # Reindex atom and residue numbers
    pred_atom_df = pred_atom_df[pred_atom_df['atom_name'] == 'CA'].reset_index(drop=True) if ca_only else pred_atom_df
    pred_atom_df = reindex_df_field_values(pred_atom_df, field_name='atom_number', start_index=1)
    pred_atom_df = reindex_df_field_values(pred_atom_df, field_name='residue_number', start_index=1)
    true_atom_df = true_atom_df[true_atom_df['atom_name'] == 'CA'].reset_index(drop=True) if ca_only else true_atom_df
    true_atom_df = reindex_df_field_values(true_atom_df, field_name='atom_number', start_index=1)
    true_atom_df = reindex_df_field_values(true_atom_df, field_name='residue_number', start_index=1)

    # Set up temporary filepaths
    tmp_dir = os.path.join('/dev', 'shm', getpass.getuser(), 'tmp')
    tmp_pred_filepath = os.path.join(tmp_dir, tmp_file_dataset, 'pred', f'{db.get_pdb_code(input_file)}.pdb')
    os.makedirs(os.path.join(tmp_dir, tmp_file_dataset, 'pred'), exist_ok=True)
    tm_pred_pdb = copy.deepcopy(pred_pdb)
    tm_pred_pdb.df['ATOM'] = copy.deepcopy(orig_pred_atom_df)
    relabel_pdb_chains(tm_pred_pdb, filepath=tmp_pred_filepath, metric='tm' if ca_only else 'mm')

    # Get shared coordinates and standard atom indices
    shared_true_coords, orig_pred_df_indices = get_shared_df_coords(pred_atom_df, true_atom_df, merge_order='left')

    # Sanity-check the construction of each protein graph a priori
    assert len(shared_true_coords) > 0, f'Protein graph to be constructed, {input_file}, must have shared coordinates'

    # Get interfacing atom indices
    orig_true_df_indices = get_shared_df_coords(pred_atom_df, true_atom_df, merge_order='right')
    interfacing_atom_indices, _ = get_interfacing_atom_indices(true_atom_df, orig_true_df_indices, idt)

    if not ca_only:
        # Assign bond states to the dataframe, and then map these bond states to covalent radii for each atom
        # Note: For example, in the case of ligand input, set the 'residue_name' value for all atoms to 'XXX'
        pred_atom_df = assign_bond_states_to_atom_dataframe(pred_atom_df)
        pred_atom_df = assign_covalent_radii_to_atom_dataframe(pred_atom_df)

    # Construct KNN graph
    feats = pred_atom_df['residue_name' if ca_only else 'atom_name']
    feats = feats.map(RESI_THREE_TO_1).tolist() if ca_only else feats
    graph, node_coords, atoms_types, chain_ids = prot_df_to_dgl_graph_with_feats(
        pred_atom_df,  # All predicted atoms when constructing the initial graph
        feats,  # Which atom selection type to use for node featurization
        get_allowable_feats(ca_only),  # Which feature values are expected
        knn + 1  # Since we do not allow self-loops, we must include an extra nearest neighbor to be removed thereafter
    )

    # Remove self-loops in graph
    graph = dgl.remove_self_loop(graph)  # By removing self-loops w/ k=21, we are effectively left w/ edges for k=20

    # Manually add isolated nodes (i.e. those with no connected edges) to the graph
    if len(atoms_types) > graph.number_of_nodes():
        num_of_isolated_nodes = len(atoms_types) - graph.number_of_nodes()
        raise ValueError(f'{num_of_isolated_nodes} isolated node(s) detected in {input_file}')

    """Encode node features and labels in graph"""
    # Include one-hot features indicating each atom's type
    graph.ndata['atom_type'] = atoms_types  # [num_nodes, num_node_feats=21 if ca_only is True, 38 otherwise]
    # Cartesian coordinates for each atom
    graph.ndata['x_pred'] = node_coords  # [num_nodes, 3]
    # True coordinates for each atom - Only populated for atoms present in both input DataFrames - All others are zeroed
    true_coords = torch.zeros((len(pred_atom_df), 3))
    true_coords[orig_pred_df_indices] = shared_true_coords
    graph.ndata['x_true'] = true_coords  # [num_nodes, 3]
    # One-hot ID representation of the atoms for which true coordinates were available in the corresponding exp. struct.
    labeled = torch.zeros((graph.number_of_nodes(), 1), dtype=torch.int32)
    labeled[orig_pred_df_indices] = 1
    graph.ndata['labeled'] = labeled  # [num_nodes, 1]
    # One-hot ID representation of the atoms present in any interface between at least two chains
    interfacing = torch.zeros((graph.number_of_nodes(), 1), dtype=torch.int32)
    interfacing[interfacing_atom_indices] = 1
    graph.ndata['interfacing'] = interfacing  # [num_nodes, 1]
    # Single scalar describing the covalent radius of each atom
    null_cov_radius = torch.zeros_like(graph.ndata['interfacing'])
    covalent_radius = null_cov_radius if ca_only else torch.FloatTensor(pred_atom_df['covalent_radius']).reshape(-1, 1)
    graph.ndata['covalent_radius'] = covalent_radius
    # Single value indicating to which chain an atom belongs
    graph.ndata['chain_id'] = chain_ids.reshape(-1, 1)  # [num_nodes, 1]
    # Integers describing to which residue an atom belongs
    graph.ndata['residue_number'] = torch.IntTensor(pred_atom_df['residue_number'].tolist())  # [num_nodes, 1]
    # Scalars describing each atom's proximity to the surface of its chain
    graph.ndata['surf_prox'] = compute_surface_proximities(graph, input_file, ca_only)  # [num_nodes, 1]
    # Binary integers (i.e., 0 or 1) describing whether an atom is a carbon-alpha (Ca) atom
    graph.ndata['is_ca_atom'] = determine_is_ca_atom(graph, pred_atom_df.reset_index(drop=True))  # [num_nodes, 1]
    # Scalars describing the cosine and sine-activated phi, psi, and omega backbone dihedral angles for each residue
    dihedral_angles = compute_dihedral_angles(tmp_pred_filepath) if ca_only else torch.zeros_like(graph.ndata['x_pred'])
    graph.ndata['dihedral_angles'] = dihedral_angles  # [num_nodes, 6]

    """Encode edge features in graph"""
    # Positional encoding for each edge (used for sequentially-ordered inputs like proteins)
    graph.edata['pos_enc'] = torch.sin((graph.edges()[0] - graph.edges()[1]).float()).reshape(-1, 1)  # [num_edges, 1]
    update_relative_positions(graph)  # Relative positions of nodes j to nodes i - [num_edges, 1]
    update_potential_values(graph, r=graph.edata['r'])  # Weights from nodes j and nodes i - [num_edges, 1]
    bond_types = torch.zeros_like(graph.edata['r']) if ca_only else compute_bond_edge_feats(graph, first_iter=True)
    graph.edata['bond_type'] = bond_types.reshape(-1, 1)  # [num_edges, 1]
    graph.apply_edges(compute_chain_matches)  # Install edata['in_same_chain'] - [num_edges, 1]
    rel_geom_feats = compute_rel_geom_feats(graph, orig_pred_atom_df, input_file=input_file)
    graph.edata['rel_geom_feats'] = rel_geom_feats  # [num_edges, 12]

    # Create a modified version of the current predicted PDB to visualize interfacing atoms via new binary chain IDs
    if output_iviz and not ca_only:
        pred_atom_df.drop(columns=['atom_bond_state', 'covalent_radius'], inplace=True)
        create_interfacing_atoms_visualization_pdb(
            input_file, pred_pdb, true_pdb,
            pred_atom_df, true_atom_df,
            orig_pred_df_indices, orig_true_df_indices,
            interfacing_atom_indices
        )

    """Remove temporary file(s)"""
    os.remove(tmp_pred_filepath)

    """Ensure no input feature values are invalid (e.g., NaN)"""
    nans_in_ndata = torch.isnan(graph.ndata['x_true']).any() \
                    or torch.isnan(graph.ndata['x_pred']).any() \
                    or torch.isnan(graph.ndata['atom_type']).any() \
                    or torch.isnan(graph.ndata['labeled']).any() \
                    or torch.isnan(graph.ndata['interfacing']).any() \
                    or torch.isnan(graph.ndata['covalent_radius']).any() \
                    or torch.isnan(graph.ndata['chain_id']).any() \
                    or torch.isnan(graph.ndata['residue_number']).any() \
                    or torch.isnan(graph.ndata['surf_prox']).any() \
                    or torch.isnan(graph.ndata['is_ca_atom']).any() \
                    or torch.isnan(graph.ndata['dihedral_angles']).any()
    nans_in_edata = torch.isnan(graph.edata['pos_enc']).any() \
                    or torch.isnan(graph.edata['rel_pos']).any() \
                    or torch.isnan(graph.edata['r']).any() \
                    or torch.isnan(graph.edata['bond_type']).any() \
                    or torch.isnan(graph.edata['in_same_chain']).any() \
                    or torch.isnan(graph.edata['rel_geom_feats']).any()
    assert not (nans_in_ndata or nans_in_edata), 'There must be no invalid (i.e., NaN) values in the graph features'

    # Return our resulting graph and predicted PDB DataFrame indices
    return graph, orig_pred_df_indices


def process_pdb_into_graph(input_filepath: str,
                           atom_selection_type: str,
                           knn: int,
                           idt: float) -> dgl.DGLGraph:
    r""" Transform a given set of predicted and true atom DataFrames into a corresponding DGL graph.

    Parameters
    ----------
    input_filepath: str
    atom_selection_type: str
    knn: int
    idt: float

    Returns
    -------
    :class:`typing.Tuple[:class:`dgl.DGLGraph`, :class:`np.ndarray`]`
        Index 1. Graph structure, feature tensors for each node and edge.
            ...     predicted_node_coords = graph.ndata['x_pred']
            ...     true_node_coords = graph.ndata['x_true']
        - ``ndata['atom_type']``: one-hot type of each node
        - ``ndata['x_pred']:`` predicted Cartesian coordinate tensors of the nodes
        - ``ndata['x_true']:`` true Cartesian coordinate tensors of the nodes
        - ``ndata['labeled']``: one-hot indication of whether a node has a ground-truth coordinates label available
        - ``ndata['interfacing']``: one-hot indication of whether node lies within 'idt' Angstrom of an inter-chain node
        - ``ndata['covalent_radius']``: scalar descriptor of the hypothesized covalent radius of a node
        - ``ndata['chain_id']``: integer descriptor of the unique ID of the chain to which a node belongs (e.g., 3)
        - ``ndata['surf_prox']``: scalar descriptor of how close to the surface of a molecular chain a node is
        - ``edata['rel_pos']``: vector descriptor of the relative position of each destination node to its source node
        - ``edata['r']``: scalar descriptor of the normalized distance of each destination node to its source node
        - ``edata['w']``: scalar descriptor of the hypothetical weight between each destination node and source node
        - ``edata['edge_dist']``: zero-to-one normalized Euclidean distance between pairs of atoms
        - ``edata['bond_type']``: one-hot description of whether a hypothetical covalent bond exists between a node pair
        - ``edata['in_same_chain']``: one-hot description of whether a node pair belongs to the same molecular chain
        Index 2. DataFrame indices corresponding to atoms shared by both the predicted structure and true structure.
    """
    """Build the input graph"""
    # Determine atom selection type requested
    ca_only = atom_selection_type == 'ca_atom'

    # Ascertain atoms in DataFrames
    input_pdb = PandasPdb().read_pdb(input_filepath)
    input_atom_df = input_pdb.df['ATOM']
    orig_pred_atom_df = copy.deepcopy(input_atom_df)

    # Reindex atom and residue numbers
    if ca_only:
        input_atom_df = input_atom_df[input_atom_df['atom_name'] == 'CA'].reset_index(drop=True)
    input_atom_df = reindex_df_field_values(input_atom_df, field_name='atom_number', start_index=1)
    input_atom_df = reindex_df_field_values(input_atom_df, field_name='residue_number', start_index=1)

    # Get interfacing atom indices
    orig_pred_df_indices = get_shared_df_coords(input_atom_df, input_atom_df, merge_order='right')
    interfacing_atom_indices, _ = get_interfacing_atom_indices(input_atom_df, orig_pred_df_indices, idt)

    if not ca_only:
        # Assign bond states to the dataframe, and then map these bond states to covalent radii for each atom
        # Note: For example, in the case of ligand input, set the 'residue_name' value for all atoms to 'XXX'
        input_atom_df = assign_bond_states_to_atom_dataframe(input_atom_df)
        input_atom_df = assign_covalent_radii_to_atom_dataframe(input_atom_df)

    # Construct KNN graph
    feats = input_atom_df['residue_name' if ca_only else 'atom_name']
    feats = feats.map(RESI_THREE_TO_1).tolist() if ca_only else feats
    try:
        graph, node_coords, atoms_types, chain_ids = prot_df_to_dgl_graph_with_feats(
            input_atom_df,  # All predicted atoms when constructing the initial graph
            feats,  # Which atom selection type to use for node featurization
            get_allowable_feats(ca_only),  # Which feature values are expected
            knn + 1  # Since we do not allow self-loops, we must include extra nearest neighbor to be removed thereafter
        )
    except DGLError:
        raise DGLError(f'In process_pdb_into_graph(), found an empty point set for {input_filepath}')

    # Remove self-loops in graph
    graph = dgl.remove_self_loop(graph)  # By removing self-loops w/ k=21, we are effectively left w/ edges for k=20

    # Manually add isolated nodes (i.e. those with no connected edges) to the graph
    if len(atoms_types) > graph.number_of_nodes():
        num_of_isolated_nodes = len(atoms_types) - graph.number_of_nodes()
        raise ValueError(f'{num_of_isolated_nodes} isolated node(s) detected in {input_filepath}')

    """Encode node features and labels in graph"""
    # Include one-hot features indicating each atom's type
    graph.ndata['atom_type'] = atoms_types  # [num_nodes, num_node_feats=21 if ca_only is True, 38 otherwise]
    # Cartesian coordinates for each atom
    graph.ndata['x_pred'] = node_coords  # [num_nodes, 3]
    # One-hot ID representation of the atoms for which true coordinates were available in the corresponding exp. struct.
    graph.ndata['labeled'] = torch.zeros((graph.number_of_nodes(), 1), dtype=torch.int32)
    # One-hot ID representation of the atoms present in any interface between at least two chains
    interfacing = torch.zeros((graph.number_of_nodes(), 1), dtype=torch.int32)
    interfacing[interfacing_atom_indices] = 1
    graph.ndata['interfacing'] = interfacing  # [num_nodes, 1]
    # Single scalar describing the covalent radius of each atom
    null_cov_radius = torch.zeros_like(graph.ndata['interfacing'])
    covalent_radius = null_cov_radius if ca_only else torch.FloatTensor(input_atom_df['covalent_radius']).reshape(-1, 1)
    graph.ndata['covalent_radius'] = covalent_radius
    # Single value indicating to which chain an atom belongs
    graph.ndata['chain_id'] = chain_ids.reshape(-1, 1)  # [num_nodes, 1]
    # Integers describing to which residue an atom belongs
    graph.ndata['residue_number'] = torch.IntTensor(input_atom_df['residue_number'].tolist())  # [num_nodes, 1]
    # Scalars describing each atom's proximity to the surface of its chain
    graph.ndata['surf_prox'] = compute_surface_proximities(graph, input_filepath, ca_only)  # [num_nodes, 1]
    # Binary integers (i.e., 0 or 1) describing whether an atom is a carbon-alpha (Ca) atom
    graph.ndata['is_ca_atom'] = determine_is_ca_atom(graph, input_atom_df.reset_index(drop=True))  # [num_nodes, 1]
    # Scalars describing the cosine and sine-activated phi, psi, and omega backbone dihedral angles for each residue
    dihedral_angles = compute_dihedral_angles(input_filepath) if ca_only else torch.zeros_like(graph.ndata['x_pred'])
    graph.ndata['dihedral_angles'] = dihedral_angles  # [num_nodes, 6]

    """Encode edge features in graph"""
    # Positional encoding for each edge (used for sequentially-ordered inputs like proteins)
    graph.edata['pos_enc'] = torch.sin((graph.edges()[0] - graph.edges()[1]).float()).reshape(-1, 1)  # [num_edges, 1]
    update_relative_positions(graph)  # Relative positions of nodes j to nodes i - [num_edges, 1]
    update_potential_values(graph, r=graph.edata['r'])  # Weights from nodes j and nodes i - [num_edges, 1]
    bond_types = torch.zeros_like(graph.edata['r']) if ca_only else compute_bond_edge_feats(graph, first_iter=True)
    graph.edata['bond_type'] = bond_types.reshape(-1, 1)  # [num_edges, 1]
    graph.apply_edges(compute_chain_matches)  # Install edata['in_same_chain'] - [num_edges, 1]
    rel_geom_feats = compute_rel_geom_feats(graph, orig_pred_atom_df, input_file=input_filepath)
    graph.edata['rel_geom_feats'] = rel_geom_feats  # [num_edges, 12]

    """Ensure no input feature values are invalid (e.g., NaN)"""
    nans_in_ndata = torch.isnan(graph.ndata['x_pred']).any() \
                    or torch.isnan(graph.ndata['atom_type']).any() \
                    or torch.isnan(graph.ndata['labeled']).any() \
                    or torch.isnan(graph.ndata['interfacing']).any() \
                    or torch.isnan(graph.ndata['covalent_radius']).any() \
                    or torch.isnan(graph.ndata['chain_id']).any() \
                    or torch.isnan(graph.ndata['residue_number']).any() \
                    or torch.isnan(graph.ndata['surf_prox']).any() \
                    or torch.isnan(graph.ndata['is_ca_atom']).any() \
                    or torch.isnan(graph.ndata['dihedral_angles']).any()
    nans_in_edata = torch.isnan(graph.edata['pos_enc']).any() \
                    or torch.isnan(graph.edata['rel_pos']).any() \
                    or torch.isnan(graph.edata['r']).any() \
                    or torch.isnan(graph.edata['bond_type']).any() \
                    or torch.isnan(graph.edata['in_same_chain']).any() \
                    or torch.isnan(graph.edata['rel_geom_feats']).any()
    assert not (nans_in_ndata or nans_in_edata), 'There must be no invalid (i.e., NaN) values in the graph features'

    # Return our resulting graph
    return graph


def process_protein_into_dict(pred_filepath: str, true_filepath: str, output_filepath: str,
                              atom_selection_type: str, knn: int, idt: float, output_iviz: bool):
    """Process protein into a dictionary representing its DGLGraph representation and ground-truth PDB."""
    # Load input protein's predicted and ground-truth PDB file, both as Pandas DataFrames
    pred_pdb = PandasPdb().read_pdb(pred_filepath)
    true_pdb = PandasPdb().read_pdb(true_filepath)

    # For a residue with alternative atom locations, choose the first unique set of atom locations to represent it
    pred_pdb.df['ATOM'] = pred_pdb.df['ATOM'].drop_duplicates(
        subset=['atom_name', 'residue_name', 'chain_id', 'residue_number']
    )
    true_pdb.df['ATOM'] = true_pdb.df['ATOM'].drop_duplicates(
        subset=['atom_name', 'residue_name', 'chain_id', 'residue_number']
    )

    # Convert each input protein into its graph representation, using all its atoms as the produced graph's nodes
    graph, orig_true_df_indices = convert_dfs_to_dgl_graph(
        pred_pdb, true_pdb, pred_filepath, atom_selection_type, knn, idt, output_iviz
    )

    # Assemble all valid combinations of chains for scoring chain-pair body intersection losses
    unique_chain_ids = np.unique(graph.ndata['chain_id'])
    chain_combinations = torch.tensor(list(combinations(unique_chain_ids, r=2))).float()

    # Collect metadata concerning the number of atoms and residues in each predicted and true structure
    num_pred_atoms = pred_pdb.df['ATOM'].shape[0]
    num_pred_residues = pred_pdb.df['ATOM'][pred_pdb.df['ATOM']['atom_name'] == 'CA'].shape[0]
    num_true_atoms = true_pdb.df['ATOM'].shape[0]
    num_true_residues = true_pdb.df['ATOM'][true_pdb.df['ATOM']['atom_name'] == 'CA'].shape[0]
    num_shared_atoms = len(orig_true_df_indices)

    # Represent each protein as a metadata dictionary containing, for example, its graph instance
    processed_protein = {
        'protein': db.get_pdb_name(pred_filepath),
        'graph': graph,
        'orig_true_df_indices': orig_true_df_indices,
        'chain_combinations': chain_combinations,
        'num_pred_atoms': num_pred_atoms,
        'num_true_atoms': num_true_atoms,
        'num_shared_atoms': num_shared_atoms,
        'num_pred_residues': num_pred_residues,
        'num_true_residues': num_true_residues
    }

    # Write into 'output_filepath'
    processed_file_dir = os.path.join(*output_filepath.split(os.sep)[: -1])
    os.makedirs(processed_file_dir, exist_ok=True)
    with open(output_filepath, 'wb') as f:
        pickle.dump(processed_protein, f)


def drop_intermediate_ter_entries(df: pd.DataFrame, chain_id_sub_value='A') -> pd.DataFrame:
    """Remove all 'TER' entries except the last one, to simulate the structure having one chain."""
    ter_entry_row_idx = df.index[df['record_name'] == 'TER'][:-1]
    df = df.drop(index=ter_entry_row_idx)
    # Substitute the chain ID for the last TER entry
    ter_entry = df.loc[df['record_name'] == 'TER']['entry'].values.tolist()[0]
    existing_chain_id = [s for s in ter_entry.split(' ') if len(s) == 1 and s.isalpha() and s.isupper()][0]
    mapped_ter_entry = re.sub(f' {existing_chain_id} ', f' {chain_id_sub_value} ', ter_entry)
    df.loc[df['record_name'] == 'TER', 'entry'] = mapped_ter_entry
    return df


def reindex_df_field_values(df: pd.DataFrame, field_name: str, start_index: int) -> pd.DataFrame:
    """
    Reindex a Series of consecutive integers, corresponding to a specific DataFrame field, to start from a given index.

    Parameters
    ----------
    df: pd.DataFrame
    field_name: str
    start_index: int

    Returns
    -------
    pd.DataFrame
    """
    field_values = df[[field_name]].values.squeeze().tolist()
    reindexed_field_values = [c for c, (k, g) in enumerate(groupby(field_values), start_index) for _ in g]
    df[[field_name]] = np.array(reindexed_field_values).reshape(-1, 1)  # Install reindexed field values
    return df


def relabel_pdb_chains(pdb: PandasPdb, filepath: str, metric: str, chain_id_sub_value='A'):
    """
    Label all chains in a given structure identically to enable full multimeric compatibility with TMscore.

    Parameters
    ----------
    pdb: PandasPdb
    filepath: str
    metric: str
    chain_id_sub_value: str
    """
    assert metric in ['tm', 'mm', 'dq'], 'Metric specified must be either TMscore, MMalign, or DockQ'
    # Flush latest PDB to storage
    pdb.to_pdb(filepath)
    # Gather PDB details
    chain_ids = np.unique(pdb.df['ATOM'][['chain_id']].values.squeeze())
    is_multimer = len(chain_ids) > 1
    # Drop all TER entries except the last, and replace all chain IDs by 'A' (by default)
    if is_multimer:
        for key in pdb.df.keys():
            if key in ['ATOM', 'HETATM', 'ANISOU', 'OTHERS']:
                if len(pdb.df[key]) > 0:
                    if metric in ['tm', 'dq']:
                        if key == 'OTHERS':
                            if metric == 'tm':  # Retain TER chain information when generating DockQ scores
                                pdb.df[key] = drop_intermediate_ter_entries(pdb.df[key], chain_id_sub_value)
                        else:
                            if metric == 'tm':  # Avoid replacing chain ID when generating DockQ scores
                                pdb.df[key][['chain_id']] = chain_id_sub_value  # Give all residues the same chain ID
                    if key in ['ATOM', 'HETATM', 'ANISOU']:
                        pdb.df[key] = reindex_df_field_values(pdb.df[key], field_name='atom_number', start_index=1)
                        pdb.df[key] = reindex_df_field_values(pdb.df[key], field_name='residue_number', start_index=1)
    # Flush updated PDB to storage
    pdb.to_pdb(filepath)


def is_divisible_by(first: int, other: int) -> bool:
    """
    Determine whether the first integer is evenly divisible by the second integer.

    Parameters
    ----------
    first: int
    other: int

    Returns
    -------
    bool
    """
    return (first % other) == 0


def collect_args() -> argparse.ArgumentParser:
    """Collect all arguments required for training/testing."""
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    # -----------------
    # Model arguments
    # -----------------
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers in a given neural network model')
    parser.add_argument('--layer_norm', type=str, default='LN', choices=['0', 'BN', 'LN', 'IN', 'GN'],
                        help='Which version of normalization to apply to each layer\'s representations')
    parser.add_argument('--optimizer_type', default='adamw', choices=['sgd', 'adam', 'adamw', 'nadam', 'radam'],
                        help='Type of network weight optimizer to use during training')
    parser.add_argument('--scheduler_type', default='warmup', choices=['warmup', 'cyclic', 'cawr', 'slr', 'rop'],
                        help='Type of learning rate scheduler to use with optimizer')
    parser.add_argument('--warmup', type=float, default=1.0, help='The warmup value to use with a given LR scheduler')
    parser.add_argument('--metric_to_track', type=str, default='val/f_rmsd', help='LR schedule and early stop crit.')
    parser.add_argument('--remove_aug_structures', default=False, action='store_true',
                        help='Whether to train models with an augmented training dataset')
    parser.add_argument('--manually_init_weights', default=False, action='store_true',
                        help='Whether to manually initialize the weights and biases in the requested IEGR module')
    parser.add_argument('--use_loss_clamping', default=False, action='store_true',
                        help='Whether to clamp the PSR HuberLoss with a maximum loss value')
    parser.add_argument('--use_ext_tool_only', default=False, action='store_true',
                        help='Whether to refine test input proteins using only an external tool (e.g., Modeller)')

    parser.add_argument('--nn_type', type=str, default='EGR', choices=['EGR', 'SEGNN', 'SET'],
                        help='Type of neural network to use for forward propagation')
    parser.add_argument('--num_iter', type=int, default=1,
                        help='Number of times to restructure input graph topology based on coordinate outputs')
    parser.add_argument('--num_attn_heads', type=int, default=4,
                        help='Number of heads to use in self-attention calculations')
    parser.add_argument('--num_hidden_channels', type=int, default=200,
                        help='Dimensionality of hidden node and edge representations during network forward passes')
    parser.add_argument('--pooling', type=str, default='None', const='None', nargs='?', choices=['None', 'max', 'avg'],
                        help='Type of graph pooling to employ after all iterations of message-passing')
    parser.add_argument('--tmscore_exec_path', type=str,
                        default=os.path.join(str(Path.home()), 'Programs', 'MMalign'),
                        help='Path to executable for the TMscore program')
    parser.add_argument('--dockq_exec_path', type=str,
                        default=os.path.join(str(Path.home()), 'Programs', 'DockQ', 'DockQ.py'),
                        help='Path to executable for the TMscore program')
    parser.add_argument('--galaxy_exec_path', type=str,
                        default=os.path.join(str(Path.home()), 'Programs', 'GalaxyRefineComplex'),
                        help='Path to executable for the GalaxyRefineComplex program')
    parser.add_argument('--galaxy_home_path', type=str,
                        default=os.path.join(
                            str(Path.home()), 'Repositories', 'Lab_Repositories', 'GalaxyRefineComplex'
                        ),
                        help='Path to home directory (i.e., GitHub repository) for the GalaxyRefineComplex program')
    parser.add_argument('--viz_every_n_epochs', type=int, default=1,
                        help='By how many epochs to space out model prediction visualizations during training')
    parser.add_argument('--eval_init_prots', action='store_true', dest='eval_init_prots',
                        help='Whether to CSV-record all test proteins\' initial qualities')
    parser.add_argument('--pdb_filepath', type=str, default='test_data/6A6I.pdb',
                        help='A filepath to the input PDB, containing all chains')
    parser.add_argument('--f_hl_weight', type=float, default=1.0, help='Weight of full HuberLoss')
    parser.add_argument('--qa_loss_weight', type=float, default=0.005, help='Weight of quality assessment loss')
    parser.add_argument('--i_hl_weight', type=float, default=0.0, help='Weight of interface HuberLoss')
    parser.add_argument('--bil_weight', type=float, default=0.0, help='Weight of intersection loss')
    parser.add_argument('--bil_sigma', type=float, default=25.0, help='Spread of intersection RBF')
    parser.add_argument('--bil_surface_ct', type=float, default=10.0, help='Intersection surface RBF point count')

    # -----------------
    # Data arguments
    # -----------------
    parser.add_argument('--atom_selection_type', type=str, default='ca_atom', help='Type(s) of atoms to use in graphs')
    parser.add_argument('--knn', type=int, default=20, help='Number of nearest neighbor edges for each node')
    parser.add_argument('--idt', type=float, default=8.0, help='Distance under which two inter-chain atoms interact')
    parser.add_argument('--main_dataset_dir', type=str, default='datasets/PSR/final/raw',
                        help='Path to directory in which to find protein graphs for training and validation')
    parser.add_argument('--testing_with_b2', action='store_true', dest='testing_with_b2', help='Test with Benchmark 2')
    parser.add_argument('--graph_return_format', type=str, default='dgl', help='Which graph format to return')
    parser.add_argument('--test_filenames_ver', type=str, default='', help='Which version of test filenames to use')
    parser.add_argument('--viz_all_orig', action='store_true', dest='viz_all_orig', help='Plot all orig. predictions')
    parser.add_argument('--b2_dataset_dir', type=str, default='datasets/Benchmark_2/final/raw',
                        help='Path to directory in which to find protein graphs for testing')
    parser.add_argument('--input_dataset_dir', type=str, default='datasets/Input/jian_data/7ALA',
                        help='Path to directory in which to generate temporary files for the given inputs')
    parser.add_argument('--output_dir', type=str, default='datasets/Input/jian_data/7ALA',
                        help='Path to directory in which to store model predictions')
    parser.add_argument('--process_proteins', action='store_true', dest='process_proteins',
                        help='Check if all proteins for a dataset are processed and, if not, process those remaining')
    parser.add_argument('--input_indep', type=str, default='none',
                        help='How and whether to zero-out input node and edge features (i.e., "none" or "full")')
    parser.add_argument('--normalize_input_feats', action='store_true', dest='normalize_input_feats',
                        help='Whether to normalize input features in-place')
    parser.add_argument('--learn_x_pred_reconstruction', default=False, action='store_true',
                        help='Whether to learn to reconstruct initial node coordinates prior to adding noise')
    parser.add_argument('--x_noise_sigma', type=float, default=0.1, help='Std. dev. of node position noise to inject')
    parser.add_argument('--ablate_node_feature', type=str, default=None, choices=['None', 'atom_type', 'surf_prox'],
                        help='Which node feature to singularly ablate, if any')
    parser.add_argument('--ablate_edge_feature', type=str, default=None,
                        choices=['None', 'pos_enc', 'in_same_chain', 'rel_geom_feats', 'bond_type'],
                        help='Which edge feature to singularly ablate, if any')

    # -----------------
    # Logging arguments
    # -----------------
    parser.add_argument('--logger_name', type=str, default='TensorBoard', help='Which logger to use for experiments')
    parser.add_argument('--experiment_name', type=str, default=None, help='Logger experiment name')
    parser.add_argument('--project_name', type=str, default='DeepRefine', help='Logger project name')
    parser.add_argument('--entity', type=str, default='bml-lab', help='Logger entity (i.e. team) name')
    parser.add_argument('--run_id', type=str, default='', help='Logger run ID')
    parser.add_argument('--offline', action='store_true', dest='offline', help='Whether to log locally or remotely')
    parser.add_argument('--online', action='store_false', dest='offline', help='Whether to log locally or remotely')
    parser.add_argument('--tb_log_dir', type=str, default='tb_logs', help='Where to store TensorBoard log files')
    parser.set_defaults(offline=False)  # Default to using online logging mode

    # -----------------
    # Seed arguments
    # -----------------
    parser.add_argument('--seed', type=int, default=None, help='Seed for NumPy and PyTorch')

    # -----------------
    # Meta-arguments
    # -----------------
    parser.add_argument('--batch_size', type=int, default=1, help='Number of samples included in each data batch')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=2.0, help='Decay rate of optimizer weight')
    parser.add_argument('--num_epochs', type=int, default=500, help='Maximum number of epochs to run for training')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout (forget) rate')
    parser.add_argument('--patience', type=int, default=100, help='Number of epochs to wait until early stopping')
    parser.add_argument('--run_recursively', action='store_true', dest='run_recursively',
                        help='Whether the training script is being launched recursively for long-term training')

    # -----------------
    # Miscellaneous
    # -----------------
    parser.add_argument('--max_hours', type=int, default=1, help='Maximum number of hours to allot for training')
    parser.add_argument('--max_minutes', type=int, default=59, help='Maximum number of minutes to allot for training')
    parser.add_argument('--max_seconds', type=int, default=00, help='Maximum number of seconds to allot for training')
    parser.add_argument('--device_type', type=str, default='gpu', help='Device type for training')
    parser.add_argument('--device_strategy', type=str, default='ddp', help='Device management backend for training')
    parser.add_argument('--num_devices', type=int, default=1, help='Number of GPUs to use (e.g. -1 = all avail. GPUs)')
    parser.add_argument('--num_compute_nodes', type=int, default=1, help='Number of compute nodes to use')
    parser.add_argument('--gpu_precision', type=int, default=32, help='Bit size used during training (e.g., 32-bit)')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of CPU threads for loading data')
    parser.add_argument('--profiler_method', type=str, default=None, help='PL profiler to use (e.g. simple)')
    parser.add_argument('--ckpt_dir', type=str, default=f'{os.path.join(os.getcwd(), "checkpoints")}',
                        help='Directory in which to save checkpoints')
    parser.add_argument('--ckpt_name', type=str, default='', help='Filename of best checkpoint')
    parser.add_argument('--min_delta', type=float, default=1e-5, help='Minimum percentage of change required to'
                                                                      ' "metric_to_track" before early stopping'
                                                                      ' after surpassing patience')
    parser.add_argument('--accum_grad_batches', type=int, default=1, help='Number of gradient batches to accumulate')
    parser.add_argument('--grad_clip_val', type=float, default=0.5, help='Value over which to clip gradients')
    parser.add_argument('--grad_clip_algo', type=str, default='norm', help='Algorithm with which to clip gradients')
    parser.add_argument('--stc_weight_avg', action='store_true', dest='stc_weight_avg', help='Smooth loss landscape')
    parser.add_argument('--check_val_every_n_train_epochs', type=int, default=1, help='Check val every n train epochs')

    return parser


def seed_everything(seed: int):
    """Set the random seed for PyTorch (& Lightning), NumPy, DGL, etc."""
    logging.info(f'Seeding everything with random seed {seed}')
    pl.seed_everything(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.seed(seed)
    dgl.random.seed(seed)


def process_args(args: argparse.Namespace) -> argparse.Namespace:
    """Process all arguments required for training/testing."""
    # ---------------------------------------
    # Seed fixing for random numbers
    # ---------------------------------------
    if not args.seed:
        args.seed = 1  # np.random.randint(100000)
    seed_everything(args.seed)  # Set the random seed for a variety of Python packages
    return args


def construct_strategy(strategy_name: str, find_unused: bool) -> Union[str, pl.plugins.PLUGIN]:
    """
    Return a Lightning training strategy requested by the user.

    Parameters
    ----------
    strategy_name: str
    find_unused: bool

    Returns
    -------
    Union[str, pl.plugins.PLUGIN]
    """
    if strategy_name == 'dp':
        strategy = strategy_name
    elif strategy_name == 'ddp':
        strategy = DDPPlugin(find_unused_parameters=find_unused)
    elif strategy_name == 'ddp_sharded':
        strategy = DDPShardedPlugin()
    elif strategy_name == 'fsdp':
        strategy = strategy_name
    elif strategy_name == 'deepspeed_stage_1':
        strategy = DeepSpeedPlugin(zero_optimization=True,
                                   stage=1,
                                   offload_optimizer=False,
                                   allgather_bucket_size=2e8,
                                   reduce_bucket_size=2e8)
    elif strategy_name == 'deepspeed_stage_2':
        strategy = DeepSpeedPlugin(zero_optimization=True,
                                   stage=2,
                                   offload_optimizer=False,
                                   overlap_comm=True,
                                   allgather_bucket_size=2e8,
                                   reduce_bucket_size=2e8)
    elif strategy_name == 'deepspeed_stage_2_offload':
        strategy = DeepSpeedPlugin(zero_optimization=True,
                                   stage=2,
                                   offload_optimizer=True,
                                   overlap_comm=True,
                                   allgather_bucket_size=2e8,
                                   reduce_bucket_size=2e8)
    elif strategy_name == 'deepspeed_stage_3':
        strategy = DeepSpeedPlugin(zero_optimization=True,
                                   stage=3,
                                   offload_optimizer=False,
                                   overlap_comm=True,
                                   allgather_bucket_size=2e8,
                                   reduce_bucket_size=2e8)
    elif strategy_name == 'deepspeed_stage_3_offload':
        strategy = DeepSpeedPlugin(zero_optimization=True,
                                   stage=3,
                                   offload_optimizer=True,
                                   overlap_comm=True,
                                   allgather_bucket_size=2e8,
                                   reduce_bucket_size=2e8)
    else:
        raise NotImplementedError(f'Training strategy {strategy_name} is not currently supported')
    return strategy


def construct_pl_logger(args: argparse.Namespace) -> pl.loggers.LightningLoggerBase:
    """Return a specific Logger instance requested by the user."""
    if args.logger_name.lower() == 'wandb':
        return construct_wandb_pl_logger(args)
    else:  # Default to using TensorBoard
        return construct_tensorboard_pl_logger(args)


def construct_wandb_pl_logger(args: argparse.Namespace) -> pl.loggers.LightningLoggerBase:
    """Return an instance of WandbLogger with corresponding project and name strings."""
    return WandbLogger(name=args.experiment_name,
                       offline=args.offline,
                       id=args.run_id,
                       resume='must' if args.run_recursively and args.ckpt_provided else False,
                       version=args.run_id,
                       project=args.project_name,
                       log_model=not args.offline,
                       entity=args.entity)


def construct_tensorboard_pl_logger(args: argparse.Namespace) -> pl.loggers.LightningLoggerBase:
    """Return an instance of TensorBoardLogger with corresponding project and experiment name strings."""
    return TensorBoardLogger(save_dir=args.tb_log_dir,
                             name=args.experiment_name)
