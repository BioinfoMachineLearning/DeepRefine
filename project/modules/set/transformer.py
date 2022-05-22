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
from typing import Optional, Dict, Literal, List

import dgl
import torch
import torch.nn as nn
from torch import Tensor

from project.modules.set.basis import get_basis, update_basis_with_fused
from project.modules.set.fiber import Fiber
from project.modules.set.layers.attention import AttentionBlockSE3
from project.modules.set.layers.convolution import ConvSE3FuseLevel, ConvSE3
from project.modules.set.layers.norm import NormSE3
from project.modules.set.layers.pooling import GPooling
from project.utils.set.runtime.utils import update_potential_values, \
    iset_copy_without_weak_connections, update_absolute_positions, update_relative_positions


def get_populated_edge_features(r: Tensor, edge_features: Optional[Dict[str, Tensor]] = None):
    """ Add relative positions to existing edge features """
    edge_features = edge_features.copy() if edge_features else {}
    if '0' in edge_features:
        edge_features['0'] = torch.cat([edge_features['0'], r], dim=1)
    else:
        edge_features['0'] = r

    return edge_features


class IterativeSE3Transformer(nn.Module):
    def __init__(self,
                 num_layers: int,
                 fiber_in: Fiber,
                 fiber_hidden: Fiber,
                 fiber_out: Fiber,
                 num_heads: int,
                 channels_div: int,
                 num_degrees: int,
                 num_channels: int,
                 knn: int,
                 batched_input: bool,
                 num_iter=3,
                 compute_gradients=True,
                 fiber_edge: Fiber = Fiber({}),
                 pooling: Optional[Literal['max', 'avg']] = None,
                 pred_struct_qual=False,
                 output_dim: int = 1,
                 norm: bool = True,
                 use_layer_norm: bool = True,
                 tensor_cores: bool = False,
                 low_memory: bool = True,
                 amp: bool = True,
                 **kwargs):
        """
        :param num_layers:          Number of attention layers per SE(3)-Transformer module
        :param fiber_in:            Input fiber description
        :param fiber_hidden:        Hidden fiber description
        :param fiber_out:           Output fiber description
        :param num_heads:           Number of attention heads
        :param channels_div:        Channels division before feeding to attention layer
        :param num_degrees:         Number of degrees (i.e., types) in a hidden layer (Note: Count starts from type-0)
        :param num_channels:        Number of channels per degree
        :param knn:                 Number of nearest neighbors with which to construct new graph edges dynamically
        :param batched_input:       Whether the input graph is a batched graph, comprised of multiple subgraphs
        :param num_iter             Number of times to restructure input graph topology based on coordinate outputs
        :param compute_gradients:   Whether to backpropagate through spherical harmonics computations
        :param fiber_edge:          Input edge fiber description
        :param pooling:             'max' or 'avg' graph pooling before MLP layers
        :param pred_struct_qual:    Whether to predict the quality of a predicted structure
        :param output_dim:          Output dimensionality of pooled features
        :param norm:                Apply a normalization layer after each attention block
        :param use_layer_norm:      Apply layer normalization between MLP layers
        :param tensor_cores:        True if using Tensor Cores (affects the use of fully fused convs, and padded bases)
        :param low_memory:          If True, will use slower ops that use less memory
        :param amp:                 If True, will use half floating-point precision for basis construction
        """
        super().__init__()
        self.num_layers = num_layers
        self.fiber_edge = fiber_edge
        self.num_heads = num_heads
        self.channels_div = channels_div
        self.num_degrees = num_degrees
        self.num_channels = num_channels
        self.knn = knn
        self.batched_input = batched_input
        self.num_iter = num_iter
        self.compute_gradients = compute_gradients
        self.pooling = pooling
        self.pred_struct_qual = pred_struct_qual
        self.max_degree = max(*fiber_in.degrees, *fiber_hidden.degrees, *fiber_out.degrees)
        self.tensor_cores = tensor_cores
        self.low_memory = low_memory
        self.amp = amp

        if low_memory:
            self.fuse_level = ConvSE3FuseLevel.NONE
        else:
            # Fully fused convolutions when using Tensor Cores (and not low memory mode)
            self.fuse_level = ConvSE3FuseLevel.FULL if tensor_cores else ConvSE3FuseLevel.PARTIAL

        outer_graph_module = []
        for _ in range(num_iter):
            # Combine the requested number of SE(3)-Transformer modules
            inner_graph_modules = []
            for _ in range(num_layers):
                # Assemble each SE(3)-Transformer module by accumulating its 'num_layers'
                inner_graph_modules.append(AttentionBlockSE3(fiber_in=fiber_in,
                                                             fiber_out=fiber_hidden,
                                                             fiber_edge=fiber_edge,
                                                             num_heads=num_heads,
                                                             channels_div=channels_div,
                                                             use_layer_norm=use_layer_norm,
                                                             max_degree=self.max_degree,
                                                             fuse_level=self.fuse_level,
                                                             low_memory=low_memory))
                if norm:
                    inner_graph_modules.append(NormSE3(fiber_hidden))
                fiber_in = fiber_hidden

            outer_graph_module.append(nn.ModuleList(inner_graph_modules))

        # To predict structures' quality, during the last iteration only output new type-0 node representations
        fiber_out = Fiber({0: self.num_degrees * self.num_channels})

        outer_graph_module.append(nn.ModuleList([ConvSE3(fiber_in=fiber_in,
                                                         fiber_out=fiber_out,
                                                         fiber_edge=fiber_edge,
                                                         self_interaction=True,
                                                         use_layer_norm=use_layer_norm,
                                                         max_degree=self.max_degree)]))
        self.outer_graph_module = nn.ModuleList(outer_graph_module)

        if self.pred_struct_qual:
            n_out_features = fiber_out.channels[0]
            if pooling is not None:
                # Prepare to predict node-pooled structure quality scores
                self.pooling_module = GPooling(feat_type=0, pool=pooling)
                self.mlp = nn.Sequential(
                    nn.Linear(n_out_features, n_out_features),
                    nn.ReLU(),
                    nn.Linear(n_out_features, output_dim)
                )
            else:
                # Prepare to predict per-node LDDT scores
                self.norm_lddt = nn.LayerNorm(n_out_features)
                self.pred_lddt = nn.Linear(n_out_features, 1)

    def forward(self, graph: dgl.DGLGraph, pdb_filepaths: List[str]):
        """Iterate through all requested SE(3)-equivariant layers to obtain type-N output features."""
        node_feats, edge_feats = {}, {}
        for i, inner_modules in enumerate(self.outer_graph_module):
            if i == 0:
                # For the first layer, restructure the latest node and edge features
                node_feats = {'0': graph.ndata['f'][:, :, None].float()}
                edge_feats = {'0': graph.edata['f'][:, :, None].float()}
            else:
                # Otherwise, update existing feature tensor(s)
                edge_feats['0'] = graph.edata['f'][:, :, None]

            # Compute the latest version of the spherical harmonics bases to support SE(3)-equivariance
            basis = get_basis(graph.edata['rel_pos'], max_degree=self.max_degree,
                              compute_gradients=self.compute_gradients,
                              use_pad_trick=self.tensor_cores and not self.low_memory,
                              amp=self.amp)

            # Add fused bases (per output degree, per input degree, and fully fused) to the dict
            basis = update_basis_with_fused(basis, self.max_degree,
                                            use_pad_trick=self.tensor_cores and not self.low_memory,
                                            fully_fused=self.fuse_level == ConvSE3FuseLevel.FULL)

            # Collect and restructure edge input features, since they may change after updating coordinates
            edge_feats = get_populated_edge_features(graph.edata['r'].reshape(-1, 1, 1), edge_feats)

            # Iterate through a single set of SE(3)-equivariant modules to obtain type-N output features
            for inner_module in inner_modules:
                node_feats = inner_module(node_feats, edge_features=edge_feats, graph=graph, basis=basis)

            # Update node coordinates in every iteration except last s.t. pooled nodes represent final graph structure
            if i == self.num_iter:
                # Store latest, unraveled node features in the graph
                graph.ndata['f'] = node_feats['0'].squeeze(-1)
            else:
                # Arbitrarily use the first type-1 feature for coordinate updates
                coord_updates = node_feats['1'][:, 0, :]
                update_absolute_positions(graph, pos_updates=coord_updates)
                update_relative_positions(graph)

                # Rewire the input graph only if more than one iteration is requested
                if self.num_iter > 1:
                    # Copy new strong-link subgraphs corresponding to weak-link subgraphs where KNN < num_nodes
                    num_nodes = graph.batch_num_nodes()
                    knn_lt_num_nodes = ((num_nodes > self.knn).float() == 1).nonzero().squeeze()
                    if torch.numel(knn_lt_num_nodes) > 0:
                        update_potential_values(graph, r=graph.edata['r'])  # Obtain new potentials to guide rewiring
                        graph = iset_copy_without_weak_connections(graph,
                                                                   graph_idx=knn_lt_num_nodes,
                                                                   edges_per_node=self.knn,
                                                                   batched_input=self.batched_input,
                                                                   pdb_filepaths=pdb_filepaths)

        if self.pred_struct_qual:
            if self.pooling is not None:
                # Predict node-pooled structure quality scores
                pooled_feats = self.pooling_module({'0': graph.ndata['f']}, graph=graph)
                graph_output = self.mlp(pooled_feats).squeeze(-1)
                graph_output = graph_output.type(graph.ndata['f'].dtype)
            else:
                # Predict per-node LDDT scores
                pred_node_feats = self.pred_lddt(self.norm_lddt(graph.ndata['f']))
                pred_lddt = torch.clamp(pred_node_feats, 0.0, 1.0)
                graph_output = pred_lddt.type(graph.ndata['f'].dtype)
            graph.gdata = {'q': graph_output}  # Add quality prediction by installing it as a global graph attribute

        # Return updated graph and its learned representations
        return graph
