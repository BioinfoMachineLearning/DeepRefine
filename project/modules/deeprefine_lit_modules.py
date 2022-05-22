import argparse
from typing import Any, Optional, List

import dgl
import pytorch_lightning as pl
import torch.nn as nn
from e3nn.o3 import Irreps

from project.modules.egr.egr import IterativeEquivariantGraphRefinementModel
from project.modules.segnn.balanced_irreps import WeightBalancedIrreps
from project.modules.segnn.segnn import SteerableEquivariantGraphRefinementModel
from project.modules.set.fiber import Fiber
from project.modules.set.transformer import IterativeSE3Transformer
from project.utils.deeprefine_constants import PROT_ATOM_NAMES
from project.utils.set.runtime.utils import using_tensor_cores, str2bool


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for DeepRefine (https://github.com/BioinfoMachineLearning/DeepRefine):
# -------------------------------------------------------------------------------------------------------------------------------------

# ------------------
# Lightning Modules
# ------------------
class LitPSR(pl.LightningModule):
    """A LightningModule for protein structure refinement (PSR)."""

    def __init__(self, num_node_input_feats: int, num_edge_input_feats: int, atom_selection_type: str,
                 graph_return_format: str, tmscore_exec_path: str, dockq_exec_path: str, galaxy_exec_path: str,
                 galaxy_home_path: str, device_strategy: str, nn_type='EGR', optimizer_type='adamw',
                 scheduler_type='warmup', warmup=1.0, patience=100, num_layers=4, num_iter=1, num_attn_heads=4,
                 num_hidden_channels=200, pooling=None, knn=20, channels_div=2, num_degrees=2, norm=True,
                 use_layer_norm=True, low_memory=False, amp=False, num_epochs=500, num_atom_types=38, atom_emb_dim=200,
                 use_surf_prox=True, use_dihedral_angles=True, use_edge_feats_in_iegr_layers=True,
                 use_dist_in_iegr_layers=True, use_local_attention=False, use_global_attention=True, attn_depth=1,
                 update_pos_with_ca_atoms=False, dropout=0.0, nonlin='lkyrelu', layer_norm='LN', layer_norm_coords='0',
                 skip_weight_h=0.5, x_connection_init=0.5, leakyrelu_neg_slope=1e-2, shared_layers=False,
                 norm_coord_updates=True, coord_norm_const=1.0, learn_skip_weights=False, lmax_attr=3, lmax_h=2,
                 pos_skip_connection=True, metric_to_track='val/f_rmsd', weight_decay=2.0, batch_size=1, lr=1e-5,
                 manually_init_weights=False, use_loss_clamping=False, use_ext_tool_only=False, f_hl_weight=1.0,
                 qa_loss_weight=0.005, i_hl_weight=0.0, bil_weight=0.0, bil_sigma=25.0, bil_surface_ct=10.0,
                 viz_every_n_epochs=1, experiment_name='LitPSR', use_wandb_logger=True, eval_init_prots=False,
                 testing_with_b2=True, test_dataset_name='Benchmark_2', viz_all_orig=False):
        """Initialize all the parameters for a LitPSR module."""
        super().__init__()

        # Design the network
        self.num_node_input_feats = num_node_input_feats
        self.num_edge_input_feats = num_edge_input_feats

        # Module keyword arguments provided via the command line
        self.atom_selection_type = atom_selection_type
        self.graph_return_format = graph_return_format
        self.tmscore_exec_path = tmscore_exec_path
        self.dockq_exec_path = dockq_exec_path
        self.galaxy_exec_path = galaxy_exec_path
        self.galaxy_home_path = galaxy_home_path
        self.device_strategy = device_strategy
        self.nn_type = nn_type.upper().strip()
        self.optimizer_type = optimizer_type.lower().strip()
        self.scheduler_type = scheduler_type.lower().strip()
        self.warmup = warmup
        self.patience = patience
        self.num_layers = num_layers
        self.num_iter = num_iter
        self.num_attn_heads = num_attn_heads
        self.num_hidden_channels = num_hidden_channels
        self.pooling = pooling
        self.knn = knn

        # SET model keyword arguments provided via the command line
        self.channels_div = channels_div
        self.num_degrees = num_degrees
        self.norm = norm
        self.use_layer_norm = use_layer_norm
        self.low_memory = low_memory
        self.amp = amp

        # SEGNN model keyword arguments provided via the command line
        self.lmax_attr = lmax_attr
        self.lmax_h = lmax_h
        self.pos_skip_connection = pos_skip_connection

        # EGR model keyword arguments provided via the command line
        self.num_atom_types = num_atom_types
        self.atom_emb_dim = atom_emb_dim
        self.use_surf_prox = use_surf_prox
        self.use_dihedral_angles = use_dihedral_angles
        self.use_edge_feats_in_iegr_layers = use_edge_feats_in_iegr_layers
        self.use_dist_in_iegr_layers = use_dist_in_iegr_layers
        self.use_local_attention = use_local_attention
        self.use_global_attention = use_global_attention
        self.attn_depth = attn_depth
        self.update_pos_with_ca_atoms = update_pos_with_ca_atoms
        self.dropout = dropout
        self.nonlin = nonlin
        self.layer_norm = layer_norm
        self.layer_norm_coords = layer_norm_coords
        self.skip_weight_h = skip_weight_h
        self.x_connection_init = x_connection_init
        self.leakyrelu_neg_slope = leakyrelu_neg_slope
        self.shared_layers = shared_layers
        self.norm_coord_updates = norm_coord_updates
        self.coord_norm_const = coord_norm_const
        self.learn_skip_weights = learn_skip_weights

        # Model hyperparameter keyword arguments provided via the command line
        self.experiment_name = experiment_name  # What to name the current run
        self.use_wandb_logger = use_wandb_logger  # Whether to use WandB as the primary means of logging

        # Shorthand values for convenient reference
        self.ca_only = self.atom_selection_type == 'ca_atom'  # Whether to only update Ca atoms' positions
        self.using_pyg_graphs = graph_return_format == 'pyg'  # Whether PyTorch Geometric graphs are expected as input
        self.pdb_ext = 'pdb'  # The file extension to use for PDB files found within this pipeline

        # Assemble the layers of the network
        self.build_nn_modules()

        # Log hyperparameters
        self.save_hyperparameters()

    def build_nn_modules(self) -> None:
        """Define all layers for the chosen neural network module."""
        # Marshal all neural network modules
        if self.nn_type == 'EGR':
            nn_layers = [IterativeEquivariantGraphRefinementModel(
                num_iegr_layers=self.num_layers,
                num_iegr_iter=self.num_iter,
                num_hidden_feats=self.num_hidden_channels,
                num_edge_input_feats=self.num_edge_input_feats,
                num_attn_heads=self.num_attn_heads,
                num_atom_types=self.num_atom_types,
                atom_emb_dim=self.atom_emb_dim,
                use_surf_prox=self.use_surf_prox,
                use_dihedral_angles=self.use_dihedral_angles,
                use_edge_feats_in_iegr_layers=self.use_edge_feats_in_iegr_layers,
                use_dist_in_iegr_layers=self.use_dist_in_iegr_layers,
                use_local_attention=self.use_local_attention,
                use_global_attention=self.use_global_attention,
                attn_depth=self.attn_depth,
                update_pos_with_ca_atoms=self.update_pos_with_ca_atoms,
                knn=self.knn,
                batched_input=False,
                pooling=self.pooling,
                pred_struct_qual=True,
                dropout=self.dropout,
                nonlin=self.nonlin,
                layer_norm=self.layer_norm,
                layer_norm_coords=self.layer_norm_coords,
                skip_weight_h=self.skip_weight_h,
                x_connection_init=self.x_connection_init,
                leakyrelu_neg_slope=self.leakyrelu_neg_slope,
                shared_layers=self.shared_layers,
                norm_coord_updates=self.norm_coord_updates,
                coord_norm_const=self.coord_norm_const,
                learn_skip_weights=self.learn_skip_weights,
                use_fairscale=self.device_strategy in ['fsdp', 'ddp_sharded'],
                use_deepspeed='deepspeed' in self.device_strategy,
                ca_only=self.ca_only
            )]
        elif self.nn_type == 'SEGNN':
            input_irreps = Irreps(f'1x1o + {self.num_node_input_feats}x0e')
            edge_attr_irreps = Irreps.spherical_harmonics(self.lmax_attr)
            node_attr_irreps = Irreps.spherical_harmonics(self.lmax_attr)
            hidden_irreps = WeightBalancedIrreps(
                Irreps(f'{self.num_hidden_channels}x0e'),
                node_attr_irreps,
                sh=True,
                lmax=self.lmax_h
            )
            output_irreps_construction_str = '1x1o + 1x0e'
            output_irreps = Irreps(output_irreps_construction_str)
            additional_message_irreps = Irreps(f'{1 + self.num_edge_input_feats}x0e')

            nn_layers = [SteerableEquivariantGraphRefinementModel(
                input_irreps=input_irreps,
                hidden_irreps=hidden_irreps,
                output_irreps=output_irreps,
                edge_attr_irreps=edge_attr_irreps,
                node_attr_irreps=node_attr_irreps,
                num_layers=self.num_layers,
                norm=self.layer_norm,
                additional_message_irreps=additional_message_irreps,
                pos_skip_connection=self.pos_skip_connection,
                use_attention=self.use_global_attention,
                attention_depth=self.attn_depth,
                num_attention_heads=self.num_attn_heads
            )]
        elif self.nn_type == 'SET':
            nn_layers = [IterativeSE3Transformer(
                num_layers=self.num_layers,
                fiber_in=Fiber({0: self.num_node_input_feats}),
                fiber_hidden=Fiber.create(self.num_degrees, self.num_hidden_channels),
                fiber_out=Fiber({0: self.num_degrees * self.num_hidden_channels, 1: 1}),
                num_heads=self.num_attn_heads,
                channels_div=self.channels_div,
                num_degrees=self.num_degrees,
                num_channels=self.num_hidden_channels,
                knn=self.knn,
                batched_input=self.batched_input,
                num_iter=self.num_iter,
                compute_gradients=True,
                fiber_edge=Fiber({0: self.num_edge_input_feats}),
                pooling=self.pooling,
                pred_struct_qual=True,
                norm=self.norm,
                use_layer_norm=self.use_layer_norm,
                tensor_cores=using_tensor_cores(self.amp),  # Use Tensor Cores more effectively
                low_memory=self.low_memory,  # Note: Currently, low memory mode is not compatible with FP16 and AMP
                amp=self.amp
            )]
        else:
            raise NotImplementedError(f'Selected module type {self.nn_type} is not currently supported')
        self.nn_module = nn.ModuleList(nn_layers)

    # ---------------------
    # Training
    # ---------------------
    def shared_forward(self, graph: dgl.DGLGraph, pdb_filepaths: List[str]) -> dgl.DGLGraph:
        """Make a forward pass through a standard neural network module."""
        # Initialize metadata for batched input graphs
        batch_num_nodes, batch_num_edges = None, None

        # Forward propagate with each layer
        for layer in self.nn_module:
            # Cache the original batch number of nodes and edges
            if not self.using_pyg_graphs:
                batch_num_nodes = graph.batch_num_nodes()
                batch_num_edges = graph.batch_num_edges()

            # Perform an iteration of information updates
            graph = layer(graph, pdb_filepaths)

            # Retain the original batch number of nodes and edges
            if not self.using_pyg_graphs:
                graph.set_batch_num_nodes(batch_num_nodes)
                graph.set_batch_num_edges(batch_num_edges)

        return graph  # Return the updated graph

    def shared_step(self, graph: dgl.DGLGraph, pdb_filepaths: List[str]) -> dgl.DGLGraph:
        """Make a forward pass through the entire network."""
        # Learn to refine the input graph's atomic coordinates and estimate the quality of the refined coordinates
        graph = self.shared_forward(graph, pdb_filepaths)

        # Return updated graph
        return graph

    def training_step(self, *args, **kwargs) -> pl.utilities.types.STEP_OUTPUT:
        """Lightning calls this inside the training loop."""
        pass

    # ---------------------
    # Evaluation
    # ---------------------
    def validation_step(self, *args, **kwargs) -> Optional[pl.utilities.types.STEP_OUTPUT]:
        """Lightning calls this inside the validation loop."""
        pass

    # ---------------------
    # Testing
    # ---------------------
    def test_step(self, *args, **kwargs) -> Optional[pl.utilities.types.STEP_OUTPUT]:
        """Lightning calls this inside the testing loop."""
        pass

    # ---------------------
    # Inference
    # ---------------------
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        """Lightning calls this inside the prediction loop."""
        # Make predictions for a batch of proteins
        graph_batch, filepaths = batch[0], batch[1]

        # Ensure input graph is on the same device as the LightningModule
        graph_batch = graph_batch.to(self.device)

        # Forward propagate with network layers
        graph_batch = self.shared_step(graph_batch, filepaths)

        # Return updated graph and corresponding filepaths
        return graph_batch, filepaths

    # ---------------------
    # Training Setup
    # ---------------------
    def configure_optimizers(self):
        """Called to configure the trainer's optimizer(s)."""
        pass

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        # -----------------------------------------------------
        # Iterative SE(3)-Transformer arguments
        # -----------------------------------------------------
        parser.add_argument('--channels_div', type=int, default=2,
                            help='Channels division before feeding to attention layer within the ISET model')
        parser.add_argument('--num_degrees',
                            help='Number of degrees to use within the ISET model. '
                                 'Hidden features will have types [0, ..., num_degrees - 1]',
                            type=int, default=2)
        parser.add_argument('--norm', type=str2bool, nargs='?', const=True, default=True,
                            help='Apply a normalization layer after each attention block within the ISET model')
        parser.add_argument('--use_layer_norm', type=str2bool, nargs='?', const=True, default=True,
                            help='Apply layer normalization between MLP layers within the ISET model')
        parser.add_argument('--low_memory', type=str2bool, nargs='?', const=True, default=False,
                            help='If true, will use fused ops that are slower but that use less memory '
                                 '(expect 25 percent less memory). '
                                 'Only has an effect if AMP is enabled on Volta GPUs, or if running on Ampere GPUs. '
                                 'Only used within the ISET model')
        parser.add_argument('--amp', type=str2bool, nargs='?', const=True, default=False,
                            help='Whether to use Automatic Mixed Precision (AMP) within the ISET model')

        # ------------------------------------------------------
        # Iterative Equivariant Graph Refiner arguments
        # -----------------------------------------------------
        parser.add_argument('--num_atom_types', type=int, default=len(PROT_ATOM_NAMES),
                            help='Number of atom types to support within the IEGR model\'s initial type embedding')
        parser.add_argument('--atom_emb_dim', type=int, default=200,
                            help='Dimensionality to which to embed initial atom types within the IEGR model')
        parser.add_argument('--use_surf_prox', default=False, action='store_true',
                            help='Whether to concatenate surface proximities to atom embeddings within the IEGR model')
        parser.add_argument('--use_dihedral_angles', default=False, action='store_true',
                            help='Whether to concatenate dihedral angles to atom embeddings within the IEGR model')
        parser.add_argument('--use_edge_feats_in_iegr_layers', default=False, action='store_true',
                            help='Whether to initialize edge representations within the IEGR model '
                                 'with precomputed features such as `bond_type` and `in_same_chain`')
        parser.add_argument('--use_dist_in_iegr_layers', default=False, action='store_true',
                            help='Whether to include distance edge features in message-passing within the IEGR model')
        parser.add_argument('--use_local_attention', default=False, action='store_true',
                            help='Whether to compute node-wise attention scores node-locally within the IEGR model')
        parser.add_argument('--use_global_attention', default=False, action='store_true',
                            help='Whether to compute node-wise attention scores node-globally within the IEGR model')
        parser.add_argument('--attn_depth', default=1, type=int,
                            help='How many global attention layers to apply within the IEGR model')
        parser.add_argument('--update_pos_with_ca_atoms', default=False, action='store_true',
                            help='Whether to update the position of non-Ca atoms using their Ca atom displacements')
        parser.add_argument('--nonlin', type=str, default='lkyrelu',
                            choices=['swish', 'tanh', 'prelu', 'elu', 'lkyrelu'],
                            help='Which nonlinearity to apply within IEGR layers')
        parser.add_argument('--layer_norm_coords', type=str, default='0', choices=['0', 'LN'],
                            help='Which version of normalization to apply to each set of IEGR-updated coordinates')
        parser.add_argument('--skip_weight_h', type=float, default=0.5,
                            help='Skip weight to apply to new node representations within the IEGR model')
        parser.add_argument('--x_connection_init', type=float, default=0.5,
                            help='Initial connection to node coordinates within the IEGR model')
        parser.add_argument('--leakyrelu_neg_slope', type=float, default=1e-2,
                            help='Value for the LeakyReLU function\'s negative slope within the IEGR model')
        parser.add_argument('--shared_layers', default=False, action='store_true',
                            help='Whether to use the same set of weights for all intermediate network layers')
        parser.add_argument('--norm_coord_updates', default=False, action='store_true',
                            help='Whether to normalize coordinate updates by their L2-norm plus a constant')
        parser.add_argument('--coord_norm_const', type=float, default=1.0,
                            help='A normalizing constant for coordinate updates')
        parser.add_argument('--learn_skip_weights', default=False, action='store_true',
                            help='Whether to learn skip connection weights')

        # -----------------------------------------------------
        # Steerable Equivariant Graph Neural Network arguments
        # -----------------------------------------------------
        parser.add_argument('--lmax_attr', type=int, default=3,
                            help='Max degree of geometric attribute embedding within the SEGNN model')
        parser.add_argument('--lmax_h', type=int, default=2,
                            help='Max degree of hidden representation within the SEGNN model')
        parser.add_argument('--pos_skip_connection', default=False, action='store_true',
                            help='Whether to connect to input graphs\' original node positions within the SEGNN model')

        return parser
