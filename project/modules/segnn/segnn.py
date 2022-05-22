import numpy as np
import torch
import torch.nn as nn
from e3nn.nn import BatchNorm
from e3nn.o3 import Irreps
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from .instance_norm import InstanceNorm
from .o3_building_blocks import O3TensorProduct, O3TensorProductSwishGate
from ...utils.segnn.utils import LinearAttentionTransformer


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from SEGNN (https://github.com/RobDHess/Steerable-E3-GNN):
# -------------------------------------------------------------------------------------------------------------------------------------
class SEGRLayer(MessagePassing):
    """Steerable E(3) equivariant graph structure refinement layer as a PyTorch Geometric MessagePassing Module."""

    def __init__(
            self,
            input_irreps: Irreps,
            hidden_irreps: Irreps,
            edge_attr_irreps: Irreps,
            node_attr_irreps: Irreps,
            norm: str = None,
            additional_message_irreps: Irreps = None,
            use_attention: bool = False,
            attention_depth: int = 1,
            num_attention_heads: int = 4
    ):
        """Steerable E(3) Equivariant Graph Structure Refinement Layer

        Parameters
        ----------
        input_irreps: Irreps
            Input irreducible representations.
        hidden_irreps: Irreps
            Intermediate irreducible representations.
        edge_attr_irreps: Irreps
            Edge input irreducible representations.
        node_attr_irreps: Irreps
            Node input irreducible representations.
        norm: str
            Feature normalization scheme to apply within message-passing layers.
        additional_message_irreps: int
            Irreducible representations of extra message features.
        use_attention: bool
            Whether to use a multi-head attention mechanism to learn new node representations with self-attention.
        attention_depth: int
            How many layers of multi-head attention to apply to type-0 node representations in each SEGR layer.
        num_attention_heads: int
            Number of attention heads to employ in multi-head attention when `use_attention` is True.
        """
        super().__init__(node_dim=-2, aggr="add")
        self.hidden_irreps = hidden_irreps
        self.use_attention = use_attention

        if use_attention:
            self.type_0_slice = self.hidden_irreps.slices()[0]
            self.num_attention_input_feats = self.type_0_slice.stop - self.type_0_slice.start
            assert self.num_attention_input_feats % num_attention_heads == 0, \
                f'{self.num_attention_input_feats % num_attention_heads} != 0'
            self.attention_module = LinearAttentionTransformer(
                dim=self.num_attention_input_feats,
                heads=num_attention_heads,
                depth=attention_depth,
                max_seq_len=12288
            )

            update_input_type_0_irreps = Irreps(f'{self.num_attention_input_feats}x0e')
            update_input_type_0_irreps = (input_irreps + hidden_irreps + update_input_type_0_irreps)
        else:
            update_input_type_0_irreps = input_irreps + hidden_irreps

        message_input_irreps = (2 * input_irreps + additional_message_irreps).simplify()
        update_input_irreps = update_input_type_0_irreps.simplify()

        self.message_layer_1 = O3TensorProductSwishGate(
            message_input_irreps, hidden_irreps, edge_attr_irreps
        )
        self.message_layer_2 = O3TensorProductSwishGate(
            hidden_irreps, hidden_irreps, edge_attr_irreps
        )
        self.update_layer_1 = O3TensorProductSwishGate(
            update_input_irreps, hidden_irreps, node_attr_irreps
        )
        self.update_layer_2 = O3TensorProduct(
            hidden_irreps, hidden_irreps, node_attr_irreps
        )

        self.setup_normalisation(norm)

    def setup_normalisation(self, norm: str):
        """Set up feature normalization, either with batch or instance normalization."""
        self.norm = norm
        self.feature_norm = None
        self.message_norm = None

        if norm == 'BN':
            self.feature_norm = BatchNorm(self.hidden_irreps)
            self.message_norm = BatchNorm(self.hidden_irreps)
        elif norm == 'IN':
            self.feature_norm = InstanceNorm(self.hidden_irreps)

    def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
            node_attr: torch.Tensor,
            batch,
            additional_message_features: torch.Tensor = None,
    ):
        """Propagate messages along edges."""
        x = self.propagate(
            edge_index,
            x=x,
            node_attr=node_attr,
            edge_attr=edge_attr,
            additional_message_features=additional_message_features
        )

        # Normalize features
        if self.feature_norm:
            if self.norm == 'BN':
                x = self.feature_norm(x)
            elif self.norm == 'IN':
                x = self.feature_norm(x, batch)
        return x

    def message(self, x_i, x_j, edge_attr, additional_message_features):
        """Create messages."""
        if additional_message_features is None:
            input = torch.cat((x_i, x_j), dim=-1)
        else:
            input = torch.cat((x_i, x_j, additional_message_features), dim=-1)

        message = self.message_layer_1(input, edge_attr)
        message = self.message_layer_2(message, edge_attr)

        if self.message_norm:
            message = self.message_norm(message)
        return message

    def update(self, message, x, node_attr):
        """Update note features."""
        if self.use_attention:
            attention_input_x = x[:, self.type_0_slice.start:self.type_0_slice.stop].clone()  # Only type-0 feats
            attention_input_x = attention_input_x.unsqueeze(0)
            attention_message = self.attention_module(attention_input_x).reshape(-1, self.num_attention_input_feats)
            input = torch.cat((x, message, attention_message), dim=-1)
        else:
            input = torch.cat((x, message), dim=-1)
        update = self.update_layer_1(input, node_attr)
        update = self.update_layer_2(update, node_attr)
        x += update  # Residual connection
        return x


class SEGR(nn.Module):
    """Steerable E(3) equivariant graph structure refinement as a PyTorch Module.
    SEGR stands for E(3) equivariant graph structure refiner. It is the
    equivalent of a series of (non-)linear layers in an MLP or convolution
    layers in a CNN.
    """

    def __init__(
            self,
            input_irreps: Irreps,
            hidden_irreps: Irreps,
            output_irreps: Irreps,
            edge_attr_irreps: Irreps,
            node_attr_irreps: Irreps,
            num_layers: int,
            norm: str = None,
            additional_message_irreps: Irreps = None,
            use_attention: bool = False,
            attention_depth: int = 1,
            num_attention_heads: int = 4
    ):
        """Steerable E(3) Equivariant Graph Structure Refinement Module

        Parameters
        ----------
        input_irreps: int
            Number of input irreducible representations.
        hidden_irreps: int
            Number of learned irreducible representations.
        output_irreps: int
            Number of output irreducible representations.
        edge_attr_irreps: Irreps
            Number of edge input irreducible representations.
        node_attr_irreps: Irreps
            Number of node input irreducible representations.
        num_layers: int
            Number of message-passing layers to employ.
        norm: str
            Feature normalization scheme to apply within message-passing layers.
        additional_message_irreps: Irreps
            Irreducible representations of extra message features.
        use_attention: bool
            Whether to use a multi-head attention mechanism to learn new node representations with self-attention.
        attention_depth: int
            How many layers of multi-head attention to apply to type-0 node representations in each SEGR layer.
        num_attention_heads: int
            Number of attention heads to employ in multi-head attention when `use_attention` is True.
        """
        super().__init__()
        # Create network, initializing its input embedding first
        self.embedding_layer = O3TensorProduct(
            input_irreps, hidden_irreps, node_attr_irreps
        )

        # Curate message passing layers
        layers = []
        for i in range(num_layers):
            layers.append(
                SEGRLayer(
                    hidden_irreps,
                    hidden_irreps,
                    edge_attr_irreps,
                    node_attr_irreps,
                    norm=norm,
                    additional_message_irreps=additional_message_irreps,
                    use_attention=use_attention,
                    attention_depth=attention_depth,
                    num_attention_heads=num_attention_heads
                )
            )
        self.layers = nn.ModuleList(layers)

        self.post_proc1 = O3TensorProductSwishGate(
            hidden_irreps, hidden_irreps, node_attr_irreps
        )
        self.post_proc2 = O3TensorProduct(
            hidden_irreps, output_irreps, node_attr_irreps
        )

    @staticmethod
    def catch_isolated_nodes(graph):
        """Ensure that isolated nodes also contain attributes."""
        if graph.has_isolated_nodes() and graph.edge_index.max().item() + 1 != graph.num_nodes:
            nr_add_attr = graph.num_nodes - (graph.edge_index.max().item() + 1)
            add_attr = graph.node_attr.new_tensor(
                np.zeros((nr_add_attr, graph.node_attr.shape[-1]))
            )
            graph.node_attr = torch.cat((graph.node_attr, add_attr), -2)
        # Trivial irrep value should always be 1 (is automatically so for connected nodes, but isolated nodes are now 0)
        graph.node_attr[:, 0] = 1.0

    def forward(self, graph: Data) -> Data:
        """Conduct an SEGR forward pass."""
        # Assemble input arguments
        x, pos, edge_index, edge_attr, node_attr, batch = (
            graph.x,
            graph.pos,
            graph.edge_index,
            graph.edge_attr,
            graph.node_attr,
            graph.batch,
        )
        try:
            additional_message_features = graph.additional_message_features
        except AttributeError:
            additional_message_features = None

        # Make sure that isolated nodes contain at least a trivial irrep value
        self.catch_isolated_nodes(graph)

        # Embed initial node features
        x = self.embedding_layer(x, node_attr)

        # Pass messages across the graph
        for layer in self.layers:
            x = layer(
                x, edge_index, edge_attr, node_attr, batch, additional_message_features
            )

        # Postprocess node representations to get type-l output(s) (e.g., that could represent new node positions)
        x = self.post_proc1(x, node_attr)
        x = self.post_proc2(x, node_attr)

        # Return resulting node representations
        return x


class SteerableEquivariantGraphRefinementModel(nn.Module):
    """Steerable E(3) equivariant graph structure refinement as a PyTorch Module."""

    def __init__(
            self,
            input_irreps: Irreps,
            hidden_irreps: Irreps,
            output_irreps: Irreps,
            edge_attr_irreps: Irreps,
            node_attr_irreps: Irreps,
            num_layers=4,
            norm='IN',
            additional_message_irreps=Irreps("16x0e"),
            pos_skip_connection=True,
            use_attention=False,
            attention_depth: int = 1,
            num_attention_heads: int = 4
    ):
        """Steerable E(3) Equivariant Graph Structure Refinement Model

        Parameters
        ----------
        input_irreps: Irreps
            Input irreducible representations.
        hidden_irreps: Irreps
            Intermediate irreducible representations.
        output_irreps: Irreps
            Output irreducible representations.
        edge_attr_irreps: Irreps
            Edge input irreducible representations.
        node_attr_irreps: Irreps
            Node input irreducible representations.
        num_layers: int
            Number of message-passing layers to employ.
        norm: str
            Feature normalization scheme to apply within message-passing layers.
        additional_message_irreps: Irreps
            Irreducible representations for extra message features.
        pos_skip_connection: bool
            Whether to make a skip connection back to the input graph's original node positions.
        use_attention: bool
            Whether to use a multi-head attention mechanism to learn new node representations with self-attention.
        attention_depth: int
            How many layers of multi-head attention to apply to type-0 node representations in each SEGR layer.
        num_attention_heads: int
            Number of attention heads to employ in multi-head attention when `use_attention` is True.
        """
        super().__init__()
        self.pos_skip_connection = pos_skip_connection
        self.output_irreps_repr_splits = output_irreps.__repr__().split('+')

        self.segr = SEGR(
            input_irreps,
            hidden_irreps,
            output_irreps,
            edge_attr_irreps,
            node_attr_irreps,
            num_layers=num_layers,
            norm=norm,
            additional_message_irreps=additional_message_irreps,
            use_attention=use_attention,
            attention_depth=attention_depth,
            num_attention_heads=num_attention_heads
        )

    def forward(self, graph: Data, *args) -> Data:
        """
        Forward pass of the network.

        Parameters
        ----------
        graph: Data
            PyTorch Geometric input graph.

        Returns
        ----------
        Data
            PyTorch Geometric output graph.
        """
        # Perform message-passing with a SEGR module to get type-l tensors as output
        new_x = self.segr(graph)

        # Parse model outputs and store them at the corresponding dictionary entries
        can_update_pos = 'x1o' in self.output_irreps_repr_splits[0]
        if can_update_pos:
            # Predict single type-1 output for each node in the input graph (e.g., representing a positional update)
            type_1_outputs = new_x[:, :3]
            graph.pos = (graph.pos + type_1_outputs) if self.pos_skip_connection else type_1_outputs

        can_update_ndata1 = 'x0e' in self.output_irreps_repr_splits[0]
        can_update_ndata2 = len(self.output_irreps_repr_splits) > 1 and 'x0e' in self.output_irreps_repr_splits[1]
        if can_update_ndata1 or can_update_ndata2:
            # Predict single type-0 output for each node in the input graph (e.g., representing positional quality)
            if can_update_ndata1:
                type_0_outputs = new_x[:, 0]
                graph.ndata = type_0_outputs
            else:
                type_0_outputs = new_x[:, 3]
                graph.ndata = type_0_outputs

        # Return updated batch graph
        return graph

    def __repr__(self):
        return f'SteerableEquivariantGraphRefinementModel {str(self.__dict__)}'
