import torch
from torch import nn

from .multi_scale_gnn import MultiScaleGnn
from ..model import Model
from ...graph import Graph


class VanillaGnn(Model):
    """Deterministic GNN model.

    Args:
        arch (dict): Dictionary with the architecture of the model. It must contain the following keys:
            - 'in_node_features' (int): Number of input node features. This is the number of features of the noisy field.
            - 'cond_node_features' (int, optional): Number of conditional node features. Defaults to 0.
            - 'cond_edge_features' (int, optional): Number of conditional edge features. Defaults to 0.
            - 'in_edge_features' (int, optional): Number of input edge features. Defaults to 0.
            - 'depths' (list): List of integers with the number of layers at each depth.
            - 'fnns_depth' (int, optional): Number of layers in the FNNs. Defaults to 2.
            - 'fnns_width' (int): Width of the FNNs.
            - 'aggr' (str, optional): Aggregation method. Defaults to 'mean'.
            - 'dropout' (float, optional): Dropout probability. Defaults to 0.0.
            - 'activation' (torch.nn.Module, optional): Activation function. Defaults to torch.nn.SELU.
            - 'pooling_method' (str, optional): Pooling method. Defaults to 'interp'.
            - 'unpooling_method' (str, optional): Unpooling method. Defaults to 'uniform'.
            - 'dim' (int, optional): Dimension of the latent space. Defaults to 2.
            - 'scalar_rel_pos' (bool, optional): Whether to use scalar relative positions. Defaults to True.    
    """

    def __init__(self, *args, **kwargs):  
        super().__init__(*args, **kwargs)

    def load_arch(self, arch: dict):
        self.arch = arch
        # Hyperparameters
        self.in_node_features   = arch['in_node_features']
        self.cond_node_features = arch.get('cond_node_features', 0)
        self.cond_edge_features = arch.get('cond_edge_features', 0)
        self.out_node_features  = arch.get('out_node_features', self.in_node_features)
        self.depths             = arch['depths']
        self.fnns_depth         = arch.get('fnns_depth', 2)
        self.fnns_width         = arch['fnns_width']
        self.aggr               = arch.get('aggr', 'mean')
        self.dropout            = arch.get('dropout', 0.0)
        self.activation         = arch.get('activation', nn.SELU)
        self.pooling_method     = arch.get('pooling_method',    'interp')
        self.unpooling_method   = arch.get('unpooling_method', 'uniform')
        self.dim                = arch.get('dim', 2)
        self.scalar_rel_pos     = arch.get('scalar_rel_pos', True)
        if 'in_edge_features' in arch: # To support backward compatibility
             self.cond_edge_features = arch['in_edge_features'] + self.cond_edge_features
        # Validate the inputs
        assert self.in_node_features >= 0, "Input node features must be a non-negative integer"
        assert self.cond_edge_features >= 0, "Input edge condition features must be a non-negative integer"
        assert len(self.depths) > 0, "Depths (`depths`) must be a list of integers"
        assert isinstance(self.depths, list), "Depths (`depths`) must be a list of integers"
        assert all([isinstance(depth, int) for depth in self.depths]), "Depths (`depths`) must be a list of integers"
        assert all([depth > 0 for depth in self.depths]), "Depths (`depths`) must be a list of positive integers"
        assert self.fnns_depth >=2 , "FNNs depth (`fnns_depth`) must be at least 2"
        assert self.fnns_width > 0, "FNNs width (`fnns_width`) must be a positive integer"
        assert self.aggr in ('mean', 'sum'), "Aggregation method (`aggr`) must be either 'mean' or 'sum'"
        assert self.dropout >= 0.0 and self.dropout < 1.0, "Dropout (`dropout`) must be a float between 0.0 and 1.0"
        # Node encoder
        self.node_encoder = nn.Linear(
            in_features  = self.in_node_features + self.cond_node_features,
            out_features = self.fnns_width,
        )
        # Edge encoder
        self.edge_encoder = nn.Linear(
            in_features  = self.cond_edge_features,
            out_features = self.fnns_width,
        )
        # MuS-GNN propagator: It propagates the scalar latent features
        self.propagator = MultiScaleGnn(
            depths         = self.depths,
            fnns_depth     = self.fnns_depth,
            fnns_width     = self.fnns_width,
            aggr           = self.aggr,
            activation     = self.activation,
            dropout        = self.dropout,
            dim            = self.dim,
            scalar_rel_pos = self.scalar_rel_pos,
        )
        # Node decoder
        self.node_decoder = nn.Linear(
            in_features  = self.fnns_width,
            out_features = self.out_node_features,
        )
 
    @property
    def num_fields(self) -> int:
        return self.out_node_features
    
    def reset_parameters(self):
            modules = [module for module in self.children() if hasattr(module, 'reset_parameters')]
            for module in modules:
                module.reset_parameters()

    def forward(
        self,
        graph: Graph,
    ) -> torch.Tensor:
        # Encode the node features
        v = self.node_encoder(
            torch.cat([
                *[f for f in [graph.get('field'), graph.get('loc'), graph.get('glob'), graph.get('omega')] if f is not None],
            ], dim=1)
        ) # Shape (num_nodes, fnns_width)
        # Encode the edge features
        e = self.edge_encoder(
            torch.cat([
                graph.edge_attr,
                *[f for f in [graph.get('edge_cond')] if f is not None],
            ], dim=1)
        )
        # Propagate the scalar latent space (conditioned on c)
        v, _ = self.propagator(graph, v, e)
        # Decode the latent node features
        v = self.node_decoder(v)
        # Return the output
        return v