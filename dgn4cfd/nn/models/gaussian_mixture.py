import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from copy import deepcopy

from .multi_scale_gnn import MultiScaleGnn
from ..model import Model
from ...graph import Graph
from ...loader import Collater


class GaussianMixtureGnn(Model):
    """GNN-based Gaussian Mixture Model.

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
        self.out_node_features  = arch.get('out_node_features')
        self.num_gaussians      = arch['num_gaussians']
        self.depths             = arch['depths']
        self.fnns_depth         = arch.get('fnns_depth', 2)
        self.fnns_width         = arch['fnns_width']
        self.aggr               = arch.get('aggr', 'mean')
        self.dropout            = arch.get('dropout', 0.0)
        self.dim                = arch.get('dim', 2)
        self.scalar_rel_pos     = arch.get('scalar_rel_pos', True)
        if 'in_edge_features' in arch: # To support backward compatibility
             self.cond_edge_features = arch['in_edge_features'] + self.cond_edge_features
        # Validate the inputs
        assert self.in_node_features >= 0, "Input node features must be a non-negative integer"
        assert self.cond_edge_features >= 0, "Input edge condition features must be a non-negative integer"
        assert self.num_gaussians > 0, "Number of Gaussians must be a positive integer"
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
            activation     = nn.SELU,
            dropout        = self.dropout,
            dim            = self.dim,
            scalar_rel_pos = self.scalar_rel_pos,
        )
        # Node decoder
        self.node_decoder = nn.Linear(
            in_features  = self.fnns_width,
            out_features = 3 * self.out_node_features * self.num_gaussians,
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
        # Get the parameters of the Gaussian mixture
        pi, mean, var = torch.chunk(v, 3, dim=1)
        pi, mean, var = pi.view(-1, self.num_fields, self.num_gaussians), mean.view(-1, self.num_fields, self.num_gaussians), var.view(-1, self.num_fields, self.num_gaussians)
        # Apply the softmax to the pi values and the exponential to the variances
        pi  = F.softmax(pi, dim=-1) 
        var = F.elu(var) + 1 + 1e-6
        return pi, mean, var # Shape (num_nodes, num_fields, num_gaussians)
    
    @torch.no_grad()
    def sample(
        self,
        graph: Graph,
        dirichlet_values: torch.Tensor = None,
        seed:  int = None,
    ) -> torch.Tensor:
        """Sample from the Gaussian Mixture Model.

        Args:
            graph (Graph): Graph object with the input features.
            dirichlet_values (torch.Tensor, optional): Field values at the Dirichlet boundary nodes. Defaults to None. Dimension (num_nodes, num_fields).
            seed (int, optional): Random seed. Defaults to None.
        """
        self.eval()
        graph = graph.to(self.device)
        if seed is not None:
            torch.manual_seed(seed)
        pi, mean, var = self(graph) # Shape (num_nodes, num_fields, num_gaussians)
        std = torch.sqrt(var)
        # Sample from the categorical distribution defined by pi
        categorical = torch.distributions.Categorical(pi)  # Shape (num_nodes, num_fields)
        component_indices = categorical.sample().unsqueeze(-1)  # Shape (num_nodes, num_fields, 1)
        # Gather the corresponding mean and std based on the sampled component
        selected_mean = torch.gather(mean, -1, component_indices).squeeze(-1) # Shape (num_nodes, num_fields)
        selected_std  = torch.gather(std,  -1, component_indices).squeeze(-1) # Shape (num_nodes, num_fields)
        # Sample from the selected Gaussian component
        eps = torch.randn_like(selected_mean)  # Shape (num_nodes, num_fields)
        z = selected_mean + selected_std * eps # Shape (num_nodes, num_fields)
        # Apply the dirichlet boundary condition (if it exists)
        if hasattr(graph, 'dirichlet_mask'):
            assert dirichlet_values is not None
            z = torch.where(graph.dirichlet_mask, dirichlet_values.to(self.device), z)
        return z # Shape (num_nodes, num_fields)
    
    @torch.no_grad()    
    def sample_n(
        self,
        num_samples:      int,
        graph:            Graph,
        dirichlet_values: torch.Tensor = None,
        batch_size:       int = 0,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Sample `num_samples` samples from the model.

        Args:
            num_samples (int): The number of samples.
            graph (Graph): The graph.
            dirichlet_values (torch.Tensor, optional): The Dirichlet boundary conditions. If `None`, no Dirichlet boundary conditions are applied. Defaults to `None`.
            batch_size (int, optional): Number of samples to generate in parallel. If `batch_size < 2`, the samples are generated one by one. Defaults to `0`.

        Returns:
            torch.Tensor: The samples. Dimension: (num_nodes, num_samples, num_fields
        """
        samples = []
        # Create (num_samples // num_workers) mini-batches with the same graph repeated num_workers times
        if batch_size > 1:
            collater = Collater()
            num_evals = num_samples // batch_size + (num_samples % batch_size > 0)
            for _ in tqdm(range(num_evals), desc=f"Generating {num_samples} samples", leave=False, position=0):
                current_batch_size = min(batch_size, num_samples - len(samples))
                batch = collater.collate([deepcopy(graph) for _ in range(current_batch_size)])
                # Sample
                sample = self.sample(batch, dirichlet_values.repeat(current_batch_size, 1) if dirichlet_values is not None else None, *args, **kwargs)
                # Split base on the batch index
                sample = torch.stack(sample.chunk(current_batch_size, dim=0), dim=1)
                samples.append(sample)
            return torch.cat(samples, dim=1)
        else:
            for _ in tqdm(range(num_samples), desc=f"Generating {num_samples} samples", leave=False, position=0):
                sample = self.sample(graph, dirichlet_values, *args, **kwargs)
                samples.append(sample)
            return torch.stack(samples, dim=1) # Dimension: (num_nodes, num_samples, num_fields)
    