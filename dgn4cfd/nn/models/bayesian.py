import torch
from torch import nn
import bayesian_torch.layers as bnn_layers
from copy import deepcopy
from tqdm import tqdm

from .vanilla import VanillaGnn
from ..blocks import FNN
from ...graph import Graph
from ...loader import Collater


class BayesianGnn(VanillaGnn):
    """ Bayesian GNN model.

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
        super().load_arch(arch)
        # Transform the Linear layers in the encoder and decoder into BayesianLinear
        self.edge_encoder = self.BayesianLinear(self.edge_encoder)
        self.node_encoder = self.BayesianLinear(self.node_encoder)
        self.node_decoder = self.BayesianLinear(self.node_decoder)
        # Transform the Linear layers in the FNNs into BayesianLinear layers
        for module in self.modules():
            if isinstance(module, FNN):
                layers = module.layers
                for i, layer in enumerate(layers):
                    if isinstance(layer, nn.Linear):
                        module.layers[i] = self.BayesianLinear(module.layers[i])

    class BayesianLinear(bnn_layers.LinearReparameterization):
        def __init__(
            self,
            linear: nn.Linear,
        ) -> None:
            super().__init__(linear.in_features, linear.out_features)
            self.to(linear.weight.data.device)

        def forward(self, x):
            return super().forward(x, return_kl=False)
        
    def kl_loss(self):
        return torch.mean(torch.stack([module.kl_loss() for module in self.modules() if isinstance(module, bnn_layers.LinearReparameterization)]))
    
    @torch.no_grad()
    def sample(
        self, 
        graph: Graph,
        seed:  int = None,
    ) -> torch.Tensor:
        self.eval()
        graph = graph.to(self.device)
        if seed is not None:
            torch.manual_seed(seed)
        return self(graph)
    
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