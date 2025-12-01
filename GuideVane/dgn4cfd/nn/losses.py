import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.utils import scatter

from .diffusion.diffusion_model import DiffusionModel
from .flow_matching import FlowMatchingModel
from .model import Model
from .. import Graph


class MseLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        model: Model,
        graph: Graph,
    ) -> torch.Tensor:
        # Inference
        output = model(graph)
        if isinstance(output, tuple):
            output = output[0]
        # Check the shapes
        assert output.shape == graph.target.shape, f'output.shape = {output.shape}, target.shape = {graph.target.shape}'
        # Compute the loss
        loss = F.mse_loss(output,  graph.target)
        return loss
    

class VaeLoss(nn.Module):
    def __init__(self, kl_reg: float = 0.0):
        super().__init__()
        self.kl_reg = kl_reg # KL regularization coefficient

    def forward(
            self,
            model: Model,
            graph: Graph,
    ) -> torch.Tensor:
        v, _, mean, logvar = model(graph)
        loss = {}
        loss['reconstruction_loss'] = F.mse_loss(v, graph.target)
        if self.kl_reg > 0.0:
            kl_divergence = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
            loss['kl_divergence'] = self.kl_reg * kl_divergence
        # Add all the terms
        return sum(loss.values())


class GaussianNLLLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        model: Model,
        graph: Graph
    ) -> torch.Tensor:
        # Inference
        output = model(graph)
        mean, logvar = output.chunk(2, dim=1)
        assert mean.size(1) == logvar.size(1), f'mean.size(1) = {mean.size(1)}, logvar.size(1) = {logvar.size(1)}'
        # Compute the loss
        var = torch.exp(logvar)
        assert (var > 0).all(), f'var = {var}'
        return F.gaussian_nll_loss(mean, graph.target, var, full=True)
    

class BayesianLoss(nn.Module):
    def __init__(self, kl_reg) -> None:
        super().__init__()
        self.kl_reg = kl_reg

    def forward(
        self,
        model: Model,
        graph: Graph,
    ) -> torch.Tensor:
        # Inference
        output = model(graph)
        if isinstance(output, tuple):
            output = output[0]
        # Check the shapes
        assert output.shape == graph.target.shape, f'output.shape = {output.shape}, target.shape = {graph.target.shape}'
        # Compute the loss
        mse = F.mse_loss(output, graph.target)
        # Get the KL divergence
        kl = model.kl_loss()
        return mse + self.kl_reg * kl


class GmmLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        model: Model,
        graph: Graph,
    ) -> torch.Tensor:
        # Inference
        pi, mean, var = model(graph)  # Shape: (num_nodes, num_fields, num_gaussians)
        target = graph.target.unsqueeze(-1)  # Shape: (num_nodes, num_fields, 1)
        assert target.shape == (graph.num_nodes, model.num_fields, 1), f'target.shape = {target.shape}'
        # Compute the probability
        log_prob = -0.5 * ((mean - target) ** 2 / var) - 0.5 * torch.log(2 * np.pi * var)
        # Compute the log-sum-exp for the mixture components
        log_pi_prob = torch.log(pi) + log_prob
        log_sum_exp = torch.logsumexp(log_pi_prob, dim=-1) # Shape: (num_nodes, num_fields)
        # Compute the negative log-likelihood
        loss = -log_sum_exp # Shape: (num_nodes, num_fields)
        return loss.mean()  
    

class HybridLoss(nn.Module):
    """Hybrid loss function for diffusion models from the paper Improved Denoising Diffusion Probabilistic Models (https://arxiv.org/abs/2102.09672)."""

    def __init__(
        self,
        lambda_vlb: float = 0.001,
    ) -> None:
        super().__init__()
        self.lambda_vlb = lambda_vlb

    def forward(
        self,
        model: DiffusionModel,
        graph: Graph,
    ) -> torch.Tensor:
        assert model.learnable_variance, 'HybridLoss requires learnable_variance = `True`'
        true_posterior_mean, true_posterior_variance = model.diffusion_process.get_posterior_mean_and_variance(graph.field_start, graph.field_r, graph.batch, graph.r)
        # Compute the model output and the MSE loss between eps and noise
        model_noise, model_v = model(graph)
        se = (model_noise - graph.noise)**2 # Dimension: (num_nodes, num_output_features)
        mse_term = batch_wise_mean(se, graph.batch) # Dimension: (batch_size)
        # Freeze eps and compute the VLB loss
        frozen_output = (model_noise.detach(), model_v)
        model_posterior_mean, model_posterior_variance = model.get_posterior_mean_and_variance_from_output(frozen_output, graph)
        vlb_term = self.lambda_vlb * self.vlb_loss(graph, true_posterior_mean, true_posterior_variance, model_posterior_mean, model_posterior_variance) # Dimension: (batch_size)
        return mse_term + vlb_term # Dimension: (batch_size)
    
    @staticmethod
    def vlb_loss(
        graph: Graph,
        true_posterior_mean:      torch.Tensor,
        true_posterior_variance:  torch.Tensor,
        model_posterior_mean:     torch.Tensor,
        model_posterior_variance: torch.Tensor
    ) -> torch.Tensor:
        """Compute the KL divergence between the true and the model posterior"""
        kl = normal_kl_divergence(true_posterior_mean, true_posterior_variance, model_posterior_mean, model_posterior_variance) # Dimension: (num_nodes, num_features)
        kl = batch_wise_mean(kl, graph.batch)/math.log(2.0) # Dimension (batch_size)
        if (graph.r == 0).any():
            decoder_nll = F.gaussian_nll_loss(model_posterior_mean, graph.field_start, model_posterior_variance, reduction='none') # Dimension: (num_nodes, num_features)
            decoder_nll = batch_wise_mean(decoder_nll, graph.batch) # Dimension (batch_size)
            kl = torch.where((graph.r == 0), decoder_nll, kl)
        return kl


def normal_kl_divergence(
    mean1:     torch.Tensor,
    variance1: torch.Tensor,
    mean2:     torch.Tensor,
    variance2: torch.Tensor
) -> torch.Tensor:
    """Compute the KL divergence between two normal distributions."""
    return 0.5 * (torch.log(variance2) - torch.log(variance1) + variance1 / variance2 + (mean1 - mean2)**2 / variance2 - 1)


def batch_wise_mean(
    field: torch.Tensor,
    batch: torch.LongTensor,
) -> torch.Tensor:
    r"""Compute the batch-wise mean of a field.
    
        Args:
            field (Tensor): The field. Dimension: (num_nodes, num_features).
            batch (LongTensor): The batch vector. Dimension: (num_nodes).

        Returns:
            Tensor: The batch-wise mean. Dimension: (batch_size).
    """
    assert field.dim() == 1 or field.dim() == 2, 'field must be one- or two-dimensional'
    if field.dim() == 2:
        field = field.mean(dim=1) # Dimension: (num_nodes)
    batch_size = batch.max().item() + 1
    return scatter(field, batch, dim=0, dim_size=batch_size, reduce='mean') # Dimension: (batch_size)


class FlowMatchingLoss(nn.Module):
    r"""Loss function for the flow matching model.
    
        Args:
            model (FlowMatchingModel): The flow matching model.
            graph (Graph): The input graph.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        model: FlowMatchingModel,
        graph: Graph,
    ) -> torch.Tensor:
        # Inference
        pred_v = model(graph) # This is the predicted advection field
        # Check the shapes
        assert pred_v.shape == graph.advection_field.shape, f'output.shape = {pred_v.shape}, advection_field.shape = {graph.advection_field.shape}'
        # Compute the loss
        return batch_wise_mean((pred_v - graph.advection_field)**2, graph.batch) # Dimension: (batch_size)