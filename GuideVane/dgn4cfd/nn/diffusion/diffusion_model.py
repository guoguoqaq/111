import contextlib
import os
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from typing import List, Tuple, Union
from abc import abstractmethod
from tqdm import tqdm
from copy import deepcopy

from .diffusion_process import DiffusionProcess, DiffusionProcessSubSet
from .step_sampler import ImportanceStepSampler
from ..model import Model
from ...graph import Graph
from ...loader import DataLoader, Collater


class DiffusionModel(Model):
    r"""Abstract class for diffusion models. This class implements the training loop and the sampling method. The forward pass must be implemented in the derived class.

    Args:
        diffusion_process (DiffusionProcess, optional): The diffusion process. It must be provided if it we are not loading a checkpoint. Defaults to `None`.
        learnable_variance (bool, optional): If `True`, the variance is learnable. If we are loading a checkpoint, this value must be the same as the one in the checkpoint. 
            If `False`, the model output the noise, as in "Denoising Diffusion Probabilistic Models". If `True`, the model output the noise and the variance, as in "Improved Denoising Diffusion Probabilistic Models".
            Defaults to `False`.

    Methods:
        fit: Train the model using the provided training settings and data loader.
        forward: Forward pass of the model. This method must be implemented for each model.
        get_posterior_mean_and_variance_from_output: Compute the posterior mean and variance from the model output.
        sample: Sample from the model.
    """


    def __init__(
        self,
        diffusion_process:  DiffusionProcess = None,
        learnable_variance: bool             = None,
        *args, 
        **kwargs
    ) -> None:
        self.learnable_variance = learnable_variance
        super().__init__(*args, **kwargs)
        # This may overwrite the diffusion_process and learnable_variance if we are loading a checkpoint.
        # So, we need to check if the provided values are the same as the ones in the model.
        if diffusion_process is None and not hasattr(self, 'diffusion_process'):
            raise RuntimeError('diffusion_process must be provided')
        elif diffusion_process is not None:
            if hasattr(self, 'diffusion_process'):
                print('Warning: diffusion_process is provided, but it is already defined in the model.')
                print('The provided diffusion_process will be used.')
            self.diffusion_process = diffusion_process
        if learnable_variance is None and not hasattr(self, 'learnable_variance'):
            raise RuntimeError('learnable_variance must be provided')
        if learnable_variance is not None and learnable_variance != self.learnable_variance:
            raise RuntimeError('learnable_variance is different from the one in the provided checkpoint')
  
    @property
    def is_latent_diffusion(self):
        return hasattr(self, 'autoencoder')

    def fit(
        self,
        training_settings: object,
        dataloader:        DataLoader,
    ) -> None:
        """Train the model using the provided training settings and data loader.

        Args:
            training_settings (TrainingSettings): The training settings.
            dataloader (DataLoader): The data loader.
        """
        # Verify the training settings
        if training_settings['scheduler']['loss'][:3].lower() == 'val':
            raise NotImplementedError("Wrong training settings: Validation loss is not implemented yet.")
        # Change the training device if needed
        if training_settings['device'] is not None and training_settings['device'] != self.device:
            self.to(training_settings['device'])
            self.device = training_settings['device']
        # Set the diffusion step sampler
        step_sampler = training_settings['step_sampler'](num_diffusion_steps = self.diffusion_process.num_steps)
        # Set the training loss
        criterion = training_settings['training_loss']
        # Load checkpoint
        checkpoint = None
        scheduler  = None
        if training_settings['checkpoint'] is not None and os.path.exists(training_settings['checkpoint']):
            print("Training from an existing check-point:", training_settings['checkpoint'])
            checkpoint = torch.load(training_settings['checkpoint'], map_location=self.device)
            self.load_state_dict(checkpoint['weights'])
            optimiser = torch.optim.Adam(self.parameters(), lr=checkpoint['lr'])
            optimiser.load_state_dict(checkpoint['optimiser'])
            if training_settings['scheduler'] is not None: 
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=training_settings['scheduler']['factor'], patience=training_settings['scheduler']['patience'], eps=0.)
                scheduler.load_state_dict(checkpoint['scheduler'])
            initial_epoch = checkpoint['epoch'] + 1
        # Initialise optimiser and scheduler if not previous check-point is used
        else:
            # If a .chk is given but it does not exist such file, notify the user
            if training_settings['checkpoint'] is not None:
                print("Not matching check-point file:", training_settings['checkpoint'])
            print('Training from randomly initialised weights.')
            optimiser = optim.Adam(self.parameters(), lr=training_settings['lr'])
            if training_settings['scheduler'] is not None: scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=training_settings['scheduler']['factor'], patience=training_settings['scheduler']['patience'], eps=0.)
            initial_epoch = 1
        # If .chk to save exists rename the old version to .bck
        path = os.path.join(training_settings["folder"], training_settings["name"]+".chk")
        if os.path.exists(path):
            print('Renaming', path, 'to:', path+'.bck')
            os.rename(path, path+'.bck')
        # Initialise tensor board writer
        if training_settings['tensor_board'] is not None: writer = SummaryWriter(os.path.join(training_settings["tensor_board"], training_settings["name"]))
        # Initialise automatic mixed-precision training
        scaler = None
        if training_settings['mixed_precision']:
            print("Training with automatic mixed-precision")
            scaler = torch.cuda.amp.GradScaler()
            # Load previos scaler
            if checkpoint is not None and checkpoint['scaler'] is not None:
                scaler.load_state_dict(checkpoint['scaler'])
        # Print before training
        print(f'Training on device: {self.device}')
        print(f'Number of learnable parameters: {self.num_learnable_params}')
        print(f'Total number of parameters:     {self.num_params}')
        # Training loop
        for epoch in tqdm(range(initial_epoch, training_settings['epochs']+1), desc="Completed epochs", leave=False, position=0):
            if optimiser.param_groups[0]['lr'] < training_settings['stopping']:
                print(f"The learning rate is smaller than {training_settings['stopping']}. Stopping training.")
                self.save_checkpoint(path, epoch, optimiser, scheduler=scheduler, scaler=scaler)
                break
            print("\n")
            print(f"Hyperparameters: lr = {optimiser.param_groups[0]['lr']}")
            self.train()
            training_loss = 0.
            gradients_norm = 0.
            for iteration, graph in enumerate(dataloader):
                graph = graph.to(self.device)
                batch_size = graph.batch.max().item() + 1
                if self.is_latent_diffusion:
                    graph = self.autoencoder.transform(graph)
                # Forward pass
                with torch.cuda.amp.autocast() if training_settings['mixed_precision'] else contextlib.nullcontext(): # Use automatic mixed-precision
                    # Sample a batch of random diffusion steps
                    graph.r, sample_weight = step_sampler(batch_size, self.device) # Dimension: (batch_size), (batch_size)
                    # Diffuse the solution/target field
                    graph.field_start = graph.x_latent_target if self.is_latent_diffusion else graph.target
                    graph.field_r, graph.noise = self.diffusion_process(
                        field_start    = graph.field_start,
                        r              = graph.r,
                        batch          = graph.batch,
                        dirichlet_mask = None if self.is_latent_diffusion else getattr(graph, 'dirichlet_mask', None)
                    ) # Shapes (num_nodes, num_fields), (num_nodes, num_fields)
                    # Compute the loss for each sample in the batch
                    loss = criterion(self, graph) # Dimension: (batch_size)
                    # Update the loss-aware diffusion-step sampler
                    if isinstance(step_sampler, ImportanceStepSampler):
                        step_sampler.update(graph.r, loss.detach())
                    # Compute the weighted loss over the batch
                    loss = (loss * sample_weight).mean()
                # Back-propagation
                if training_settings['mixed_precision']:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                # Save training loss and gradients norm before applying gradient clipping to the weights
                training_loss  += loss.item()
                gradients_norm += self.grad_norm()
                # Update the weights
                if training_settings['mixed_precision']:
                    # Clip the gradients
                    if training_settings['grad_clip'] is not None and epoch > training_settings['grad_clip']["epoch"]:
                        scaler.unscale_(optimiser)
                        nn.utils.clip_grad_norm_(self.parameters(), training_settings['grad_clip']["limit"])
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    # Clip the gradients
                    if training_settings['grad_clip'] is not None and epoch > training_settings['grad_clip']["epoch"]:
                        nn.utils.clip_grad_norm_(self.parameters(), training_settings['grad_clip']["limit"])
                    optimiser.step()
                # Reset the gradients
                optimiser.zero_grad()
            training_loss  /= (iteration + 1)
            gradients_norm /= (iteration + 1)
            # Display on terminal
            print(f"Epoch: {epoch:4d}, Training loss: {training_loss:.4e}, Gradients: {gradients_norm:.4e}")
            # Log in TensorBoard
            if training_settings['tensor_board'] is not None:
                writer.add_scalar('Loss/train', training_loss,   epoch)
            # Update lr
            if scheduler is not None:
                scheduler.step(training_loss)
            # Create training checkpoint
            if not epoch % training_settings["chk_interval"]:
                print('Saving checkpoint in:', path)
                self.save_checkpoint(path, epoch, optimiser, scheduler=scheduler, scaler=scaler)
        writer.close()
        print("Finished training")
        return
    
    @abstractmethod
    def forward(self, graph: Graph) -> torch.Tensor:
        """Forward pass of the model. This method must be implemented for each model."""
        pass

    def get_posterior_mean_and_variance_from_output(
        self,
        model_output:      Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        graph:             Graph,
        diffusion_process: DiffusionProcess = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the posterior mean and variance from the model output.

        Args:
            model_output (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]): The model output.
                This can be the noise ("Denoising Diffusion Probabilistic Models") or a tuple with the noise and v ("Improved Denoising Diffusion Probabilistic Models")
            graph (Graph): The graph.
            diffusion_process (DiffusionProcess, optional): The diffusion process. If `None`, the diffusion process of the model is used. Defaults to `None`.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The posterior mean and variance.
        """
        dp = self.diffusion_process if diffusion_process is None else diffusion_process
        if isinstance(model_output, tuple):
            assert self.learnable_variance, 'The model output is a tuple, but learnable_variance=False'
        else:
            assert not self.learnable_variance, 'The model output is not a tuple, but learnable_variance=True'
        # Pre-compute needed coefficients
        betas_r = dp.get_index_from_list(dp.betas, graph.batch, graph.r)
        sqrt_one_minus_alphas_cumprod_r = dp.get_index_from_list(dp.sqrt_one_minus_alphas_cumprod, graph.batch, graph.r)
        sqrt_recip_alphas_r = dp.get_index_from_list(dp.sqrt_recip_alphas, graph.batch, graph.r)
        if self.learnable_variance:
            eps, v = model_output
            v = (v + 1) / 2
            # For min_log we use the clipped posterior variance to avoid nan values in the backward pass
            min_log = dp.get_index_from_list(dp.posterior_log_variance_clipped, graph.batch, graph.r)
            max_log = torch.log(dp.get_index_from_list(dp.betas, graph.batch, graph.r))
            log_variance = v * max_log + (1 - v) * min_log
            variance = torch.exp(log_variance)
        else:
            eps = model_output
            variance = dp.get_index_from_list(dp.posterior_variance, graph.batch, graph.r)
        mean =  sqrt_recip_alphas_r * (graph.field_r - betas_r * eps/sqrt_one_minus_alphas_cumprod_r)
        return mean, variance

    @torch.no_grad()
    def sample(
        self,
        graph:            Graph,
        steps:            List[int]    = None,
        dirichlet_values: torch.Tensor = None,
    ) -> torch.Tensor:
        """Sample from the model.
        
        Args:
            graph (Graph): The graph.
            steps (List[int], optional): The steps to sample. If `None`, all steps are sampled. Defaults to `None`.
            dirichlet_values (torch.Tensor, optional): The Dirichlet boundary conditions. If `None`, no Dirichlet boundary conditions are applied. Defaults to `None`.

        Returns:
            torch.Tensor: The sample.
        """

        if steps is not None:
            assert all([isinstance(s, int) for s in steps]), 'steps must be a list of integers'
            # Sort the steps in ascending order
            steps = sorted(steps)
            # We cannot have repeated steps and the last step must be smaller than the number of steps in the "full" diffusion process
            assert len(steps) == len(set(steps)), 'steps must not have repeated values'
            assert steps[-1] < self.diffusion_process.num_steps, 'The last step in steps must be smaller than the number of steps in the "full" diffusion process'
        self.eval()
        if not hasattr(graph, 'batch') or graph.batch is None:
            graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=self.device)
        if graph.pos.device != self.device:
            graph.to(self.device)
        if dirichlet_values is not None:
            dirichlet_values = dirichlet_values.to(self.device)
        batch_size = graph.batch.max().item() + 1
        # Get the latent features if the model is a latent diffusion model
        if self.is_latent_diffusion:
            c_latent_list, e_latent_list, edge_index_list, batch_list = self.autoencoder.cond_encoder(
                graph,
                torch.cat([f for f in [graph.get('loc'), graph.get('glob'), graph.get('omega')] if f is not None], dim=1),
                torch.cat([f for f in [graph.get('edge_attr'), graph.get('edge_cond')] if f is not None], dim=1),
                graph.edge_index
            )
            graph.c_latent   = c_latent_list  [-1].clone()
            graph.e_latent   = e_latent_list  [-1].clone()
            graph.edge_index = edge_index_List[-1].clone()
            graph.batch      = batch_list     [-1].clone()
        # Use the spaced diffusion process if it is defined
        dp = DiffusionProcessSubSet(self.diffusion_process, steps) if steps is not None else self.diffusion_process
        # Sample the noisy solution
        graph.field_r = torch.randn(graph.batch.size(0), self.num_fields, device=self.device) # Shape (num_nodes, num_fields)
        # Take into account the dirichlet boundary conditions if they are defined and if we are not using a latent diffusion model
        if hasattr(graph, 'dirichlet_mask') and not self.is_latent_diffusion:
            assert dirichlet_values is not None, 'dirichlet_values must be provided if graph has dirichlet_mask'
            assert dirichlet_values.shape == (graph.num_nodes, self.num_fields), f'dirichlet_values.shape must be (num_nodes, num_fields), but it is {dirichlet_values.shape}'
            dirichlet_field_r, _ = dp(
                field_start    = dirichlet_values,
                r              = torch.tensor(batch_size * [dp.num_steps - 1], dtype=torch.long, device=self.device),
                batch          = graph.batch,
                dirichlet_mask = graph.dirichlet_mask
            ) # Shapes (num_nodes, num_fields), (num_nodes, num_fields)
            graph.field_r = graph.field_r * (~graph.dirichlet_mask) + dirichlet_field_r * graph.dirichlet_mask
        # Denoise 'field_r' step-by-step
        for r in dp.steps[::-1]:
            graph.r = torch.full((batch_size,), r, dtype=torch.long, device=self.device)
            # Compute the posterior mean and variance
            model_output = self(graph)
            mean, variance = self.get_posterior_mean_and_variance_from_output(model_output, graph, dp)
            if hasattr(graph, 'dirichlet_mask') and not self.is_latent_diffusion:
                gaussian_noise = torch.randn_like(mean) *  (~graph.dirichlet_mask) # Shape (num_nodes, num_fields)
            else:
                gaussian_noise = torch.randn_like(mean) # Shape (num_nodes, num_fields)
            graph.field_r = mean + torch.sqrt(variance) * gaussian_noise
        # Decode the denoised latent features
        if self.is_latent_diffusion:
            return self.autoencoder.decode(
                graph           = graph,
                v_latent        = graph.field_r,
                c_latent_list   = c_latent_list,
                e_latent_list   = e_latent_list,
                edge_index_list = edge_index_list,
                batch_list      = batch_list,
                dirichlet_mask  = graph.dirichlet_mask if hasattr(graph, 'dirichlet_mask') else None,
                v_0             = dirichlet_values     if hasattr(graph, 'dirichlet_mask') else None,
            )
        else:
            return graph.field_r
        
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
                sample = self.sample(batch, dirichlet_values=dirichlet_values.repeat(current_batch_size, 1) if dirichlet_values is not None else None, *args, **kwargs)
                # Split base on the batch index
                sample = torch.stack(sample.chunk(current_batch_size, dim=0), dim=1)
                samples.append(sample)
            return torch.cat(samples, dim=1)
        else:
            for _ in tqdm(range(num_samples), desc=f"Generating {num_samples} samples", leave=False, position=0):
                sample = self.sample(graph, dirichlet_values, *args, **kwargs)
                samples.append(sample)
            return torch.stack(samples, dim=1) # Dimension: (num_nodes, num_samples, num_fields)