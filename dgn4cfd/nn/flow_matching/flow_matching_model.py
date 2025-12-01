import contextlib
import os
from typing import Callable
import numpy as np
import torch
from torch import nn, optim
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from abc import abstractmethod
from tqdm import tqdm

from .import integrators
from ..model import Model
from ...graph import Graph
from ...loader import DataLoader, Collater


class FlowMatchingModel(Model):
    r""""Abstract class for a flow-matching model. This class implements the training loop for a flow-matching model.
    The forward method must be implemented for each model.

    Args:
        sigma_min (float, optional): $\sigma_{\min}$ parameter. Default: 0.001.

    Methods:
        flow: Compute the field at a given r value.
        advection_field: Compute the flow-matching vector field between the start and end phisical fields.
        fit: Train the model using the provided training settings and data loader.
        forward: Forward pass of the model. This method must be implemented for each model.
        ode: Returns a function that integrates the ODE d field_r / d r = self.forward(graph).
        sample: Samples by integrating the ODE for a given set of r values.
        sample_n: Generate multiple samples.
    """


    def __init__(
        self,
        sigma_min: float = None,
        *args, 
        **kwargs
    ) -> None:
        self.sigma_min = sigma_min
        super().__init__(*args, **kwargs)
        # This may overwrite sigma_min if we are loading a checkpoint.
        # So, we need to check if the provided values are the same as the ones in the model.
        if self.sigma_min is None:
            self.sigma_min = 0.001
        else:
            if sigma_min is not None:
                assert self.sigma_min == sigma_min, "The provided sigma_min is different from the one in the model."

    @property
    def is_latent(self):
        return hasattr(self, 'autoencoder')
    
    def flow(
        self,
        field_start: torch.Tensor,
        field_end:   torch.Tensor,
        r:           torch.Tensor,
        batch:       torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute the field at a given r value."""
        assert r.dim() == 1, "r must be one-dimensional"
        r = r[batch].unsqueeze(1)
        return (1 - (1 - self.sigma_min) * r) * field_start + r * field_end
    
    def advection_field(
        self,
        field_start: torch.Tensor,
        field_end: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute the flow-matching (optimal transport) vector field between the start and end phisical fields."""
        return field_end - (1 - self.sigma_min) * field_start

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
                if self.is_latent:
                    graph = self.autoencoder.transform(graph)
                # Forward pass
                with torch.cuda.amp.autocast() if training_settings['mixed_precision'] else contextlib.nullcontext(): # Use automatic mixed-precision
                    # Sample a batch of random r values between 0 and 1
                    graph.r = torch.rand(batch_size, device=self.device) # Dimension: (batch_size)
                    # Field end is the target field
                    field_end = (graph.x_latent_target if self.is_latent else graph.target).clone()
                    # Sample field_start from a Gaussian distribution
                    field_start = torch.randn_like(field_end) # Dimension: (num_nodes, num_fields)
                    # Push-forward field_start to graph_field using the probabily density function
                    graph.field_r = self.flow(field_start, field_end, graph.r, graph.batch) # Dimension: (num_nodes, num_fields)
                    # Compute the target advection field
                    graph.advection_field = self.advection_field(field_start, field_end)
                    # Compute the loss (between predicted and target advection field) for each sample in the batch
                    loss = criterion(self, graph) # Dimension: (batch_size)
                    # Compute the weighted loss over the batch
                    loss = loss.mean()
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

    def ode(
        self,
        graph:    Graph,
        callback: Callable = None,
    ) -> Callable:
        """Returns a function that integrates the ODE d field_r / d r = self.forward(graph)"""
        def wrapper(t, y):
            if isinstance(t, torch.Tensor):
                t = t.item()
            graph.r = torch.full((graph.batch.max().item() + 1,), t, device=self.device)
            graph.field_r = y
            if callback is not None:
                callback(graph)
            return self.forward(graph)
        return wrapper

    @torch.no_grad()
    def sample(
        self,
        graph:   Graph,
        steps:   List[int],
        verbose: bool = False,
    ) -> torch.Tensor:
        """Samples by integrating the ODE for a given set of r values.

        Args:
            graph (Graph): The input graph.
            steps (List[int]): The r values where the flow is evaluated during the integration.
            verbose (bool, optional): Print the r value. Default: False.

        Returns:
            torch.Tensor: The sampled field. Dimension: (num_nodes, num_fields)
        """
        steps = np.array(sorted(list(set([0, 1] + list(steps)))))
        self.eval()
        if not hasattr(graph, 'batch') or graph.batch is None:
            graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=self.device)
        if graph.pos.device != self.device:
            graph.to(self.device)
        # Get the latent features if the model is a latent diffusion model
        if self.is_latent:
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
        # Sample field_r(start) from a Gaussian distribution
        graph.field_r = torch.randn(graph.batch.size(0), self.num_fields, device=self.device) # Dimension: (num_nodes, num_fields)
        # Define the ODE right-hand side
        rhs = self.ode(
            graph    = graph,
            callback = lambda graph: print(f"Evaluating advection field at r = {graph.r[0].item():.4f}") if verbose else None,
        )
        # Integrate the ODE (only Euler is implemented)  
        graph.field_r = integrators.euler(
            rhs = rhs,
            y0  = graph.field_r,
            t   = steps,
        )
        # Decode the denoised latent features
        if self.is_latent:
            return self.autoencoder.decode(
                graph           = graph,
                v_latent        = graph.field_r,
                c_latent_list   = c_latent_list,
                e_latent_list   = e_latent_list,
                edge_index_list = edge_index_list,
                batch_list      = batch_list,
            )
        else:
            return graph.field_r

    @torch.no_grad()    
    def sample_n(
        self,
        num_samples: int,
        graph:       Graph,
        batch_size:  int = 0,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Generate multiple samples.

        Args:
            num_samples (int): The number of samples.
            graph (Graph): The input graph.
            batch_size (int, optional): Number of samples to generate in parallel. if 'batch_size' is lower than 2, the samples are generated one by one. Default: 0.
            *args: Additional arguments for the sample method.
            **kwargs: Additional keyword arguments for the sample method.

        Returns:
            torch.Tensor: The generated samples. Dimension: (num_nodes, num_samples, num_fields)
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
                sample = self.sample(batch, *args, **kwargs)
                # Split base on the batch index
                sample = torch.stack(sample.chunk(current_batch_size, dim=0), dim=1)
                samples.append(sample)
            return torch.cat(samples, dim=1)
        else:
            for _ in tqdm(range(num_samples), desc=f"Generating {num_samples} samples", leave=False, position=0):
                sample = self.sample(graph, *args, **kwargs)
                samples.append(sample)
            return torch.stack(samples, dim=1) # Dimension: (num_nodes, num_samples, num_fields)