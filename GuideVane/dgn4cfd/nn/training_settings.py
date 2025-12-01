import torch
from typing import Callable

from .diffusion.step_sampler import StepSampler, UniformStepSampler


class TrainingSettings():
    r"""Class to store the training configuration of a model.
    
    Args:
        name (str): Name of the model.
        folder (str, optional): Folder where the model is saved. Defaults to `'./'`.
        checkpoint (Union[None, str], optional): Path of a previous checkpoint to load. If `None`, no checkpoint is loaded. Defaults to `None`.
        tensor_board (Union[None, str], optional): Path of a tensor board to save the training progress. If `None`, no tensor board is saved. Defaults to `None`.
        chk_interval (int, optional): Number of epochs between checkpoints. Defaults to `1`.
        training_loss (Callable, optional): Training loss function. Defaults to `None`.
        validation_loss (Callable, optional): Validation loss function. Defaults to `None`.
        epochs (int, optional): Number of epochs to train. Defaults to `1`.
        batch_size (int, optional): Batch size. Defaults to `1`.
        lr (float, optional): Initial learning rate. Defaults to `1e-3`.
        grad_clip (Union[None, dict], optional): Dictionary with the parameters of the gradient clipping. If `None`, no gradient clipping is used.
            The dictioary must contain the keys `'epoch'` and `'limit'`, indicating from which epoch the gradient clipping is applied and the maximum gradient norm, respectively.
            Defaults to `None`.
        scheduler (Union[None, dict], optional): Dictionary with the parameters of the learning rate scheduler. The dictioary must contain the keys `'factor'`, `'patience'` and `'loss'`.
            The `'factor'` is the factor by which the learning rate is reduced, `'patience'` is the number of epochs with no improvement after which learning rate will be reduced and `'loss'`
            is the loss function used to monitor the improvement (`'training'` or `'validation'`). Defaults to `None`.
        stopping (float, optional): Minimum value of the learning rate. If the learning rate falls below this value, the training is stopped. Defaults to `0.`.
        step_sampler (ScheduleSampler, optional): Diffusio-step sampler. Only used if the model is a diffusion model. Defaults to `UniformSampler`.
        mixed_precision (bool, optional): If `True`, mixed precision is used. Defaults to `False`.
        device (Optional[torch.device], optional): Device where the model is trained. If `None`, the model is trained on its current device. Defaults to `None`.
    """
    def __init__(
        self,
        name:             str,
        folder:           str          = './',    
        checkpoint:       str          = None,
        tensor_board:     str          = None,
        chk_interval:     int          = 1,
        training_loss:    Callable     = None,
        validation_loss:  Callable     = None,
        epochs:           int          = 1,
        batch_size:       int          = 1,
        lr:               float        = 1e-3,
        grad_clip:        dict         = None,
        scheduler:        dict         = None,
        stopping:         float        = 0.,
        step_sampler:     StepSampler  = UniformStepSampler,
        mixed_precision:  bool         = False,
        device:           torch.device = None
    ) -> None: 
        self.name             = name    
        self.folder           = folder
        self.checkpoint       = checkpoint
        self.tensor_board     = tensor_board
        self.chk_interval     = chk_interval
        self.training_loss    = training_loss
        self.validation_loss  = validation_loss
        self.epochs           = epochs
        self.batch_size       = batch_size
        self.lr               = lr
        self.grad_clip        = grad_clip
        self.scheduler        = scheduler
        self.stopping         = stopping
        self.step_sampler     = step_sampler     # Only used for diffusion models
        self.mixed_precision  = mixed_precision
        self.device           = device

    def __repr__(self):
        return repr(self.__dict__)

    def __getitem__(self, key):
        return self.__dict__.get(key)