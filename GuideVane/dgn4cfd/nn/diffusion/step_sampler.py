import torch
import numpy as np
from abc import ABC, abstractmethod


class StepSampler(ABC):
    def __init__(
        self,
        num_diffusion_steps: int
    ) -> None:
        self.num_diffusion_steps = num_diffusion_steps
                 
    @property
    @abstractmethod
    def weights(self) -> torch.Tensor:
        pass

    def __call__(self,*args, **kwargs):
        return self.sample(*args, **kwargs)

    def sample(
        self,
        batch_size: int,
        device:     torch.device = torch.device('cpu'),
    ) -> torch.Tensor:
        w = self.weights
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights
    

class UniformStepSampler(StepSampler):
    """Uniform sampler for the diffusion steps."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._weights = np.ones([self.num_diffusion_steps], dtype=np.float64)

    @property
    def weights(self) -> torch.Tensor:
        return self._weights
    

class ImportanceStepSampler(StepSampler):
    """Resampler based on the second moment of the loss. From the paper: "Improved Denoising Diffusion Probabilistic Models" (https://arxiv.org/abs/2102.09672)."""
    def __init__(
        self,
        min_history_length: int   = 10,
        uniform_prob:       float = 0.001,
        *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.min_history_length = min_history_length
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros([self.num_diffusion_steps, min_history_length], dtype=np.float64)
        self._loss_counts = np.zeros([self.num_diffusion_steps], dtype=np.int64)

    @property
    def weights(self) -> torch.Tensor:
        if not self._warmed_up():
            return np.ones([self.num_diffusion_steps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update(
        self,
        rs:     torch.Tensor,
        losses: torch.Tensor
    ) -> None:
        for r, loss in zip(rs, losses):
            if self._loss_counts[r] == self.min_history_length:
                # Shift out the oldest loss term.
                self._loss_history[r, :-1] = self._loss_history[r, 1:]
                self._loss_history[r, -1] = loss
            else:
                self._loss_history[r, self._loss_counts[r]] = loss
                self._loss_counts[r] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.min_history_length).all()   

