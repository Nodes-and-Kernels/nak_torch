import torch
from typing import Optional
from nak_torch.tools.func import UnweightedAdaptiveNAKAlgorithm
from nak_torch.tools.types import (
    BatchGradLogDensityEvaluator,
    BatchPtType,
    DeviceLike,
)
from nak_torch.tools.util import (
    sym_sqrtm,
)

__all__ = ["GradALDI"]


def grad_aldi_step(
    particles: BatchPtType,
    grad_log_dens: BatchPtType,
    rng: torch.Generator,
) -> tuple[BatchPtType, BatchPtType]:

    batch, dim = particles.shape
    particles_mean = particles.mean(dim=0, keepdim=True)
    particles_diff = particles - particles_mean
    particles_cov = (particles_diff.T @ particles_diff) / batch
    # -C(U) ∇Φ(u^i)--- note that Φ = -log p
    term1 = grad_log_dens @ particles_cov.T
    # (D+1)/N (u^i - m(U))
    term2 = particles_diff.mul_((dim + 1) / batch)
    # Get noise
    particles_sqrt_cov = sym_sqrtm(2 * particles_cov)
    # sqrt(2) comes from noise
    particles_noise_iid = torch.normal(
        0.0,
        1.0,
        size=particles.shape,
        generator=rng,
        dtype=particles.dtype,
        device=particles.device,
    )
    particles_noise = particles_noise_iid @ particles_sqrt_cov
    drift_term = term1.add_(term2)
    return drift_term, particles_noise


class GradALDI(UnweightedAdaptiveNAKAlgorithm[BatchGradLogDensityEvaluator, None]):
    rng: torch.Generator

    def _sqrt(self, x: float):
        return torch.sqrt_(torch.as_tensor(x, device=self.device, dtype=self.dtype))

    def __init__(
        self,
        dim: int,
        n_particles: int,
        device: Optional[DeviceLike] = None,
        dtype: Optional[torch.dtype] = None,
        *_,
        rng: torch.Generator,
        **kwargs,
    ):
        super().__init__(dim, n_particles, device, dtype, **kwargs)
        self.rng = rng

    def initialize(self, init_particles, target, target_args):
        return None, None

    def step(self, lr, particles, target, algorithm_args, target_args):
        grad_log_dens_evals = target(particles, target_args)
        particles_diff, particles_noise = grad_aldi_step(
            particles, grad_log_dens_evals, self.rng
        )
        particles_diff.mul_(lr)
        particles_noise.mul_(self._sqrt(lr))
        new_particles = particles_diff.add_(particles).add_(particles_noise)
        return new_particles, None, None
