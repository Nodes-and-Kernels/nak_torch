from dataclasses import astuple, dataclass

import torch
from typing import Optional
from nak_torch.tools.func import UnweightedAdaptiveNAKAlgorithm
from nak_torch.tools.types import (
    BatchLogDensityEvaluator,
    BatchType,
    BatchPtType,
    DeviceLike,
)
from nak_torch.tools.util import sym_sqrtm

__all__ = ["CBS"]


def cbs_step(
    particles: BatchPtType,
    log_dens: BatchType,
    inverse_temp: float,
    motion_scaling_sq: float,
    rng: torch.Generator,
) -> tuple[BatchPtType, BatchPtType]:
    temper_log_dens = log_dens.mul_(inverse_temp)
    wts = torch.nn.functional.softmax(temper_log_dens, dim=0)
    particles_mean = wts @ particles
    particles_diff = particles - particles_mean
    particles_cov = torch.einsum("bi,b,bj->ij", particles_diff, wts, particles_diff)
    drift_term = particles_diff.neg_()
    noise_sqrt_cov = sym_sqrtm(particles_cov.mul_(motion_scaling_sq))
    motion_term = (
        torch.normal(0.0, 1.0, particles.shape, generator=rng, device=rng.device)
        @ noise_sqrt_cov
    )
    return drift_term, motion_term


@dataclass
class CBSAlgorithmArgs:
    inverse_temp: float
    motion_scaling_sq_div_lr: float


class CBS(UnweightedAdaptiveNAKAlgorithm[BatchLogDensityEvaluator, CBSAlgorithmArgs]):
    default_inverse_temp: float
    rng: torch.Generator

    def __init__(
        self,
        dim: int,
        n_particles: int,
        device: Optional[DeviceLike] = None,
        dtype: Optional[torch.dtype] = None,
        *_,
        default_inverse_temp: float,
        rng: torch.Generator,
    ):
        super().__init__(dim, n_particles, device, dtype)
        self.default_inverse_temp = default_inverse_temp
        self.rng = rng

    def initialize(self, init_particles, target, target_args):
        inverse_temp = self.default_inverse_temp
        motion_scaling_sq_div_lr = 2 * (1 + inverse_temp)
        alg_args = CBSAlgorithmArgs(inverse_temp, motion_scaling_sq_div_lr)
        return None, alg_args

    def step(self, lr, particles, target, algorithm_args, target_args):
        inverse_temp, motion_scaling_sq_div_lr = astuple(algorithm_args)
        motion_scaling_sq = motion_scaling_sq_div_lr * lr
        log_dens_eval = target(particles, None, target_args)
        particles_diff, particles_noise = cbs_step(
            particles, log_dens_eval, inverse_temp, motion_scaling_sq, self.rng
        )
        particles_diff.mul_(lr)
        new_particles = particles_diff.add_(particles).add_(particles_noise)
        return new_particles, None, algorithm_args
