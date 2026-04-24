from dataclasses import astuple

import torch
from typing import Any, Optional
from jaxtyping import Float
from torch import Tensor
from nak_torch.tools.func import UnweightedAdaptiveNAKAlgorithm
from nak_torch.tools.types import (
    BatchPtType,
    CovType,
    DeviceLike,
    GaussianModel,
    PtType,
)
from nak_torch.tools.util import sym_sqrtm


# def build_gradfree_aldi_step(
#     model: GaussianModel, rng: torch.Generator, compile_step: bool
# ):
#     prior_mean = model.prior_mean
#     likelihood_precision = model.likelihood_precision
#     prior_precision = model.prior_precision
#     true_obs = model.true_obs
#     if isinstance(true_obs, Tensor):
#         true_obs.reshape(1, -1)

#     sqrt_2 = torch.sqrt(torch.tensor(2, dtype=true_obs.dtype, device=true_obs.device))


def gradfree_aldi_step(
    particles: BatchPtType,
    forecast_observations: Float[Tensor, "batch obs"],
    prior_mean: PtType,
    likelihood_precision: CovType,
    prior_precision: CovType,
    true_observation: Float[Tensor, " obs"],
    rng: torch.Generator,
) -> tuple[BatchPtType, BatchPtType]:

    N_batch, dim = particles.shape
    particle_mean = particles.mean(dim=0, keepdim=True)
    forecast_obs_mean = forecast_observations.mean(dim=0, keepdim=True)
    prior_err = particles
    if prior_mean != 0.0:
        prior_err -= prior_mean
    obs_error = forecast_observations - true_observation
    obs_deviation = forecast_observations - forecast_obs_mean
    forecast_deviation = particles - particle_mean
    cov_forecast = (forecast_deviation.T @ forecast_deviation) / N_batch
    cov_obs_forecast = (obs_deviation.T @ forecast_deviation) / N_batch

    if isinstance(likelihood_precision, float):
        likely_term = obs_error @ cov_obs_forecast
        likely_term.mul_(likelihood_precision)
    else:
        likely_term = torch.chain_matmul(
            obs_error, likelihood_precision, cov_obs_forecast
        )

    sqrt_cov_forecast = sym_sqrtm(cov_forecast)

    if isinstance(prior_precision, float):
        prior_term1 = prior_err @ cov_forecast
        prior_term1.mul_(prior_precision)
    else:
        prior_term1 = torch.chain_matmul(cov_forecast, prior_precision, prior_err)

    prior_term2 = forecast_deviation.mul_((dim + 1) / N_batch)
    particle_diff = prior_term2.sub_(prior_term1).sub_(likely_term)
    noise = torch.normal(0.0, 1.0, particles.shape, generator=rng)
    motion = torch.matmul(noise, sqrt_cov_forecast)

    return particle_diff, motion


class GradFreeALDI(UnweightedAdaptiveNAKAlgorithm[GaussianModel, None]):
    rng: torch.Generator

    def sqrt_scalar(self, scalar: float) -> Float:
        return torch.as_tensor(scalar, device=self.device, dtype=self.dtype).sqrt()

    def __init__(
        self,
        dim: int,
        n_particles: int,
        device: Optional[DeviceLike] = None,
        dtype: Optional[torch.dtype] = None,
        *_,
        rng_or_seed: Optional[torch.Generator | int] = None,
        **kwargs,
    ):
        super().__init__(dim, n_particles, device, dtype, **kwargs)
        if isinstance(rng_or_seed, int):
            self.rng = torch.Generator(self.device).set_state(
                torch.default_generator.get_state()
            )
            self.rng.manual_seed(rng_or_seed)
        elif rng_or_seed is None:
            self.rng = torch.Generator(self.device).set_state(
                torch.default_generator.get_state()
            )
        else:
            self.rng = rng_or_seed
            if self.rng.device != self.device:
                raise ValueError(
                    f"Expected rng to live on device {self.device}, got {self.rng.device}"
                )

    def initialize(
        self, init_particles: Tensor, target: GaussianModel, target_args: Any
    ) -> tuple[None, None]:
        return None, None

    def step(
        self,
        lr: float,
        particles: Tensor,
        target: GaussianModel,
        algorithm_args: None,
        target_args: Any,
    ) -> tuple[Tensor, None, None]:
        forward_model, likelihood_precision, prior_precision, true_obs, prior_mean = (
            astuple(target)
        )
        forecast_observations = forward_model(particles, target_args)
        particles_diff, particles_noise = gradfree_aldi_step(
            particles,
            forecast_observations,
            prior_mean,
            likelihood_precision,
            prior_precision,
            true_obs,
            self.rng,
        )
        sqrt_lr = self.sqrt_scalar(2 * lr)
        new_particles = (
            particles_diff.mul_(lr).add_(particles).add_(particles_noise.mul_(sqrt_lr))
        )
        return new_particles, None, algorithm_args
