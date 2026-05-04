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

__all__ = ["EKS"]


def eks_step(
    particles: BatchPtType,
    forecast_observations: Float[Tensor, "batch obs"],
    prior_mean: PtType,
    likelihood_precision: CovType,
    prior_precision: CovType,
    true_observation: Float[Tensor, " obs"],
    dt: float,
    rng: torch.Generator,
) -> BatchPtType:
    device, dtype = particles.device, particles.dtype
    N_batch, dim = particles.shape
    particle_mean = particles.mean(0, True)
    forecast_obs_mean = forecast_observations.mean(0, True)
    obs_diff = forecast_observations - true_observation
    forecast_diff = forecast_observations - forecast_obs_mean
    prior_ens_diff = particles - particle_mean
    if prior_mean != 0.0:
        prior_ens_diff -= prior_mean
    cov_forecast = (prior_ens_diff.T @ prior_ens_diff) / N_batch

    if isinstance(likelihood_precision, float) or likelihood_precision.numel() == 1:
        likely_term = torch.einsum("ko,jo,kd->jd", forecast_diff, obs_diff, particles)
        likely_term.mul_(dt * likelihood_precision / N_batch)
    else:
        likely_term = torch.einsum(
            "kp,pq,jq,kd->jd",
            forecast_diff,
            likelihood_precision,
            obs_diff,
            particles,
        )
        likely_term.mul_(dt / N_batch)
    # INPLACE
    cov_forecast.mul_(dt)
    sqrt_prior_cov = sym_sqrtm(cov_forecast)
    sqrt_2 = torch.as_tensor(2.0, device=device, dtype=dtype).sqrt()
    sqrt_prior_cov.mul_(sqrt_2)
    if isinstance(prior_precision, float) or prior_precision.numel() == 1:
        prior_term_premul = cov_forecast.mul_(prior_precision)
    elif isinstance(prior_precision, Tensor):
        prior_term_premul = torch.matmul(cov_forecast, prior_precision)
    else:
        raise ValueError()

    prior_term_premul.add_(torch.eye(dim, device=device))
    new_particles: BatchPtType = torch.linalg.solve(
        prior_term_premul, particles - likely_term, left=False
    )
    noise_tens = torch.normal(
        0.0,
        1.0,
        size=particles.shape,
        device=particles.device,
        dtype=particles.dtype,
        generator=rng,
    )
    noise_samp = noise_tens @ sqrt_prior_cov
    return new_particles.add_(noise_samp)


class EKS(UnweightedAdaptiveNAKAlgorithm[GaussianModel, None]):
    rng: torch.Generator

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
        (
            forward_model,
            likelihood_precision,
            prior_precision,
            true_obs,
            prior_mean,
            _,
        ) = astuple(target)
        forecast_observations = forward_model(particles, target_args)
        new_particles = eks_step(
            particles,
            forecast_observations,
            prior_mean,
            likelihood_precision,
            prior_precision,
            true_obs,
            lr,
            self.rng,
        )
        return new_particles, None, algorithm_args
