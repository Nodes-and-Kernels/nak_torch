import torch
from typing import Optional
from jaxtyping import Float
from torch import Tensor
from nak_torch.tools.types import BatchPtType, GaussianModel
import warnings
from tqdm import tqdm
import numpy as np
from nak_torch.tools.util import sym_sqrtm, initialize_particles


def build_eks_step(
    eks_model: GaussianModel,
    dt: float,
    device: Optional[torch.device],
    compile_step: bool,
):
    likelihood_precision = torch.as_tensor(
        eks_model.likelihood_precision, device=device
    )
    prior_precision = torch.as_tensor(eks_model.prior_precision, device=device)
    true_obs = torch.as_tensor(eks_model.true_obs, device=device)
    if isinstance(true_obs, Tensor):
        true_obs.reshape(1, -1)

    sqrt_2 = torch.sqrt(torch.tensor(2, dtype=true_obs.dtype, device=device))

    def eks_step(
        particles: BatchPtType, forecast_observations: Float[Tensor, "batch obs"]
    ) -> tuple[BatchPtType, Float[Tensor, "dim dim"]]:
        N_batch, dim = particles.shape
        particle_mean = particles.mean(0, True)
        forecast_obs_mean = forecast_observations.mean(0, True)
        obs_diff = forecast_observations - true_obs
        forecast_diff = forecast_observations - forecast_obs_mean
        prior_ens_diff = particles - particle_mean
        cov_forecast = (prior_ens_diff.T @ prior_ens_diff) / N_batch

        if isinstance(likelihood_precision, float) or likelihood_precision.numel() == 1:
            likely_term = torch.einsum(
                "ko,jo,kd->jd", forecast_diff, obs_diff, particles
            )
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
        sqrt_prior_cov.mul_(sqrt_2)
        if isinstance(prior_precision, float) or prior_precision.numel() == 1:
            prior_term_premul = cov_forecast.mul_(prior_precision)
        elif isinstance(prior_precision, Tensor):
            prior_term_premul = torch.matmul(cov_forecast, prior_precision)
        else:
            raise ValueError()

        prior_term_premul.add_(torch.eye(dim, device=device))
        new_particles = torch.linalg.solve(
            prior_term_premul, particles - likely_term, left=False
        )
        return new_particles, sqrt_prior_cov

    return torch.compile(eks_step) if compile_step else eks_step


def eks(
    eks_model: GaussianModel,
    n_particles: int,
    n_steps: int,
    dim: int,
    lr: float,
    noise=None,
    seed=None,
    device=None,
    init_particles: Optional[torch.Tensor | np.ndarray] = None,
    bounds: Optional[tuple[float, float]] = None,
    keep_all: bool = True,
    rng: Optional[torch.Generator] = None,
    verbose: bool = False,
    compile_step: bool = True,
    **unused_kwargs,
):
    if verbose and len(unused_kwargs) > 0:
        warnings.warn("Unused kwargs:\n{}".format(unused_kwargs))

    if rng is None:
        rng = torch.default_generator
    if seed is not None:
        rng.manual_seed(seed)

    particles = initialize_particles(
        n_particles, dim, init_particles, device, bounds, rng
    )

    if keep_all:
        trajectories = torch.empty(
            (n_steps, *particles.shape), device=device, dtype=particles.dtype
        )
        trajectories[0].copy_(particles)
    else:
        trajectories = torch.empty(())

    eks_step = build_eks_step(eks_model, lr, device, compile_step)
    noise_tens = torch.empty_like(particles)
    for idx in tqdm(range(n_steps), disable=not verbose):
        forecast_obs = eks_model.forward_model(particles)
        with torch.no_grad():
            particles, noise_sqrt_cov = eks_step(particles, forecast_obs)
            noise_tens = torch.normal(
                mean=0.0, std=1.0, size=particles.shape, generator=rng, out=noise_tens
            )
            noise_samp = noise_tens @ noise_sqrt_cov
            particles = particles.add_(noise_samp)
            if bounds is not None:
                particles.clamp_(bounds[0], bounds[1])
        if keep_all:
            trajectories[idx].copy_(particles)

    return trajectories.detach() if keep_all else particles.unsqueeze_(0)
