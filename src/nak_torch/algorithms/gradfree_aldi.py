import torch
from typing import Optional
from jaxtyping import Float
from torch import Tensor
from nak_torch.tools.types import BatchPtType, GaussianModel
import warnings
from tqdm import tqdm
import numpy as np
from nak_torch.tools.util import initialize_particles, sym_sqrtm


def build_gradfree_aldi_step(
    model: GaussianModel, rng: torch.Generator, compile_step: bool
):
    prior_mean = model.prior_mean
    likelihood_precision = model.likelihood_precision
    prior_precision = model.prior_precision
    true_obs = model.true_obs
    if isinstance(true_obs, Tensor):
        true_obs.reshape(1, -1)

    sqrt_2 = torch.sqrt(torch.tensor(2, dtype=true_obs.dtype, device=true_obs.device))

    def gradfree_aldi_step(
        particles: BatchPtType, forecast_observations: Float[Tensor, "batch obs"]
    ) -> tuple[BatchPtType, Float[Tensor, "dim dim"]]:
        N_batch, dim = particles.shape
        particle_mean = particles.mean(0, True)
        forecast_obs_mean = forecast_observations.mean(0, True)
        prior_err = particles - prior_mean
        obs_error = forecast_observations - true_obs
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
        sqrt_cov_forecast.mul_(sqrt_2)

        if isinstance(prior_precision, float):
            prior_term1 = prior_err @ cov_forecast
            prior_term1.mul_(prior_precision)
        else:
            prior_term1 = torch.chain_matmul(cov_forecast, prior_precision, prior_err)

        prior_term2 = forecast_deviation.mul_((dim + 1) / N_batch)
        particle_diff = prior_term2.sub_(prior_term1).sub_(likely_term)
        noise = torch.normal(0.0, 1.0, particles.shape, generator=rng, out=prior_err)
        motion = torch.matmul(noise, sqrt_cov_forecast, out=prior_term1)

        return particle_diff, motion

    return torch.compile(gradfree_aldi_step) if compile_step else gradfree_aldi_step


def gradfree_aldi(
    model: GaussianModel,
    n_particles: int,
    n_steps: int,
    dim: int,
    lr: float,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    init_particles: Optional[torch.Tensor | np.ndarray] = None,
    bounds: Optional[tuple[float, float]] = None,
    rng: Optional[torch.Generator] = None,
    keep_all: bool = True,
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
    gradfree_aldi_step = build_gradfree_aldi_step(model, rng, compile_step)
    sqrt_lr = torch.sqrt(torch.tensor(lr))

    for idx in tqdm(range(n_steps), disable=not verbose):
        forecast_observations = model.forward_model(particles)
        with torch.no_grad():
            particles_diff, particles_noise = gradfree_aldi_step(
                particles, forecast_observations
            )
            particles_diff.mul_(lr)
            particles_noise.mul_(sqrt_lr)
            particles.add_(particles_diff).add_(particles_noise)
            if bounds is not None:
                particles.clamp_(bounds[0], bounds[1])
        if keep_all:
            trajectories[idx].copy_(particles)

    return trajectories.detach() if keep_all else particles.unsqueeze_(0)
