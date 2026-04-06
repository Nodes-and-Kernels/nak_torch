import torch
from typing import Optional
from nak_torch.tools.types import BatchGradLogDensity, BatchPtType
import warnings
from tqdm import tqdm
import numpy as np
from nak_torch.tools.util import (
    batched_grad_log_density_factory,
    initialize_particles,
    sym_sqrtm,
)


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
        0.0, 1.0, size=particles.shape, generator=rng, device=particles.device
    )
    particles_noise = particles_noise_iid @ particles_sqrt_cov
    drift_term = term1.add_(term2)
    return drift_term, particles_noise


def grad_aldi(
    log_density,
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
    is_log_density_batched: bool = False,
    grad_log_density: Optional[BatchGradLogDensity] = None,
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

    grad_log_p = batched_grad_log_density_factory(
        log_density, is_log_density_batched, grad_log_density
    )
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

    sqrt_lr = torch.sqrt(torch.tensor(lr))
    g_aldi_step = grad_aldi_step
    if compile_step:
        g_aldi_step = torch.compile(g_aldi_step)

    for idx in tqdm(range(n_steps), disable=not verbose):
        grad_log_dens_eval = grad_log_p(particles)
        with torch.no_grad():
            particles_diff, particles_noise = g_aldi_step(
                particles, grad_log_dens_eval, rng
            )
            particles_diff.mul_(lr)
            particles_noise.mul_(sqrt_lr)
            particles.add_(particles_diff).add_(particles_noise)
            if bounds is not None:
                particles.clamp_(bounds[0], bounds[1])
        if keep_all:
            trajectories[idx].copy_(particles)

    return trajectories.detach() if keep_all else particles.unsqueeze_(0)
