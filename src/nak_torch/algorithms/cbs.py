import torch
from typing import Optional
from nak_torch.tools.types import BatchType, BatchPtType
import warnings
from tqdm import tqdm
import numpy as np
from nak_torch.tools.util import initialize_particles, sym_sqrtm


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


def cbs(
    log_density,
    n_particles: int,
    n_steps: int,
    dim: int,
    lr: float,
    inverse_temp: float,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    init_particles: Optional[torch.Tensor | np.ndarray] = None,
    bounds: Optional[tuple[float, float]] = None,
    rng: Optional[torch.Generator] = None,
    keep_all: bool = True,
    is_log_density_batched: bool = False,
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
    _cbs_step = cbs_step
    if compile_step:
        _cbs_step = torch.compile(cbs_step)

    log_p = log_density if is_log_density_batched else torch.vmap(log_density)
    motion_scaling_sq = lr * 2 * (1 + inverse_temp)

    for idx in tqdm(range(n_steps), disable=not verbose):
        log_dens_eval = log_p(particles)
        with torch.no_grad():
            particles_diff, particles_noise = _cbs_step(
                particles, log_dens_eval, inverse_temp, motion_scaling_sq, rng
            )
            particles_diff.mul_(lr)
            particles = particles.add_(particles_diff).add_(particles_noise)
            if bounds is not None:
                particles.clamp_(bounds[0], bounds[1])
        if keep_all:
            trajectories[idx].copy_(particles)

    return trajectories.detach() if keep_all else particles.unsqueeze_(0)
