from typing import Any, Iterator, Optional
import warnings

from tqdm import tqdm
import numpy as np
import torch
from torch import Tensor

from nak_torch.tools.util import initialize_particles

from nak_torch.tools.func import (
    AlgorithmArgsT,
    GeneralAdaptiveNAKAlgorithm,
    NAKTargetT,
    WeightT,
)

__all__ = ["nak"]


def nak(
    target: NAKTargetT,
    algorithm: GeneralAdaptiveNAKAlgorithm[NAKTargetT, WeightT, AlgorithmArgsT],
    n_steps: int,
    lr: float,
    rng_or_seed: Optional[int | torch.Generator] = None,
    init_particles: Optional[Tensor | np.ndarray] = None,
    bounds: Optional[tuple[float, float]] = None,
    keep_all: bool = True,
    target_args: Any = None,
    get_target_args: Optional[Iterator] = None,
    **kwargs,
) -> Tensor | tuple[Tensor, Tensor]:
    r"""
    TODO: Document
    target_args: If `get_target_args` is not None, nak uses this for initializing the algorithm's parameters.
    """
    verbose, n_particles = algorithm.verbose, algorithm.n_particles
    if verbose and len(kwargs) > 0:
        warnings.warn(f"Discarding kwargs {kwargs}")
    if n_steps < 0:
        raise ValueError("Expected positive number of steps.")

    dim, device, dtype = algorithm.dim, algorithm.device, algorithm.dtype
    rng: torch.Generator
    if isinstance(rng_or_seed, int):
        rng = torch.Generator(device)
        rng.manual_seed(rng_or_seed)
    elif rng_or_seed is not None:
        rng = rng_or_seed
    else:
        rng = torch.default_generator

    particles = initialize_particles(
        n_particles, dim, init_particles, device, dtype, bounds, rng=rng
    )

    particle_wts, algorithm_args = algorithm.initialize(particles, target, target_args)

    if keep_all:
        trajectories = torch.empty(
            (n_steps + 1, *particles.shape), device=device, dtype=dtype
        )
        trajectories[0].copy_(particles)
        if algorithm.is_weighted():
            traj_wts = torch.empty(
                (n_steps + 1, particles.shape[0]), device=device, dtype=dtype
            )
            traj_wts[0].copy_(particle_wts)
        else:
            traj_wts = torch.empty(())
    else:
        trajectories = torch.empty(())
        traj_wts = torch.empty(())

    for idx in tqdm(range(n_steps - 1), disable=not verbose):
        if keep_all:
            trajectories[idx + 1].copy_(particles)
            if algorithm.is_weighted():
                traj_wts[idx + 1].copy_(particle_wts)

        if get_target_args is not None:
            target_args = next(get_target_args)

        particles, particle_wts, algorithm_args = algorithm.step(
            lr, particles, target, algorithm_args, target_args
        )

        if bounds is not None:
            particles.clamp_(bounds[0], bounds[1])

    if keep_all:
        trajectories[-1].copy_(particles)
        if algorithm.is_weighted():
            traj_wts[-1].copy_(particle_wts)
    else:
        trajectories = particles.unsqueeze_(0)
        if algorithm.is_weighted():
            traj_wts = particle_wts.unsqueeze_(0)

    if algorithm.is_weighted():
        return trajectories.detach(), traj_wts.detach()

    return trajectories.detach()
