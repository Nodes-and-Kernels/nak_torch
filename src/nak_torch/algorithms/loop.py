from typing import Any, Optional

from tqdm import tqdm
import numpy as np
import torch
from torch import Tensor

from nak_torch.tools.util import initialize_particles
from nak_torch.tools.types import (
    BatchDensityEvaluator,
)

from nak_torch.tools.func import NAKAlgorithm, WeightedNAKAlgorithm


def nak(
    log_density: BatchDensityEvaluator,
    algorithm: NAKAlgorithm,
    n_particles: int,
    n_steps: int,
    lr: float,
    seed: Optional[int] = None,
    init_particles: Optional[Tensor | np.ndarray] = None,
    bounds: Optional[tuple[float, float]] = None,
    keep_all: bool = True,
    target_args: Any = None,
    verbose: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    r"""
    TODO: Document
    """

    if n_steps < 0:
        raise ValueError("Expected positive number of steps.")

    if seed is not None:
        torch.manual_seed(seed)

    dim, device, dtype = algorithm.dim, algorithm.device, algorithm.dtype
    particles = initialize_particles(
        n_particles, dim, init_particles, device, dtype, bounds
    )
    is_weighted = isinstance(algorithm, WeightedNAKAlgorithm)
    if keep_all:
        trajectories = torch.empty(
            (n_steps + 1, *particles.shape), device=device, dtype=dtype
        )
        trajectories[0].copy_(particles)
        if is_weighted:
            traj_wts = torch.empty(
                (n_steps + 1, particles.shape[0]), device=device, dtype=dtype
            )
        else:
            traj_wts = torch.empty(())
    else:
        trajectories = torch.empty(())
        traj_wts = torch.empty(())
    particle_wts = torch.empty(())
    for idx in tqdm(range(n_steps + 1), disable=not verbose):
        algorithm_args = algorithm.update(particles)

        if keep_all and is_weighted:
            particle_wts = algorithm.get_weights(particles, target_args)
            traj_wts[idx].copy_(particle_wts)

        if idx < n_steps:
            particles, algorithm_args = algorithm(
                lr, log_density, particles, algorithm_args, target_args
            )

            if bounds is not None:
                particles.clamp_(bounds[0], bounds[1])

            if keep_all:
                trajectories[idx + 1].copy_(particles)

    if not keep_all:
        trajectories = particles.unsqueeze_(0)
        if is_weighted:
            traj_wts = particle_wts.unsqueeze_(0)
    if is_weighted:
        return trajectories.detach(), traj_wts.detach()

    return trajectories.detach()
