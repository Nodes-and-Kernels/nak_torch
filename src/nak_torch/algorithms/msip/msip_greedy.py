#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains the implementation of mean shift interacting particles_greedy
# Ayoub Belhadji
# 05/12/2025

import numpy as np
import torch
from tqdm import tqdm
from typing import Optional

from nak_torch.tools.util import initialize_particles
from .msip_map import msip_map


def update_one_particle(
    objective_function,
    particles: torch.Tensor,
    idx: int,
    lr: float = 0.1,
    inner_tol: float = 1e-4,
    max_inner_steps: int = 50,
    # kernel_bandwidth: float = 1.0,
    # bandwidth_factor: float = 0.5,
    # bounds: tuple[float, float] = (-torch.inf, torch.inf),
    # projection: bool = True,
    # gradient_informed: bool = True,
    # kernel_diag_infl: float = 0.0,
    **msip_kwargs,
):
    """
    Coordinate-wise MSIP update:
    - All particles are kept fixed except particle `idx`
    - For particle `idx`, we iterate until the MSIP update is small
      or max_inner_steps is reached.
    Mutates `particles` in-place and returns it.
    """

    new_list_particles = []
    for _ in range(max_inner_steps):
        # Compute full MSIP map given current particles

        t_arr = msip_map(
            objective_function,
            particles,
            # kernel_bandwidth,
            # bandwidth_factor,
            # bounds,
            # projection,
            # gradient_informed,
            # kernel_diag_infl,
            output_idx=idx,
            **msip_kwargs,
        )

        with torch.no_grad():
            old_pos = particles[idx]
            new_pos = (1.0 - lr) * old_pos + lr * t_arr

            move_norm = (new_pos - old_pos).norm()
            particles[idx].copy_(new_pos)
            new_list_particles.append(particles.detach().cpu().numpy().copy())

        if move_norm.isnan():
            print("nan")

        if move_norm.item() < inner_tol:
            break

    return new_list_particles


def msip_greedy(
    log_density,
    n_particles: int,
    # now interpreted as "epochs" (passes over all particles)
    n_steps: int,
    dim: int,
    lr: float,
    noise: float = 0.05,  # currently unused, kept for compatibility
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    init_particles: Optional[torch.Tensor | np.ndarray] = None,
    bounds: Optional[tuple[float, float]] = None,
    **msip_kwargs,
):

    if seed is not None:
        torch.manual_seed(seed)

    particles = initialize_particles(n_particles, dim, init_particles, device, bounds)

    trajectories = [particles.detach().cpu().numpy().copy()]

    # Outer loop: epochs
    pbar = tqdm(total=n_steps * n_particles)
    for _ in range(n_steps):
        # Loop over particles, one at a time
        for i in range(n_particles):
            new_list_particles = update_one_particle(
                log_density, particles, idx=i, lr=lr, bounds=bounds, **msip_kwargs
            )
            # If you want a very fine-grained trajectory, record after each particle:
            # trajectories.append(particles.detach().cpu().numpy().copy())
            trajectories = trajectories + new_list_particles
            pbar.update()

        # If you prefer only one snapshot per epoch, move the append here instead:
        # trajectories.append(particles.detach().cpu().numpy().copy())

    return torch.tensor(trajectories)
