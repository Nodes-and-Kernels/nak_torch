#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains the implementation of mean shift interacting particles
# Ayoub Belhadji
# 05/12/2025

import warnings
import numpy as np
import torch
from typing import Optional
from tqdm import tqdm
from nak_torch.tools.kernel import sqexp_kernel_elem
from jaxtyping import Float
from torch import Tensor
from nak_torch.tools.types import (
    BatchLogDensity,
    BatchType,
    KernelFunction,
    KernelMatrixType,
    LogDensity,
)
from nak_torch.tools.util import initialize_particles


def kfr_step(
    kernel_matrix: KernelMatrixType,
    grad1_kernel_tens: Float[Tensor, "batch batch dim"],
    log_likely_eval: BatchType,
    kernel_diag_infl: float,
    delta_t: Float,
):
    M_batch = log_likely_eval.shape[0]
    diffusion_matrix = torch.einsum(
        "ild,imd->lm", grad1_kernel_tens, grad1_kernel_tens
    )  # (M, M), 1/M factor cancels in creation of kernelized_wts
    diffusion_matrix[torch.arange(M_batch), torch.arange(M_batch)] += kernel_diag_infl
    log_wts = log_likely_eval - log_likely_eval.mean()
    wts = log_wts * delta_t
    kernelized_wts = kernel_matrix @ wts
    diffusion_soln = torch.linalg.solve(diffusion_matrix, kernelized_wts)
    return torch.einsum("jkd,k->jd", grad1_kernel_tens, diffusion_soln)


"""
    grad_kernel_tens = kernel.first_grad_kernel_tens(points, points)
    kernel_mat = kernel.kernel_matrix(points, points)
    unnorm_weights = target.dens_val(points)
    weights = unnorm_weights - unnorm_weights.mean()
    # Poisson integral operator
    # Note: N_ens cancels out when you do `kernel_mat.dot(weights)` later
    poisson_solver = jnp.einsum(
        'ild,imd->lm', grad_kernel_tens, grad_kernel_tens
    )
    poisson_solver = jax.lax.cond(
        inflation > 0,
        lambda: poisson_solver + inflation * jnp.eye(poisson_solver.shape[0]),
        lambda: poisson_solver
    )
    precond = jax.scipy.linalg.solve(
        poisson_solver, kernel_mat.dot(weights), assume_a="sym"
    )
    point_diff = jnp.einsum(
        'k,jkd->jd', precond, grad_kernel_tens
    )
"""


def kfr_kernel_tens_factory(kernel_elem: KernelFunction):
    kernel_grad_val = torch.func.grad_and_value(kernel_elem)
    kernel_matricization = torch.vmap(
        torch.vmap(kernel_grad_val, in_dims=(None, 0, None)), in_dims=(0, None, None)
    )
    return kernel_matricization


def kfrflow(
    log_like: LogDensity | BatchLogDensity,
    n_particles: int,
    n_steps_or_delta_ts: int | Float[Tensor, " T"],
    dim: int,
    lr: Optional[float] = None,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    init_particles: Optional[torch.Tensor | np.ndarray] = None,
    kernel_length_scale: float = 1.0,
    kernel_diag_infl: float = 0.0,
    kernel_elem: KernelFunction = sqexp_kernel_elem,
    bounds: Optional[tuple[float, float]] = None,
    keep_all: bool = True,
    is_log_density_batched: bool = False,
    verbose: bool = False,
    compile_step: bool = True,
    **unused_kwargs,
):
    if lr is not None:
        warnings.warn("learning rate is not used in KFR-I. See n_steps_or_delta_ts")

    if len(unused_kwargs) > 0:
        warnings.warn("Unused kwargs:\n{}".format(unused_kwargs))

    if seed is not None:
        torch.manual_seed(seed)
    kfr_step_ = kfr_step
    if compile_step:
        kfr_step_ = torch.compile(kfr_step)

    particles = initialize_particles(n_particles, dim, init_particles, device, bounds)
    if isinstance(n_steps_or_delta_ts, int):
        n_steps = n_steps_or_delta_ts
        delta_ts = torch.ones(n_steps) / n_steps
    elif isinstance(n_steps_or_delta_ts, Tensor):
        delta_ts = n_steps_or_delta_ts
        if delta_ts.ndim != 1 or torch.any(delta_ts <= 0.0):
            raise ValueError("Unexpected values encountered in delta ts")
        delta_ts /= delta_ts.sum()
        n_steps = len(delta_ts)
    else:
        raise ValueError("Unexpected value encountered in n_steps_or_delta_ts")

    if keep_all:
        trajectories = torch.empty(
            (n_steps, *particles.shape), device=device, dtype=particles.dtype
        )
        trajectories[0].copy_(particles)
    else:
        trajectories = torch.empty(())

    if not is_log_density_batched:
        log_like = torch.vmap(log_like)

    kernel_fcn = kfr_kernel_tens_factory(kernel_elem)

    for idx in tqdm(range(n_steps), disable=not verbose):
        delta_t = delta_ts[idx]
        grad1_kernel_tens, kernel_mat = kernel_fcn(
            particles, particles, kernel_length_scale
        )
        log_likely_eval = log_like(particles)
        particles_diff = kfr_step_(
            kernel_mat, grad1_kernel_tens, log_likely_eval, kernel_diag_infl, delta_t
        )
        with torch.no_grad():
            particles = particles + particles_diff
            if bounds is not None:
                particles.clamp_(bounds[0], bounds[1])
        if keep_all:
            trajectories[idx].copy_(particles)

    return trajectories.detach() if keep_all else particles.unsqueeze_(0)
