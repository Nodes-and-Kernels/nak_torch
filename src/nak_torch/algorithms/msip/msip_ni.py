#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains the implementation of mean shift interacting particles
# Ayoub Belhadji
# 05/12/2025

# TODO: FIX THIS CODE OR DISCARD IT

import numpy as np
import torch
from nak_torch.tools.average import recursive_weighted_average_alpha_v


def update_particles_ni(
    objective_function,
    particles,
    t,
    lr=0.1,
    kernel_bandwidth=1.0,
    noise_level=0.01,
    noise_injection=None,  # <-- new argument: a map for noise
):
    # Make sure this is a leaf with grad
    particles = particles.detach().clone()
    particles.requires_grad_(True)

    fitness = objective_function(particles)  # shape (N,)
    # If fitness is vector, you usually do sum() to get grads w.r.t. all particles
    (grads,) = torch.autograd.grad(fitness.sum(), particles)
    # Be aware that this is the gradient of the log(p), which is the grad of V

    # expf_times_y = torch.exp(fitness.unsqueeze(-1)) * particles

    # From here on, think of 'fitness' and 'grads' as just arrays
    with torch.no_grad():
        diff = particles.unsqueeze(1) - particles.unsqueeze(0)
        sigma2 = kernel_bandwidth**2
        kernel_matrix = torch.exp(-(diff**2).sum(dim=-1) / sigma2)
        # print(kernel_matrix)

        K_minus_one = torch.linalg.inv(kernel_matrix)
        # print(K_minus_one)

        N, d = particles.shape
        t_list = []
        for i in range(N):
            alpha_i = K_minus_one[i, :]

            # all these calls now use 'plain' tensors
            # t1 = recursive_weighted_average_alpha_v(
            #     particles, alpha_i, log_v=torch.log(fitness)
            # )

            # t1_1 = recursive_weighted_average_alpha_v(expf_times_y, alpha_i)
            # t2_1 = recursive_weighted_average_alpha_v(grads, alpha_i)
            # t2_2 = recursive_weighted_average_alpha_v(fitness.unsqueeze(-1), alpha_i)
            # t2 = t2_1 / t2_2
            t1 = recursive_weighted_average_alpha_v(particles, alpha_i, log_v=fitness)
            t2 = recursive_weighted_average_alpha_v(grads, alpha_i, log_v=fitness)
            #
            # t1_1 / t2_2
            # print(t2)
            t_list.append(t1 + sigma2 * t2)

        t_arr = torch.stack(t_list, dim=0)

        # deterministic transport step
        particles_det = (1 - lr) * particles + lr * t_arr

        # optional noise injection map
        if noise_injection is not None:
            # expect: noise_injection(particles, t, noise_level) -> new_particles
            particles_new = noise_injection(particles_det, t, noise_level)
        else:
            particles_new = particles_det

    return particles_new


def sqexp_noise_injection(particles, t, noise_level):
    return particles + noise_level * torch.randn_like(particles)


def msip_ni(
    objective_function,
    n_particles=50,
    n_steps=100,
    dim=2,
    bounds=(-5, 5),
    lr=0.1,
    noise_level_0=0.05,
    kernel_bandwidth=1.0,
    seed=None,
    device="cpu",
):
    if seed is not None:
        torch.manual_seed(seed)

    particles = torch.empty((n_particles, dim), device=device).uniform_(*bounds)

    trajectories = [particles.detach().cpu().numpy().copy()]

    for t in range(n_steps):
        noise_level = noise_level_0 / (t + 1)
        particles = update_particles_ni(
            objective_function,
            particles,
            0,
            lr,
            kernel_bandwidth,
            noise_level,
            sqexp_noise_injection,
        )
        trajectories.append(particles.detach().cpu().numpy().copy())

    return np.array(trajectories), bounds
