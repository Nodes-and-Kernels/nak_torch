from dataclasses import astuple

import torch

from nak_torch.algorithms.msip.msip_tools import GeneralMSIPAlgorithm, MSIPAlgorithmArgs
from .msip_map import msip_map, get_msip_wts

__all__ = ["MSIP"]


class MSIP(GeneralMSIPAlgorithm[MSIPAlgorithmArgs]):
    def initialize(self, init_particles, target, target_args):
        kernel_lengthscale = self.get_adaptive_lengthscale(init_particles)
        estimator_output = target(init_particles, kernel_lengthscale, target_args)
        kernel_matrix = self.get_infl_kernel_matrix(init_particles, kernel_lengthscale)
        wts = get_msip_wts(init_particles, estimator_output, kernel_matrix)
        return wts, MSIPAlgorithmArgs(
            kernel_lengthscale, kernel_matrix, estimator_output
        )

    def step(self, lr, particles, target, algorithm_args, target_args):
        kernel_lengthscale, kernel_matrix, estimator_output = astuple(algorithm_args)
        kernel_matrix_inverse = torch.linalg.pinv(kernel_matrix)

        # Update the particles
        particles_diff = msip_map(
            estimator_output,
            particles,
            kernel_matrix_inverse,
            output_idx=None,
        )
        new_particles = particles.mul(1 - lr).add_(particles_diff.mul_(lr))

        # Update the parameters
        kernel_lengthscale = self.get_adaptive_lengthscale(new_particles)
        kernel_matrix = self.get_infl_kernel_matrix(new_particles, kernel_lengthscale)
        msip_estimator_output = target(particles, kernel_lengthscale, target_args)
        algorithm_args = MSIPAlgorithmArgs(
            kernel_lengthscale, kernel_matrix, msip_estimator_output
        )
        new_weights = get_msip_wts(new_particles, estimator_output, kernel_matrix)
        return new_particles, new_weights, algorithm_args
