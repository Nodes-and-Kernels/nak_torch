from dataclasses import astuple

import torch

from .msip_map import msip_map, get_msip_wts
from .msip_tools import GeneralMSIPAlgorithm, MSIPGSAlgorithmArgs


class MSIPGS(GeneralMSIPAlgorithm[MSIPGSAlgorithmArgs]):
    def initialize(self, init_particles, target, target_args):
        kernel_lengthscale = self.get_adaptive_lengthscale(init_particles)
        estimator_output = target(init_particles, kernel_lengthscale, target_args)
        kernel_matrix = self.get_infl_kernel_matrix(init_particles, kernel_lengthscale)
        wts = get_msip_wts(init_particles, estimator_output, kernel_matrix)
        return wts, MSIPGSAlgorithmArgs(kernel_lengthscale, estimator_output)

    def step(self, lr, particles, target, algorithm_args, target_args):
        kernel_lengthscale, _, estimator_output = astuple(algorithm_args)
        est_out_0, est_out_1 = estimator_output
        new_particles = particles.clone()
        for i in range(particles.shape[0]):
            km_i = self.get_infl_kernel_matrix(particles, kernel_lengthscale)
            km_inv_i = torch.linalg.pinv(km_i)
            est_out_i_0, est_out_i_1 = target(
                new_particles[i].unsqueeze(0), kernel_lengthscale, target_args
            )
            est_out_0[i].copy_(est_out_i_0.squeeze())
            est_out_1[i].copy_(est_out_i_1.squeeze())

            target_i = msip_map(estimator_output, particles, km_inv_i, output_idx=i)

            new_particles[i].mul_(1.0 - lr).add_(target_i.mul_(lr))

        # Update the parameters
        new_kernel_lengthscale = self.get_adaptive_lengthscale(new_particles)
        kernel_matrix = self.get_infl_kernel_matrix(new_particles, kernel_lengthscale)
        if new_kernel_lengthscale != kernel_lengthscale:
            estimator_output = target(particles, new_kernel_lengthscale, target_args)
            kernel_lengthscale = new_kernel_lengthscale
        algorithm_args = MSIPGSAlgorithmArgs(kernel_lengthscale, estimator_output)
        new_weights = get_msip_wts(new_particles, estimator_output, kernel_matrix)
        return new_particles, new_weights, algorithm_args
