from dataclasses import astuple

import torch


from .msip_map import get_msip_wts, msip_map_one_output
from .msip_tools import GeneralMSIPAlgorithm, MSIPGSAlgorithmArgs

__all__ = ["MSIPGS"]


class MSIPGS(GeneralMSIPAlgorithm[MSIPGSAlgorithmArgs]):
    def initialize(self, init_particles, target, target_args):
        kernel_lengthscale = self.get_adaptive_lengthscale(init_particles)
        estimator_output = target(init_particles, kernel_lengthscale, target_args)
        kernel_matrix = self.get_infl_kernel_matrix(init_particles, kernel_lengthscale)
        wts = get_msip_wts(init_particles, estimator_output, kernel_matrix)
        return wts, MSIPGSAlgorithmArgs(kernel_lengthscale, estimator_output)

    def step(self, lr, particles, target, algorithm_args, target_args):
        kernel_lengthscale, estimator_output = astuple(algorithm_args)
        est_out_0, est_out_1 = estimator_output
        new_particles = particles.clone()
        kernel_matrix = self.get_infl_kernel_matrix(new_particles, kernel_lengthscale)
        for i in range(new_particles.shape[0]):
            km_inv_i = torch.linalg.pinv(kernel_matrix)
            est_out_i_0, est_out_i_1 = target(
                new_particles[i].unsqueeze(0), kernel_lengthscale, target_args
            )
            est_out_0[i].copy_(est_out_i_0.squeeze())
            est_out_1[i].copy_(est_out_i_1.squeeze())
            target_i = msip_map_one_output(
                estimator_output, new_particles, km_inv_i, output_idx=i
            )
            new_particles[i] = new_particles[i].mul(1.0 - lr).add_(target_i.mul_(lr))
            kernel_matrix = self.get_infl_kernel_matrix(
                new_particles, kernel_lengthscale
            )

        # Update the parameters
        new_kernel_lengthscale = self.get_adaptive_lengthscale(new_particles)
        if new_kernel_lengthscale != kernel_lengthscale:
            estimator_output = target(
                new_particles, new_kernel_lengthscale, target_args
            )
            kernel_matrix = self.get_infl_kernel_matrix(
                new_particles, new_kernel_lengthscale
            )
            kernel_lengthscale = new_kernel_lengthscale
        algorithm_args = MSIPGSAlgorithmArgs(kernel_lengthscale, estimator_output)
        new_weights = get_msip_wts(new_particles, estimator_output, kernel_matrix)
        return new_particles, new_weights, algorithm_args
