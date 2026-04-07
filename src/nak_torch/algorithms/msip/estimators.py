import torch
from abc import ABC, abstractmethod
from nak_torch.tools.average import recursive_weighted_average_alpha_v
from nak_torch.tools.types import (
    BatchPtType,
    MSIPEstimatorOutput,
    BatchLogDensityGradVal,
    BatchLogDensity,
    BatchQuadratureRule,
)

__all__ = ["MSIPFredholm", "MSIPQuadGradientFree", "MSIPQuadGradientInformed"]


class MSIPEstimator(ABC):
    @abstractmethod
    def get_v_evals(
        self, particles: BatchPtType, kernel_length_scale: float
    ) -> MSIPEstimatorOutput:
        """Function that returns estimation of $(log(v_0), -y + v_1 / v_0)$"""
        pass


class MSIPFredholm(MSIPEstimator):
    # gamma > 0, constant decay on the gradient (i.e., multiplies grad)
    gradient_decay: float
    log_dens_grad_val: BatchLogDensityGradVal

    def __init__(
        self, gradient_decay: float, log_dens_grad_val: BatchLogDensityGradVal
    ):
        self.gradient_decay = gradient_decay
        self.log_dens_grad_val = log_dens_grad_val

    def get_v_evals(self, particles, kernel_length_scale):
        grads, vals = self.log_dens_grad_val(particles)
        sigma_sq = kernel_length_scale * kernel_length_scale
        ret_v1_ratio = grads.mul_(sigma_sq * self.gradient_decay)
        return vals, ret_v1_ratio


vmap_recursive_weighted_average_alpha_v = torch.vmap(
    recursive_weighted_average_alpha_v, in_dims=(0, 0, 0)
)


class MSIPQuadGradientFree(MSIPEstimator):
    quadrature: BatchQuadratureRule
    log_dens: BatchLogDensity

    def __init__(
        self,
        log_dens: BatchLogDensity,
        quadrature: BatchQuadratureRule,
    ):
        self.quadrature = quadrature
        self.log_dens = log_dens

    def get_v_evals(self, particles, kernel_length_scale):
        n_particles, dim = particles.shape
        quad_pts, quad_wts = self.quadrature(n_particles)

        particle_quad_pts = quad_pts.mul_(kernel_length_scale).add(
            particles.reshape(n_particles, 1, -1)
        )
        log_dens_evals = self.log_dens(particle_quad_pts.reshape(-1, dim)).reshape(
            n_particles, -1
        )
        v1_ratio, log_v0 = vmap_recursive_weighted_average_alpha_v(
            quad_pts, quad_wts, log_dens_evals
        )

        return log_v0, v1_ratio


class MSIPQuadGradientInformed(MSIPEstimator):
    quadrature: BatchQuadratureRule
    gradient_decay: float
    log_dens_grad_val: BatchLogDensityGradVal

    def __init__(
        self,
        log_dens_grad_val: BatchLogDensityGradVal,
        quadrature: BatchQuadratureRule,
        gradient_decay: float,
    ):
        self.quadrature, self.gradient_decay = quadrature, gradient_decay
        self.log_dens_grad_val = log_dens_grad_val

    def get_v_evals(self, particles, kernel_length_scale):
        quad_pts, quad_wts = self.quadrature(particles.shape[0])
        sigma_sq = kernel_length_scale * kernel_length_scale
        particle_quad_pts = quad_pts.mul_(kernel_length_scale).add(
            particles.unsqueeze(1)
        )  # (N_part, N_quad, dim)
        log_dens_grads, log_dens_evals = self.log_dens_grad_val(
            particle_quad_pts.reshape(-1, particles.shape[1])
        )

        log_dens_grads = log_dens_grads.reshape_as(particle_quad_pts)
        log_dens_evals = log_dens_evals.reshape(particle_quad_pts.shape[:-1])

        v1_integrand = quad_pts.mul_(1 - self.gradient_decay).add_(
            log_dens_grads.mul_(self.gradient_decay * sigma_sq)
        )
        v1_ratio, log_v0 = vmap_recursive_weighted_average_alpha_v(
            v1_integrand, quad_wts, log_dens_evals
        )
        return log_v0, v1_ratio
