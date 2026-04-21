import torch
from abc import abstractmethod
from nak_torch.tools.average import recursive_weighted_average_alpha_v
from nak_torch.tools.types import (
    MSIPEstimatorOutput,
    BatchLogDensityGradVal,
    BatchLogDensity,
    BatchQuadratureRule,
    BatchDensityEvaluator,
)
from jaxtyping import Float
from torch import Tensor

__all__ = ["MSIPFredholm", "MSIPQuadGradientFree", "MSIPQuadGradientInformed"]


class MSIPEstimator(BatchDensityEvaluator[MSIPEstimatorOutput]):
    @abstractmethod
    def __call__(self, particles, evaluator_args, target_args) -> MSIPEstimatorOutput:
        r"""
        Function that returns estimation of $(\log(v_0), sigma^2 * \nabla \log v_0(y)$
        Note that
        $$
        \sigma^2 \nabla \log v_0(y) = \frac{v_1(y)}{v_0(y)} - y.
        $$
        """
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

    def __call__(self, particles, kernel_length_scale, target_args):
        grads, v0 = self.log_dens_grad_val(particles, target_args)
        sigma_sq_log_v0 = grads.mul_(kernel_length_scale * self.gradient_decay)
        return v0, sigma_sq_log_v0


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

    def __call__(self, particles, kernel_length_scale, target_args):
        n_particles, dim = particles.shape
        quad_pts, quad_wts = self.quadrature(n_particles)

        particle_quad_pts = quad_pts.mul_(kernel_length_scale).add(
            particles.reshape(n_particles, 1, -1)
        )
        log_dens_evals = self.log_dens(
            particle_quad_pts.reshape(-1, dim), target_args
        ).reshape(n_particles, -1)
        sigma_sq_score_v0, log_v0 = vmap_recursive_weighted_average_alpha_v(
            quad_pts, quad_wts, log_dens_evals
        )

        return log_v0, sigma_sq_score_v0


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

    def __call__(self, particles, kernel_length_scale, target_args):
        quad_pts, quad_wts = self.quadrature(particles.shape[0])
        particle_quad_pts = quad_pts.mul_(kernel_length_scale).add(
            particles.unsqueeze(1)
        )  # (N_part, N_quad, dim)
        log_dens_grads, log_dens_evals = self.log_dens_grad_val(
            particle_quad_pts.reshape(-1, particles.shape[1]), target_args
        )

        log_dens_grads = log_dens_grads.reshape_as(particle_quad_pts)
        log_dens_evals = log_dens_evals.reshape(particle_quad_pts.shape[:-1])

        v1_integrand = quad_pts.mul_(1 - self.gradient_decay).add_(
            log_dens_grads.mul_(self.gradient_decay * kernel_length_scale)
        )  # Note that previously multiplied particle_quad_pts by kernel_length_scale
        sigma_sq_score_v0, log_v0 = vmap_recursive_weighted_average_alpha_v(
            v1_integrand, quad_wts, log_dens_evals
        )
        return log_v0, sigma_sq_score_v0


class MSIPGMMGaussianKernel(MSIPEstimator):
    weights: Float[Tensor, " K"]
    means: Float[Tensor, "K dim"]
    covariances: Float[Tensor, "K dim dim"]
    bandwidth: float

    def __init__(
        self,
        weights: torch.Tensor,
        means: torch.Tensor,
        covariances: torch.Tensor,
        bandwidth: float,
    ):
        self.weights = weights
        self.means = means
        self.covariances = covariances
        self.bandwidth = bandwidth

    def __call__(self, particles, kernel_length_scale, _) -> MSIPEstimatorOutput:
        N, D = particles.shape
        dtype, device = particles.dtype, particles.device
        sigma_sq = torch.as_tensor(
            kernel_length_scale * kernel_length_scale, device=device, dtype=dtype
        )
        use_cholesky_upper = False

        # Calculate the smoothed covariances and related quantities
        eyes = torch.eye(D, device=device, dtype=dtype)  # (K, D, D)
        smoothed_covs = self.covariances + sigma_sq * eyes.unsqueeze(0)  # (K, D, D)
        # __ Calculate the Cholesky decompositions of all the smoothed covs
        L = torch.linalg.cholesky(smoothed_covs, upper=use_cholesky_upper)  # (K, D, D)
        # __ Calculate the log-normalisation per component:
        # __  -0.5*(D*log(2pi) + log|C_k|) = -0.5*(D*log(2pi) + 2\sum log|L_k|_ii)
        log_det = 2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(-1)  # (K,)
        log_norm = -0.5 * log_det

        # Calculating the distances rescaled by the covariance matrices
        deltas = particles.unsqueeze(1) - self.means.unsqueeze(0)  # (N, K, D)
        L_exp = L.unsqueeze(0).expand(N, -1, -1, -1)  # (N, K, D, D)
        diff_col = deltas.unsqueeze(-1)  # (N, K, D, 1)
        # __ Calculate the differences rescales by the smoothed covariances
        z = torch.linalg.solve_triangular(
            L_exp, diff_col, upper=use_cholesky_upper
        )  # (N, K, D, 1)
        sq_mahal_distances = z.squeeze(-1).square().sum(-1)  # (N, K)

        # Calculating log v_0
        log_w = self.weights.log().unsqueeze(0)  # (1, K)
        # __ Calculate the evaluations of each Gaussian for all the particles
        log_g = log_norm.unsqueeze(0) - 0.5 * sq_mahal_distances  # (N, K)
        # __ Calculate the evaluation of log_v0,
        # __ without the normalizing constant of the Gaussian kernel
        log_sigma = 0.5 * D * torch.log(sigma_sq)
        log_v0 = torch.logsumexp(log_w + log_g, dim=1) + log_sigma  # (N,)

        # Calculating grad_log_v0
        # __ Computing r_{n,k} = (w_k g_k(x_n)) / sum_j (w_j g_j(x_n))
        # __ in a stable way: r_{n,k} = softmax(log w_k + log g_k(x_n))
        r = torch.softmax(log_w + log_g, dim=1)  # (N, K)

        LT_inv_neg_z: torch.Tensor = torch.linalg.solve_triangular(
            L_exp.mT,
            z.neg(),
            upper=not use_cholesky_upper,  # since we transpose the last two dimensions of L_exp.
        )  # (N, K, D, 1)
        score_components = LT_inv_neg_z.squeeze(-1)  # (N, K, D)
        score_v0 = (r.unsqueeze(-1) * score_components).sum(dim=1)  # (N, D)
        sigma_sq_score_v0 = score_v0.mul_(kernel_length_scale**2)
        return log_v0, sigma_sq_score_v0
