from typing import Optional
from abc import ABC, abstractmethod
from jaxtyping import Float

import torch
from torch import Tensor
from .types import (
    BatchGradLogDensity,
    BatchLogDensityGradVal,
    BatchType,
    BatchPtType,
    BatchLogDensity,
    GradLogDensity,
    LogDensity,
    LogDensityGradVal,
    MatSelfKernelFunction,
)

from .kernel import sqexp_kernel_elem, stein_kernel_mat_factory

__all__ = ["InclusiveKullbackLeibler", "KernelSteinDiscrepancy", "RelativeESS"]


class Metric(ABC):
    @abstractmethod
    def __call__(self, pts: BatchPtType, wts: Optional[BatchType] = None) -> Float:
        pass


AnyLogDensEval = (
    LogDensity | LogDensityGradVal | BatchLogDensity | BatchLogDensityGradVal
)


class GradFreeMetric(Metric):
    log_dens: BatchLogDensity | LogDensity
    is_log_dens_vectorized: bool

    def __init__(
        self,
        log_dens: AnyLogDensEval,
        is_log_dens_vectorized: bool = True,
        is_log_dens_grad_val: bool = False,
    ):
        log_dens_val = (
            log_dens if not is_log_dens_grad_val else lambda x: log_dens(x)[1]
        )
        self.log_dens = log_dens_val
        self.is_log_dens_vectorized = is_log_dens_vectorized


class InclusiveKullbackLeibler(GradFreeMetric):
    r"""
    Given target $\pi$ and particle approximation $\mu$, estimate $D_{KL}(\mu || \pi)$.
    """

    def __call__(self, pts, wts=None):
        N = pts.shape[0]
        N_tens = torch.as_tensor(N, device=pts.device, dtype=pts.dtype)
        kl: Float
        if self.is_log_dens_vectorized:
            log_dens_evals = self.log_dens(pts)
            if wts is None:
                entropy = -torch.log(N_tens)
                cross_entropy = log_dens_evals.mean()
                kl = entropy - cross_entropy
            else:
                entropy = (wts.log() * wts).sum()
                cross_entropy = log_dens_evals @ wts
                kl = entropy - cross_entropy
        else:
            kl = torch.zeros_like(N_tens)
            for idx in range(pts.shape[0]):
                cross_entropy_eval = self.log_dens(pts[idx])
                entropy_eval: Float
                if wts is None:
                    cross_entropy_eval /= N_tens
                    entropy_eval = -torch.log(N_tens) / N_tens
                else:
                    cross_entropy_eval *= wts[idx]
                    entropy_eval = wts[idx].log() * wts[idx]
                kl += entropy_eval - cross_entropy_eval
        return kl


class ExclusiveKullbackLeibler(GradFreeMetric):
    r"""
    Given target $\pi$ and particle approximation $\mu$, estimate $D_{KL}(\pi || \mu)$ using importance sampling.
    DO NOT USE. MATHEMATICALLY INCORRECT.
    """

    def __call__(self, pts, wts=None):
        N = pts.shape[0]
        N_tens = torch.as_tensor(N, device=pts.device, dtype=pts.dtype)
        log_dens_evals: Tensor
        if self.is_log_dens_vectorized:
            log_dens_evals = self.log_dens(pts)
        else:
            log_dens_evals = torch.zeros(N, device=pts.device, dtype=pts.dtype)
            for idx in range(pts.shape[0]):
                log_dens_evals[idx] = self.log_dens(pts[idx])
        kl: Float
        if wts is None:
            log_ratios = log_dens_evals - N_tens.log()
            kl = torch.mean(log_ratios * torch.exp(log_ratios))
        else:
            wts /= wts.sum()
            log_ratios = log_dens_evals - wts.log()
            kl = (log_ratios * torch.exp(log_ratios)) @ wts
        return kl


class RelativeESS(GradFreeMetric):
    r"""
    Calculate the relative effective sample size (rESS) as
    $$rESS(Y,w; \pi) = \frac{1}{\sum_i v_i^2},  v_i = \frac{w_i \pi(y_i)}{\sum_j w_j \pi(y_j)}$.$
    """

    def __call__(self, pts, wts=None):
        evals: torch.Tensor
        N = pts.shape[0]
        if self.is_log_dens_vectorized:
            evals = self.log_dens(pts)
        else:
            evals = torch.zeros(N, device=pts.device, dtype=pts.dtype)
            for idx in range(N):
                evals[idx] = self.log_dens(pts[idx])

        log_weights = evals
        if wts is not None:
            if torch.any(wts < 0.0):
                raise ValueError("Expected nonnegative weights")
            log_weights = log_weights.add_(wts.log())
        log_norm_wts = torch.log_softmax(log_weights, dim=0)
        return torch.logsumexp(2 * log_norm_wts, dim=0).neg_().exp_().div_(N)


AnyLogDensGrad = GradLogDensity | BatchGradLogDensity


class KernelSteinDiscrepancy(Metric):
    """Implementation of Kernel Stein Discrepancy."""

    stein_kernel_mat: MatSelfKernelFunction
    kernel_length_scale: float

    def __init__(
        self,
        grad_log_dens: AnyLogDensGrad,
        kernel_length_scale: float,
        kernel_elem=None,
        is_grad_vectorized: bool = True,
        use_compiled: bool = False,
    ):
        if kernel_elem is None:
            kernel_elem = sqexp_kernel_elem
        self.kernel_length_scale = kernel_length_scale
        self.stein_kernel_mat = stein_kernel_mat_factory(
            grad_log_dens,
            kernel_elem,
            is_grad_vectorized=is_grad_vectorized,
            use_compiled=use_compiled,
        )

    def __call__(self, pts, wts=None):
        stein_mat = self.stein_kernel_mat(pts, self.kernel_length_scale)
        if wts is None:
            return stein_mat.mean().sqrt()
        else:
            return (wts @ stein_mat @ wts).sqrt()
