import os
from typing import Callable, Optional
from abc import ABC, abstractmethod
import pickle
from warnings import warn
from jaxtyping import Float

import torch
from torch import Tensor
import pickle
from .types import (
    BatchLogDensityGrad,
    BatchLogDensityGradVal,
    BatchType,
    BatchPtType,
    BatchLogDensity,
    LogDensityGrad,
    KernelFunction,
    KernelMatrixType,
    LogDensity,
    LogDensityGradVal,
    MatSelfKernelFunction,
    PtType,
)

from .kernel import (
    DEFAULT_KERNEL_ELEM,
    matricize_kernel_elem,
    sqexp_kernel_elem,
    stein_kernel_mat_factory,
)

__all__ = ["CrossEntropy", "KernelSteinDiscrepancy", "RelativeESS"]


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
            log_dens if not is_log_dens_grad_val else lambda x, a: log_dens(x, a)[1]
        )
        self.log_dens = log_dens_val
        self.is_log_dens_vectorized = is_log_dens_vectorized


class CrossEntropy(GradFreeMetric):
    r"""
    Given target $\pi$ and particle approximation $\mu$, estimate $D_{KL}(\mu || \pi)$.
    """

    def __call__(self, pts, wts=None, target_args=None):
        N = pts.shape[0]
        N_tens = torch.as_tensor(N, device=pts.device, dtype=pts.dtype)
        cross_entropy: Float
        if self.is_log_dens_vectorized:
            log_dens_evals = self.log_dens(pts, target_args)
            if wts is None:
                cross_entropy = -log_dens_evals.mean()
            else:
                cross_entropy = -log_dens_evals @ wts
        else:
            cross_entropy = torch.zeros_like(N_tens)
            for idx in range(pts.shape[0]):
                cross_entropy_eval = self.log_dens(pts[idx], target_args)
                if wts is None:
                    cross_entropy_eval /= N_tens
                else:
                    cross_entropy_eval *= wts[idx]
                cross_entropy -= cross_entropy_eval
        return cross_entropy


class ExclusiveKullbackLeibler(GradFreeMetric):
    r"""
    Given target $\pi$ and particle approximation $\mu$, estimate $D_{KL}(\pi || \mu)$ using importance sampling.
    DO NOT USE. MATHEMATICALLY INCORRECT.
    """

    def __call__(self, pts, wts=None, target_args=None):
        warn("Exclusive Kullback Leibler is not mathematically correct.")
        N = pts.shape[0]
        N_tens = torch.as_tensor(N, device=pts.device, dtype=pts.dtype)
        log_dens_evals: Tensor
        if self.is_log_dens_vectorized:
            log_dens_evals = self.log_dens(pts, target_args)
        else:
            log_dens_evals = torch.zeros(N, device=pts.device, dtype=pts.dtype)
            for idx in range(pts.shape[0]):
                log_dens_evals[idx] = self.log_dens(pts[idx], target_args)
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

    def __call__(self, pts, wts=None, target_args=None):
        evals: torch.Tensor
        N = pts.shape[0]
        if self.is_log_dens_vectorized:
            evals = self.log_dens(pts, target_args)
        else:
            evals = torch.zeros(N, device=pts.device, dtype=pts.dtype)
            for idx in range(N):
                evals[idx] = self.log_dens(pts[idx], target_args)

        log_weights = evals
        if wts is not None:
            if torch.any(wts < 0.0):
                raise ValueError("Expected nonnegative weights")
            log_weights = log_weights.add_(wts.log())
        log_norm_wts = torch.log_softmax(log_weights, dim=0)
        return torch.logsumexp(2 * log_norm_wts, dim=0).neg_().exp_().div_(N)


AnyLogDensGrad = LogDensityGrad | BatchLogDensityGrad


class KernelSteinDiscrepancy(Metric):
    """Implementation of Kernel Stein Discrepancy."""

    stein_kernel_mat: MatSelfKernelFunction
    kernel_length_scale: float

    def __init__(
        self,
        grad_log_dens: AnyLogDensGrad,
        kernel_length_scale: float,
        target_args=None,
        kernel_elem=None,
        is_grad_vectorized: bool = True,
        use_compiled: bool = False,
    ):
        if kernel_elem is None:
            kernel_elem = DEFAULT_KERNEL_ELEM
        self.kernel_length_scale = kernel_length_scale
        self.stein_kernel_mat = stein_kernel_mat_factory(
            grad_log_dens,
            kernel_elem,
            target_args=target_args,
            is_grad_vectorized=is_grad_vectorized,
            use_compiled=use_compiled,
        )

    def __call__(self, pts, wts=None):
        stein_mat = self.stein_kernel_mat(pts, self.kernel_length_scale)
        if wts is None:
            return stein_mat.mean().sqrt()
        else:
            return (wts @ stein_mat @ wts).sqrt()


class SampleMMD(Metric):
    kernel_sample_reduce: Callable[
        [BatchPtType], BatchType
    ]  # k(X,y,sigma), X as vector
    kernel_mat: Callable[[BatchPtType], KernelMatrixType]
    self_mmd: float
    kernel_lengthscale: float

    def __init__(
        self,
        samples: BatchPtType,
        kernel_lengthscale: float,
        kernel_elem: Optional[KernelFunction] = None,
        self_mmd: Optional[Float] = None,
        self_mmd_serial: Optional[str] = None,
        use_compiled: bool = True,
    ):
        if kernel_elem is None:
            kernel_elem = DEFAULT_KERNEL_ELEM
        kernel_vec = torch.vmap(kernel_elem, in_dims=(0, None, None))

        def _kernel_reduce(y: PtType) -> Float:
            return kernel_vec(samples, y, kernel_lengthscale).mean()

        kernel_reduce_elem: Callable[[PtType], Float]
        if use_compiled:
            kernel_reduce_elem = torch.compile(_kernel_reduce)
        else:
            kernel_reduce_elem = _kernel_reduce
        kernel_reduce = torch.vmap(kernel_reduce_elem)
        self.kernel_sample_reduce = kernel_reduce
        kernel_mat = matricize_kernel_elem(kernel_elem, use_compiled)
        self.kernel_mat = lambda pts: kernel_mat(pts, kernel_lengthscale)
        if self_mmd is None:
            if self_mmd_serial is None:
                self.self_mmd = kernel_reduce(samples).mean()
            else:
                import fcntl  # TODO: Fix for windows, I guess

                self_mmd_dict: dict[float, Float] = {}
                file_existed_prev = os.path.exists(self_mmd_serial)
                with open(self_mmd_serial, "rb+") as f:
                    if file_existed_prev:
                        try:
                            self_mmd_dict = pickle.load(f)
                        except EOFError:
                            self_mmd_dict = {}
                self_mmd_pkl: float
                if kernel_lengthscale in self_mmd_dict.keys():
                    self_mmd_pkl = self_mmd_dict[kernel_lengthscale]
                else:
                    self_mmd_pkl = kernel_reduce(samples).mean()
                    # Ensure file is locked and up-to-date in case things are written in parallel
                    with open(self_mmd_serial, "wb+") as f:
                        fcntl.lockf(f, fcntl.LOCK_EX)
                        try:
                            new_self_mmd_dict = pickle.load(f)
                        except EOFError:
                            new_self_mmd_dict = self_mmd_dict
                        self_mmd_dict[kernel_lengthscale] = self_mmd_pkl
                        pickle.dump(new_self_mmd_dict, f)
                        fcntl.lockf(f, fcntl.LOCK_UN)
                    # End lock
                self.self_mmd = self_mmd_pkl
        else:
            self.self_mmd = self_mmd

    def __call__(self, pts, wts=None):
        self_mmd = self.self_mmd
        cross_kernel_vec = self.kernel_sample_reduce(pts)
        pts_kernel_mat = self.kernel_mat(pts)
        cross_mmd: Float
        pts_mmd: Float
        if wts is None:
            cross_mmd = cross_kernel_vec.mean()
            pts_mmd = pts_kernel_mat.mean()
        else:
            cross_mmd = cross_kernel_vec @ wts
            pts_mmd = wts @ (pts_kernel_mat @ wts)
        return torch.sqrt(self_mmd - 2 * cross_mmd + pts_mmd)
