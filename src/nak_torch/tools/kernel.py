import torch
from typing import Optional, Callable
from jaxtyping import Float
from torch import Tensor
from .types import (
    BatchType,
    KernelFunction,
    MatSelfKernelFunction,
    PtType,
    BatchPtType,
    KernelMatrixType,
    GradLogDensity,
    BatchGradLogDensity,
    GradKernelMatrixType,
)

__all__ = [
    "default_kernel_matrix",
    "sqexp_kernel_matrix",
    "sqexp_kernel_elem",
    "matricize_kernel_elem",
    "stein_kernel_mat_factory",
    "kernel_optimal_weight_factory",
    "kernel_grad_and_value_factory",
    "kernel_elem_dict",
]


def sqexp_kernel_matrix(
    pts: BatchPtType, kernel_length_scale: float, pts2: Optional[BatchPtType] = None
) -> KernelMatrixType:
    if pts2 is None:
        pts2 = pts
    diff = pts.unsqueeze(1) - pts2.unsqueeze(0)  # (N, N, d)

    return torch.exp(
        -(diff**2).sum(dim=-1) / (2 * kernel_length_scale * kernel_length_scale)
    )


default_kernel_matrix = sqexp_kernel_matrix


def sqexp_kernel_elem(x: PtType, y: PtType, kernel_length_scale: float) -> Float:
    torch._assert(
        x.shape == y.shape and y.ndim == 1, "Invalid input dimensions of x and y"
    )
    ret = torch.exp(
        -(x - y).square().sum() / (2 * kernel_length_scale * kernel_length_scale)
    )
    return ret


def matricize_kernel_elem(
    kernel: KernelFunction, use_compiled: bool = True
) -> MatSelfKernelFunction:
    r"""
    Vectorize elementwise kernel(pt1, pt2, length_scale) and return a function kernel_mat(pt, length_scale[, pt2=pt])
    """
    kernel_v: MatSelfKernelFunction = torch.vmap(
        torch.vmap(kernel, in_dims=(0, None, None), out_dims=0),
        in_dims=(None, 0, None),
        out_dims=1,
    )

    def kernel_self_v(pts, kernel_length_scale, pts2=None):
        return kernel_v(pts, pts if pts2 is None else pts2, kernel_length_scale)

    if use_compiled:
        kernel_self_v = torch.compile(kernel_self_v)

    return kernel_self_v


def kernel_optimal_weight_factory(
    pts: BatchPtType,
    log_dens_evals: BatchType,
    kernel_matrix: KernelMatrixType,
) -> BatchType:
    v0 = (log_dens_evals - log_dens_evals.max()).exp_()
    wts = torch.linalg.solve(kernel_matrix, v0)
    return wts.div_(wts.sum())


def stein_kernel_diffs_factory(
    kernel_fcn: KernelFunction,
) -> Callable[
    [BatchPtType, BatchPtType, float],
    tuple[KernelMatrixType, Float[Tensor, "batch batch d"], KernelMatrixType],
]:

    kernel_grad = torch.func.grad_and_value(kernel_fcn, argnums=0)

    def process_kernel_grad(x, y, length_scale):
        grad, aux = kernel_grad(x, y, length_scale)
        return grad, (grad, aux)

    kernel_jac = torch.func.jacrev(process_kernel_grad, has_aux=True, argnums=1)

    def process_kernel_jac(x, y, length_scale):
        jac, (grad, ev) = kernel_jac(x, y, length_scale)
        return torch.trace(jac), grad, ev

    kernel_diffs = torch.vmap(
        torch.vmap(process_kernel_jac, in_dims=(0, None, None), out_dims=0),
        in_dims=(None, 0, None),
        out_dims=1,
    )
    return kernel_diffs


def stein_kernel_mat_factory(
    grad_log_p: GradLogDensity | BatchGradLogDensity,
    kernel_fcn: KernelFunction,
    is_grad_vectorized: bool = False,
    use_compiled: bool = True,
) -> MatSelfKernelFunction:
    kernel_diffs = stein_kernel_diffs_factory(kernel_fcn)
    grad_log_p_v = grad_log_p if is_grad_vectorized else torch.vmap(grad_log_p)

    def stein_kernel_mat(
        pts: BatchPtType, kernel_length_scale: float, pts2: Optional[BatchPtType] = None
    ) -> KernelMatrixType:
        grad_log_p_eval1 = grad_log_p_v(pts)
        if pts2 is None:
            pts2 = pts
            grad_log_p_eval2 = grad_log_p_eval1
        else:
            grad_log_p_eval2 = grad_log_p_v(pts2)
        trace_kernel, grad1_kernel, eval_kernel = kernel_diffs(
            pts, pts2, kernel_length_scale
        )
        grad_term_1 = torch.einsum("ijd,jd->ij", grad1_kernel, grad_log_p_eval2)
        grad_term_2 = grad_term_1.T
        ev_term = torch.einsum(
            "ij,id,jd->ij", eval_kernel, grad_log_p_eval1, grad_log_p_eval2
        )
        return trace_kernel + grad_term_1 + grad_term_2 + ev_term

    return torch.compile(stein_kernel_mat) if use_compiled else stein_kernel_mat


def kernel_grad_and_value_factory(
    kernel_elem: KernelFunction, which_argnum: int, *kernel_args
) -> Callable[
    [BatchPtType, BatchPtType], tuple[GradKernelMatrixType, KernelMatrixType]
]:
    kernel_grad_val = torch.func.grad_and_value(
        lambda x, y: kernel_elem(x, y, *kernel_args), argnums=which_argnum
    )
    kernel_grad_val_vec = torch.vmap(
        torch.vmap(kernel_grad_val, in_dims=(None, 0)), in_dims=(0, None)
    )
    return kernel_grad_val_vec


kernel_elem_dict = {"sqexp": sqexp_kernel_elem}
