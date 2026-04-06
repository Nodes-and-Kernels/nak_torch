import pytest
import torch
from nak_torch.tools.kernel import sqexp_kernel_elem, sqexp_kernel_matrix, kernel_grad_and_value_factory

FD_DELTA = 1e-7

def test_kernel_grad_val_argnum0():
    # Check [grad_1 𝛋(X, Y)]_{ijk} = 𝟃_{x_ik} 𝛋(x_i, y_j)
    GRAD_ARGNUM, bandwidth = 0, 3.231
    k_grad_val = kernel_grad_and_value_factory(sqexp_kernel_elem, GRAD_ARGNUM, bandwidth)
    pts = torch.ones((2, 3)) * torch.tensor([1., 2.77], dtype=torch.float64).reshape(-1, 1)
    pts2 = torch.ones((5, 3)) * torch.tensor([3.43, -1.2, 1.52321, -5.1, -1.7], dtype=torch.float64).reshape(-1, 1)
    grad, val = k_grad_val(pts, pts2)
    actual_val = sqexp_kernel_matrix(pts, bandwidth, pts2)
    assert val == pytest.approx(actual_val)
    pts_fd = pts.clone()
    grad_fd = 1000*torch.ones_like(grad)
    for pt_idx in range(pts_fd.shape[0]):
        pt, pt_fd = pts[pt_idx], pts_fd[pt_idx]
        for pt2_idx in range(pts2.shape[0]):
            pt2 = pts2[pt2_idx]
            grad_fd_pt = grad_fd[pt_idx, pt2_idx]
            ev = sqexp_kernel_elem(pt, pt2, bandwidth)
            for dim_idx in range(pts_fd.shape[1]):
                pt_fd[dim_idx] += FD_DELTA
                ev_fd = sqexp_kernel_elem(pt_fd, pt2, bandwidth)
                grad_fd_pt[dim_idx] = (ev_fd - ev)/FD_DELTA
                pt_fd[dim_idx] -= FD_DELTA
    assert grad == pytest.approx(grad_fd, rel=50*FD_DELTA)


def test_kernel_grad_val_argnum1():
    # Check [grad_2 𝛋(X, Y)]_{ijk} = 𝟃_{y_jk} 𝛋(x_i, y_j)
    GRAD_ARGNUM, bandwidth = 1, 3.231
    k_grad_val = kernel_grad_and_value_factory(sqexp_kernel_elem, GRAD_ARGNUM, bandwidth)
    pts = torch.ones((5, 3)) * torch.tensor([3.43, -1.2, 1.52321, -5.1, -1.7], dtype=torch.float64).reshape(-1, 1)
    pts2 = torch.ones((2, 3)) * torch.tensor([1., 2.77], dtype=torch.float64).reshape(-1, 1)
    grad, val = k_grad_val(pts, pts2)
    actual_val = sqexp_kernel_matrix(pts, bandwidth, pts2)
    assert val == pytest.approx(actual_val)
    pts2_fd = pts2.clone()
    grad_fd = 1000*torch.ones_like(grad)
    for pt_idx in range(pts.shape[0]):
        pt = pts[pt_idx]
        for pt2_idx in range(pts2_fd.shape[0]):
            pt2, pt2_fd = pts2[pt2_idx], pts2_fd[pt2_idx]
            grad_fd_pt2 = grad_fd[pt_idx, pt2_idx]
            ev = sqexp_kernel_elem(pt, pt2, bandwidth)
            for dim_idx in range(pts2_fd.shape[1]):
                pt2_fd[dim_idx] += FD_DELTA
                ev_fd = sqexp_kernel_elem(pt, pt2_fd, bandwidth)
                grad_fd_pt2[dim_idx] = (ev_fd - ev)/FD_DELTA
                pt2_fd[dim_idx] -= FD_DELTA
    assert grad == pytest.approx(grad_fd, rel=50*FD_DELTA)
