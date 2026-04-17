import pytest
import torch
from torch import Tensor
from nak_torch import metrics
from nak_torch.tools import kernel

MAX_POW10 = 4

def normal_logpdf(x: Tensor):
    return x.square().sum(-1).neg().div(2)

def test_ksd():
    grad_log_normal = torch.neg
    KSD_KERNEL_ELEM = kernel.inverse_multi_quadric_kernel_elem
    rng = torch.Generator()
    rng.manual_seed(321393021)
    kernel_length_scale = 0.1
    ksd = metrics.KernelSteinDiscrepancy(grad_log_normal, kernel_length_scale, kernel_elem=KSD_KERNEL_ELEM)
    in_sizes = torch.arange(MAX_POW10 + 1)
    # 15 is empirical, does not really matter.
    expected_ksd = 15*torch.pow(10.0, -0.5 * in_sizes)
    outs_unweighted = []
    outs_weighted = []
    for j in range(MAX_POW10 + 1):
        N_pts = 10**j
        pts = torch.randn((N_pts, 2), generator=rng)
        outs_unweighted.append(ksd(pts))
        wts = torch.ones(N_pts) / N_pts
        outs_weighted.append(ksd(pts, wts))
    ksd_unweighted = torch.tensor(outs_unweighted)
    ksd_weighted = torch.tensor(outs_weighted)
    dev_unweighted = torch.std(ksd_unweighted / expected_ksd)
    dev_weighted = torch.std(ksd_weighted / expected_ksd)
    assert dev_unweighted < 0.05
    assert dev_weighted < 0.05

def test_cross_entropy():
    rng = torch.Generator()
    rng.manual_seed(10429102)
    cross_entropy = metrics.CrossEntropy(normal_logpdf)
    cross_entropy_v = torch.vmap(cross_entropy)
    N_trial = 1000
    rets = torch.tensor([cross_entropy_v(torch.randn((N_trial, 10**j, 2), generator=rng)).mean() for j in range(MAX_POW10)])
    qoi = rets.std() / rets.mean()
    # Cross entropy has no explicit sample size dependence
    assert qoi < 1e-2

def test_relative_ess():
    rng = torch.Generator()
    rng.manual_seed(2380123)
    ress = metrics.RelativeESS(normal_logpdf)
    ress_v = torch.vmap(ress)
    N_trial = 1000
    rets = torch.tensor([ress_v(torch.randn((N_trial, 10**j, 2), generator=rng)).mean() for j in range(1,MAX_POW10)])
    qoi = rets.std() / rets.mean()
    assert qoi < 1e-2
