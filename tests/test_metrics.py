import os
import math
import pickle

import pytest
import torch
from torch import Tensor
from nak_torch import metrics
from nak_torch.tools import kernel

MAX_POW10 = 4

def normal_logpdf(x: Tensor, _ = None):
    return x.square().sum(-1).neg().div(2)

def grad_normal_logpdf(x: Tensor, _ = None):
    return x.neg()

def test_ksd():
    KSD_KERNEL_ELEM = kernel.inverse_multi_quadric_kernel_elem
    rng = torch.Generator()
    rng.manual_seed(321393021)
    kernel_length_scale = 0.1
    ksd = metrics.KernelSteinDiscrepancy(grad_normal_logpdf, kernel_length_scale, kernel_elem=KSD_KERNEL_ELEM)
    in_sizes = torch.arange(MAX_POW10 + 1)
    # 15 is empirical, does not really matter.
    expected_ksd = 15 * torch.pow(10.0, -0.5 * in_sizes)
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
    rets = torch.tensor(
        [
            cross_entropy_v(torch.randn((N_trial, 10**j, 2), generator=rng)).mean()
            for j in range(MAX_POW10)
        ]
    )
    qoi = rets.std() / rets.mean()
    # Cross entropy has no explicit sample size dependence
    assert qoi < 1e-2


def test_relative_ess():
    rng = torch.Generator()
    rng.manual_seed(2380123)
    ress = metrics.RelativeESS(normal_logpdf)
    ress_v = torch.vmap(ress)
    N_trial = 1000
    rets = torch.tensor(
        [
            ress_v(torch.randn((N_trial, 10**j, 2), generator=rng)).mean()
            for j in range(1, MAX_POW10)
        ]
    )
    qoi = rets.std() / rets.mean()
    assert qoi < 1e-2

@pytest.fixture
def sample_mmd_file():
    import tempfile
    f = tempfile.NamedTemporaryFile()
    yield f.name

def test_sample_mmd(sample_mmd_file):
    rng = torch.Generator()
    rng.manual_seed(2380123)
    N_REF, N_TARGET = 10000, 1000
    USE_COMPILED = False
    kernel_lengthscale = 0.1
    ref_samples = torch.randn((N_REF, 2), generator=rng)
    mmd_1 = metrics.SampleMMD(ref_samples, kernel_lengthscale, use_compiled=USE_COMPILED)
    target_samples = torch.randn((N_TARGET, 2), generator=rng)
    mmd_1_eval = mmd_1(target_samples).item()
    assert mmd_1_eval < (10 / math.sqrt(N_REF))
    mmd_1_eval_wt = mmd_1(target_samples, torch.ones(N_TARGET) / N_TARGET).item()
    assert mmd_1_eval_wt == pytest.approx(mmd_1_eval)
    mmd_2 = metrics.SampleMMD(ref_samples, kernel_lengthscale, self_mmd = mmd_1.self_mmd, use_compiled=USE_COMPILED)
    assert mmd_2(target_samples).item() == pytest.approx(mmd_1_eval)
    mmd_serial = metrics.SampleMMD(ref_samples, kernel_lengthscale, self_mmd_serial = sample_mmd_file, use_compiled=USE_COMPILED)
    assert mmd_serial(target_samples).item() == pytest.approx(mmd_1_eval)
    with open(sample_mmd_file, "rb") as f:
        d = pickle.load(f)
        assert kernel_lengthscale in d.keys()
        assert d[kernel_lengthscale] == mmd_1.self_mmd
    mmd_serial_2 = metrics.SampleMMD(ref_samples, kernel_lengthscale, self_mmd_serial = sample_mmd_file, use_compiled=USE_COMPILED)
    assert mmd_serial_2(target_samples).item() == pytest.approx(mmd_1_eval)

