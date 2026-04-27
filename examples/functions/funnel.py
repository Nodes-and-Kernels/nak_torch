import numpy as np
import torch
from torch import Tensor
from dataclasses import dataclass
from functools import partial
from typing import Callable

__any__ = ['logpdf_factory', 'sample_factory']

@dataclass
class FunnelData:
    dim: int
    x1_var: float = 9.
    def __repr__(self) -> str:
        return f"dim: {self.dim}, x1_var: {self.x1_var}"


def funnel_logpdf_full(data: FunnelData, x: Tensor):
    x1 = x[...,0]
    x2_end = x[...,1:]
    logpdf_x1 = x1.square().div_(data.x1_var).add_(np.log(2 * np.pi * data.x1_var))
    exp_x1 = x1.exp()
    logpdf_x2_end = x2_end.square().sum(-1).divide_(exp_x1).add_(np.log(2 * np.pi)).add_((data.dim - 1) * x1)
    return -0.5 * (logpdf_x1 + logpdf_x2_end)

def logpdf_factory(dim: int, **funnel_kwargs) -> Callable[[Tensor], Tensor]:
    if dim < 2:
        raise ValueError(f"Expected parameter dim > 1. Got {dim}")
    return partial(funnel_logpdf_full, FunnelData(dim, **funnel_kwargs))

def sample_factory(dim: int, **funnel_kwargs)->Callable[[torch.Generator, int], Tensor]:
    data = FunnelData(dim, **funnel_kwargs)
    x1_std: float = np.sqrt(data.x1_var)
    def sampler(rng: torch.Generator, N_samples: int) -> Tensor:
        x1 = x1_std * torch.randn(N_samples, generator=rng)
        x2_end_std = x1.mul(0.5).exp_().unsqueeze_(1)
        x2_end = torch.randn((N_samples, dim-1), generator=rng).mul_(x2_end_std)
        return torch.column_stack((x1, x2_end))

    return sampler

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    N_plt = 1000
    x = y = torch.linspace(-3, 3, N_plt)
    X,Y = torch.meshgrid(x, y, indexing='ij')
    grid = torch.column_stack((X.flatten(), Y.flatten()))
    out_like = logpdf(grid).reshape(N_plt, N_plt)
    sig_pr = 2
    out_pr = -(X**2 + Y**2) / (2*sig_pr**2)
    out = out_like + out_pr
    plt.contourf(X, Y, out.exp(), levels=20)
    plt.gca().set_aspect(1.0)
    plt.show()
    samps = sample(torch.default_generator, 10000)
    print(samps.T.cov())