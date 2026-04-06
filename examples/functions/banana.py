import torch
from jaxtyping import Float
from torch import Tensor
from dataclasses import dataclass
from functools import partial
from typing import Callable, Any, Optional

__any__ = ['logpdf', 'sample']

@dataclass
class RosenbrockData:
    shift_x: Float[Tensor, "1"] = torch.zeros(()) # For more typical rosenbrock, torch.ones(())
    scale_y: Float[Tensor, "1"] = torch.ones(()) # For more typical rosenbrock, 2 * torch.ones(())
    shift: Float[Tensor, "2"] = torch.zeros(2)
    inv_std: Float[Tensor, "2 2"] = torch.eye(2)
    def to(self, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):
        new = RosenbrockData()
        new.shift_x = self.shift_x.to(dtype=dtype, device=device).contiguous()
        new.scale_y = self.scale_y.to(dtype=dtype, device=device).contiguous()
        new.shift = self.shift.to(dtype=dtype, device=device).contiguous()
        new.inv_std = self.inv_std.to(dtype=dtype, device=device).contiguous()
        return new
    def __repr__(self) -> str:
        return f"""{self.shift_x}
{self.scale_y}\n
{self.shift}\n
{self.inv_std}\n
"""


def rosenbrock_logpdf_full(data: RosenbrockData, x):
    x = (x - data.shift) @ data.inv_std
    term_0 = torch.square_(data.shift_x - x[...,0])
    term_1 = torch.square_(x[...,1] - x[...,0]**2) * data.scale_y
    return - (term_1 + term_0)

def RosenbrockSampler(data: RosenbrockData) -> Callable[[torch.Generator, int], Float[Tensor, "N 2"]]:
    full_scale = torch.linalg.inv(data.inv_std)

    def sampler(rng: torch.Generator, N_samples: int) -> Tensor:
        z_samples = torch.normal(0., 1., size=(N_samples,2), generator=rng, device=rng.device)
        dtype = z_samples.dtype
        x_samples_0 = z_samples[:,0] + data.shift_x
        x_samples_1 = (z_samples[:,1] / data.scale_y) + x_samples_0**2
        return torch.column_stack((x_samples_0, x_samples_1)) @ full_scale.to(dtype) + data.shift.to(dtype)

    return sampler

logpdf: Callable[[Tensor], Tensor] = partial(rosenbrock_logpdf_full, RosenbrockData())
sample: Callable[[torch.Generator, int], Tensor] = RosenbrockSampler(RosenbrockData())

def prior_sample(rng: torch.Generator, N_samples: int):
    return torch.normal(0., 2., size=(N_samples, 2), generator=rng, device=rng.device)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dat = RosenbrockData(shift_x=torch.zeros(()), scale_y=torch.ones(()))
    samps = sample(torch.default_generator, 10000)

    N_plt = 1000
    x_min, x_max, y_min, y_max = samps[:,0].min(), samps[:,0].max(), samps[:,1].min(), samps[:,1].max()
    x, y = torch.linspace(x_min, x_max, N_plt), torch.linspace(y_min, y_max, N_plt)
    X,Y = torch.meshgrid(x, y, indexing='ij')
    grid = torch.column_stack((X.flatten(), Y.flatten()))
    out_like = logpdf(grid).reshape(N_plt, N_plt)
    sig_pr = 2
    out_pr = -(X**2 + Y**2) / (2*sig_pr**2)
    out = out_like + 0.*out_pr
    fig, axs = plt.subplots(1,2, figsize=(10,5), sharex=True, sharey=True)
    axs[0].contourf(X, Y, out.exp(), levels=100)
    axs[1].scatter(samps[:,0], samps[:,1], c="k", alpha=0.01)
    plt.show()
    print(samps.T.cov())