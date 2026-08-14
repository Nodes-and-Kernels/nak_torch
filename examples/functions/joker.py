import torch
from torch import Tensor
from dataclasses import dataclass
from functools import partial
from typing import Callable, Any, Optional

__any__ = ['logpdf', 'sample']


def joker_eye_logpdf(mean: Tensor, inv_std: Tensor, x: Tensor) -> Tensor:
    z = (x - mean) @ inv_std
    return -0.5*torch.sum(torch.square(z), dim=-1) + torch.log(torch.linalg.det(inv_std)) - 0.5*torch.sqrt(torch.pi*2*torch.ones(()))


def joker_smile_logpdf(mean: Tensor, inv_std: Tensor, x: Tensor) -> Tensor:
    z = (x - mean) @ inv_std
    z1 = z[..., 0]
    z2 = z[..., 1] - z1**2
    return -0.5*(z1**2 + z2**2) + torch.log(torch.linalg.det(inv_std)) - 0.5*torch.sqrt(torch.pi*2*torch.ones(()))


@dataclass
class JokerData:
    right_eye_shift = torch.tensor([4.,  10.])
    left_eye_shift = torch.tensor([-5.,  10.])
    right_eye_inv_stds = torch.diag(1 / torch.tensor([2.0, 2.0]))
    left_eye_inv_stds = torch.diag(1 / torch.tensor([1.0,  3.0]))
    smile_inv_stds = torch.diag(1 / torch.tensor([5.0,  3.0]))
    smile_shift = torch.tensor([0.0, -15.])
    face_shift = torch.tensor([0., 0.25])
    face_inv_scale = torch.linalg.inv(torch.tensor([
        [0.197907, 0.000539511],
        [0.000539511, 0.0911001]
    ]).contiguous())
    def to(self, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None):
        new = JokerData()
        new.right_eye_shift = self.right_eye_shift.to(dtype=dtype, device=device).contiguous()
        new.left_eye_shift = self.left_eye_shift.to(dtype=dtype, device=device).contiguous()
        new.right_eye_inv_stds = self.right_eye_inv_stds.to(dtype=dtype, device=device).contiguous()
        new.left_eye_inv_stds = self.left_eye_inv_stds.to(dtype=dtype, device=device).contiguous()
        new.smile_inv_stds = self.smile_inv_stds.to(dtype=dtype, device=device).contiguous()
        new.smile_shift = self.smile_shift.to(dtype=dtype, device=device).contiguous()
        new.face_shift = self.face_shift.to(dtype=dtype, device=device).contiguous()
        new.face_inv_scale = self.face_inv_scale.to(dtype=dtype, device=device).contiguous()
        return new
    def __repr__(self) -> str:
        return f"""{self.right_eye_shift}
{self.left_eye_shift}\n
{self.right_eye_inv_stds}\n
{self.left_eye_inv_stds}\n
{self.smile_inv_stds}\n
{self.smile_shift}\n
{self.face_shift}\n
{self.face_inv_scale}\n
"""


def joker_logpdf_full(data: JokerData, x, _ = None):
    x = (x - data.face_shift) @ data.face_inv_scale
    left_eye_eval = joker_eye_logpdf(
        data.left_eye_shift, data.left_eye_inv_stds, x
    )
    right_eye_eval = joker_eye_logpdf(
        data.right_eye_shift, data.right_eye_inv_stds, x
    )
    smile_eval = joker_smile_logpdf(data.smile_shift, data.smile_inv_stds, x)
    all_mode_evals = torch.stack((left_eye_eval, right_eye_eval, smile_eval))
    mix_logpdf = torch.logsumexp(all_mode_evals, 0)
    return torch.nan_to_num(mix_logpdf, nan=-1000)


def sample_eye(std: torch.Tensor, shift: torch.Tensor, sample: torch.Tensor):
    return sample @ std.to(sample.dtype) + shift.to(sample.dtype)


def sample_mouth(std: torch.Tensor, shift: torch.Tensor, sample: torch.Tensor):
    z = sample
    x_norm = torch.column_stack((z[..., 0], z[..., 1] + z[..., 0]**2))
    x = x_norm @ std.to(sample.dtype) + shift.to(sample.dtype)
    return x


def JokerSampler(data: JokerData) -> Callable[[Any, int], Any]:
    left_std = torch.linalg.inv(data.left_eye_inv_stds)
    right_std = torch.linalg.inv(data.right_eye_inv_stds)
    smile_std = torch.linalg.inv(data.smile_inv_stds)
    face_scale = torch.linalg.inv(data.face_inv_scale)
    left_eye = partial(sample_eye, left_std, data.left_eye_shift)
    right_eye = partial(sample_eye, right_std, data.right_eye_shift)
    mouth = partial(sample_mouth, smile_std, data.smile_shift)

    def sampler(rng: torch.Generator, N_samples: int) -> Tensor:
        z_samples = torch.normal(0., 1., size=(N_samples, 2), generator=rng, device=rng.device)
        which_modes_float = 3 * torch.rand(N_samples, generator=rng, device=rng.device)
        which_modes = torch.floor(which_modes_float).int()
        mode_0 = left_eye(z_samples[which_modes == 0])
        mode_1 = right_eye(z_samples[which_modes == 1])
        mode_2 = mouth(z_samples[which_modes == 2])
        return torch.concat((mode_0, mode_1, mode_2)) @ face_scale.to(z_samples.dtype) + data.face_shift.to(z_samples.dtype)

    return sampler


logpdf: Callable[[Tensor, Any], Tensor] = partial(joker_logpdf_full, JokerData())
sample: Callable[[torch.Generator, int], Tensor] = JokerSampler(JokerData())
def prior_sample(rng: torch.Generator, N_samples: int):
    return torch.normal(0., 2., size=(N_samples, 2), generator=rng, device=rng.device)

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