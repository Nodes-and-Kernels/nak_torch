from dataclasses import asdict, dataclass, field
import math
import os
import random
import sys

from numpy.typing import ArrayLike
import torch
from torch import Tensor
from nak_torch.tools.types import BatchLogDensity, BatchType, BatchPtType, GaussianModel, KernelFunction
from nak_torch.algorithms.msip import estimators, MSIPFredholm, MSIPQuadGradientFree, MSIPQuadGradientInformed
from nak_torch.tools.kernel import stein_kernel_mat_factory
import problems
import pickle
import datetime
from typing import Any, Callable, Optional
from jaxtyping import Float
import nak_torch
from nak_torch.tools.util import sym_sqrtm

MeanType = Float[ArrayLike, " dim"]
CovType = Float[ArrayLike, "dim dim"]


@dataclass
class TestConfiguration:
    algorithm: str
    problem: str
    lr: float
    n_particles: int
    dim: int
    n_steps: int
    kernel_length_scale: Optional[float] = None
    device: str | None | torch.device = None
    kernel_diag_infl: Optional[float] = None
    inner_quad: Optional[str] = None
    inner_quad_kwargs: dict[str, Any] = field(default_factory=lambda: {})
    bounds: Optional[tuple[float, float]] = None
    test_length_scale: Optional[float] = None
    test_kernel: Optional[str] = None
    run_seed: Optional[int] = None
    gradient_decay: Optional[float] = None
    alg_kwargs: dict[str, Any] = field(default_factory=lambda: {})


alg_dict = nak_torch.algorithms.__dict__
inner_quad_dict = nak_torch.tools.quadrature.__dict__
msip_estimators = ["fredholm", "gradientfree", "gradientinformed"]
problem_dict: dict[str, Callable[[int], problems.Problem]] = problems.__dict__
uses_gaussian_model = ["gradfree_aldi", "eks"]


def check_msip_config_valid(config: TestConfiguration):
    infl = config.kernel_diag_infl

    if infl is None:
        raise ValueError("Missing required field: kernel_diag_infl")
    if infl < 0:
        raise ValueError(f"Invalid kernel_diag_infl {infl}")
    _, est = config.algorithm.split("_")
    if est not in msip_estimators:
        raise ValueError(f"Invalid MSIP estimator {est}")
    if est.lower() != "fredholm":
        quad = config.inner_quad
        if quad is None:
            raise ValueError("Missing inner_quad")
        if quad not in inner_quad_dict:
            raise ValueError(f"Invalid inner_quad value {quad}")
        inner_quad_dict[quad](
            2, **config.inner_quad_kwargs   # check quadrature
        )


def check_config_valid(config: TestConfiguration):
    alg = config.algorithm
    if not (alg.startswith('msip') or alg in alg_dict):
        raise ValueError(f"Missing algorithm {alg}")
    problem = config.problem
    if problem != 'test' and problem not in problem_dict:
        raise ValueError(f"Unrecognized problem {problem}")
    if alg.startswith('msip'):
        check_msip_config_valid(config)
    if alg == 'cbs':
        if 'inverse_temp' not in config.alg_kwargs:
            raise ValueError("Missing algorithm kwarg 'inverse_temp'")


def msip_estimator_factory(
    log_dens: BatchLogDensity,
    msip_estimator: Optional[str],
    inner_quad: Optional[str],
    gradient_decay: Optional[float],
    **inner_quad_kwargs
) -> estimators.MSIPEstimator:
    if msip_estimator is None:
        raise ValueError("Expected value for msip_estimator")
    def eval_log_dens(pts):
        out = log_dens(pts)
        return out.sum(), out
    grad_val_dens = torch.func.grad(eval_log_dens, has_aux=True)
    if msip_estimator.lower() == 'fredholm':
        if gradient_decay is None:
            raise ValueError(
                f"Estimator {msip_estimator} expects value gradient_decay")
        return MSIPFredholm(gradient_decay, grad_val_dens)
    else:
        if inner_quad is None:
            raise ValueError(
                f"Estimator {msip_estimator} expects value inner_quad")
        quad_fcn = inner_quad_dict[inner_quad]

        def quad(batch_sz: int):
            return quad_fcn(batch_sz, **inner_quad_kwargs)

        if msip_estimator == "gradientfree":
            return MSIPQuadGradientFree(log_dens, quad)
        elif msip_estimator == "gradientinformed":
            if gradient_decay is None:
                raise ValueError(
                    f"Estimator {msip_estimator} expects value gradient_decay")
            return MSIPQuadGradientInformed(grad_val_dens, quad, gradient_decay)
        else:
            raise ValueError(f"Unexpected estimator name: {msip_estimator}")


def get_bounds(bounds_str: Optional[str]) -> Optional[tuple[float, float]]:
    if bounds_str is None:
        return None
    if not (bounds_str.startswith("(") and bounds_str.endswith(")")):
        raise ValueError(
            f"Unexpected value {bounds_str} encountered for `bounds`."
            "Expect (lb, ub)"
        )
    bsplit = bounds_str[1:-1].split(";")
    if len(bsplit) != 2:
        raise ValueError(
            f"Unexpected value {bounds_str} encountered for `bounds`."
            "Expect (lb, ub)"
        )
    lb, ub = bsplit
    return (float(lb), float(ub))


def configuration_factory(config_dict: dict) -> TestConfiguration:
    alg_kwargs = {}
    alg_specific = {}
    device_name = config_dict.get("device", None)
    if device_name is None or device_name == "":
        device_name = "cpu"
    device = torch.device(device_name)
    torch.set_default_device(device)
    if device_name == 'mps':
        torch.set_default_dtype(torch.float32)
    alg_name = config_dict.get("algorithm", None)
    if alg_name is None:
        raise ValueError("Expected field `algorithm`")
    dim = config_dict.get("dim", None)
    if dim is None:
        raise ValueError("Expected field `dim`")
    inner_quad_kwargs = {}
    for (key, value) in config_dict.items():
        key_l = key.lower()
        if key_l not in TestConfiguration.__annotations__.keys():
            alg_kwargs[key] = value
        if key_l.startswith(alg_name.lower() + "_"):
            alg_specific[key[len(alg_name) + 1:]] = value
        if key_l.startswith("inner_quad_"):
            inner_quad_kwargs[key[len("inner_quad_"):]] = value
    for key in alg_kwargs.keys():
        config_dict.pop(key)
    for key in inner_quad_kwargs.keys():
        alg_kwargs.pop("inner_quad_" + key)
    inner_quad_kwargs['d'] = dim
    for (key,val) in alg_specific.items():
        config_dict[key] = val
    bounds = get_bounds(config_dict.pop("bounds", None))
    config = TestConfiguration(
        **config_dict, bounds=bounds,
        inner_quad_kwargs = inner_quad_kwargs,
        alg_kwargs = alg_kwargs
    )
    if config.kernel_length_scale is None:
        config.kernel_length_scale = math.sqrt(config.dim / config.n_particles)
    check_config_valid(config)
    if config.test_length_scale is None:
        config.test_length_scale = config.kernel_length_scale
    if config.run_seed is None:
        config.run_seed = random.randint(0, sys.maxsize >> 8)
    return config


def run_config(config: TestConfiguration, verbose: bool) -> tuple[problems.Problem, BatchLogDensity, BatchPtType, BatchType | None]:
    alg_name = config.algorithm.lower()
    if alg_name.startswith('msip'):
        alg_name = 'msip'
    alg = alg_dict[alg_name]
    dim = config.dim
    problem = problem_dict[config.problem + "_logpdf"](dim)
    model = problem.model
    if not isinstance(model, GaussianModel) and alg_name in uses_gaussian_model:
        raise ValueError(
            f"Invalid problem {config.problem} for algorithm {alg_name}")
    seed = config.run_seed
    if seed is None:
        raise ValueError("Invalid seed.")
    rng = torch.Generator(torch.get_default_device())
    config.device = torch.get_default_device()
    rng = rng.manual_seed(seed)
    init_particles = problem.prior_sample(rng, config.n_particles)
    if isinstance(model, GaussianModel):
        log_dens = nak_torch.tools.types.gaussian_log_dens_factory(model)
    else:
        log_dens = model

    kwargs = {
        'init_particles': init_particles,
        'keep_all': False,
        'rng': rng,
        'verbose': verbose,
        'is_log_density_batched': True,
        **config.__dict__,
        **config.alg_kwargs
    }
    if alg_name == 'msip':
        _, est_name = config.algorithm.lower().split("_")
        estimator = msip_estimator_factory(
            log_dens, est_name, config.inner_quad, config.gradient_decay, **config.inner_quad_kwargs
        )
        pts, wts = alg(estimator, **kwargs)
        return problem, log_dens, pts[-1], wts[-1]
    else:  # Not weighted
        problem_input = model if alg_name in uses_gaussian_model else log_dens
        pts = alg(problem_input, **kwargs)
        return problem, log_dens, pts[-1], None


@dataclass
class TestOutput:
    points: Float[ArrayLike, "batch dim"]
    weights: Float[ArrayLike, " batch"] | None
    ksd: float
    mean: Float[ArrayLike, " dim"]
    cov: Float[ArrayLike, "dim dim"]
    rmse: float | None
    foerstner: float | None
    spectral: float | None
    sample_sq_mmd: float | None


def get_stein_mat_fcn(log_dens: BatchLogDensity, kernel_elem: KernelFunction):
    grad_log_dens = torch.func.grad(lambda p: log_dens(p).sum())
    return stein_kernel_mat_factory(grad_log_dens, kernel_elem, is_grad_vectorized=True)


def get_moments(
    pts: BatchPtType, wts: Optional[BatchType],
    reference_samples: Optional[BatchPtType]
) -> tuple[MeanType, CovType, Optional[float], Optional[float], Optional[float]]:
    if wts is None:
        p_mean = pts.mean(0)
        p_cov = pts.T.cov()
    else:
        p_mean = wts @ pts
        p_cov = torch.einsum("b,bi,bj", wts, pts, pts)
    if reference_samples is None:
        rmse, foerstner, spectral = None, None, None
    else:
        ref_mean = reference_samples.mean(0)
        ref_cov = reference_samples.T.cov()
        ref_cov_inv_sqrt = sym_sqrtm(ref_cov, use_inv=True)
        rmse = torch.linalg.norm(p_mean - ref_mean).item()
        gen_eigvals: Tensor = torch.linalg.eigvalsh(
            ref_cov_inv_sqrt @ p_cov @ ref_cov_inv_sqrt
        )
        foerstner = gen_eigvals.log_().square_().sum().sqrt().item()
        spectral = torch.linalg.norm(ref_cov - p_cov, ord=2).item()

    p_mean = p_mean.cpu().numpy()
    p_cov = p_cov.cpu().numpy()
    return p_mean, p_cov, rmse, foerstner, spectral


def get_ksd(
    log_dens: BatchLogDensity,
    pts: BatchPtType,
    wts: Optional[BatchType],
    kernel_elem: KernelFunction,
    test_kernel_length_scale: float
) -> float:
    stein_mat = get_stein_mat_fcn(log_dens, kernel_elem)
    stein_mat_eval = stein_mat(pts, test_kernel_length_scale, pts)
    ksd: Float
    if wts is None:
        ksd = torch.sqrt(stein_mat_eval.sum() / (pts.shape[0]**2))
    else:
        ksd = torch.sqrt((stein_mat_eval @ wts) @ wts)
    return ksd.item()

@torch.compile
def get_mmd(
    pts: BatchPtType,
    wts: Optional[BatchType],
    kernel_elem: KernelFunction,
    test_kernel_length_scale: float,
    ref_samples: Optional[BatchPtType],
) -> Optional[float]:
    kernel_mat = nak_torch.tools.kernel.matricize_kernel_elem(kernel_elem)
    if ref_samples is None:
        return None
    ref_mmd = kernel_mat(ref_samples, test_kernel_length_scale).mean()
    cross_mmd_vec = kernel_mat(pts, test_kernel_length_scale, ref_samples).mean(1)
    cross_mmd: Float
    pts_mmd: Float
    if wts is None:
        cross_mmd = cross_mmd_vec.mean()
        pts_mmd = kernel_mat(pts, test_kernel_length_scale).mean()
    else:
        cross_mmd = wts @ cross_mmd_vec
        K_mat = kernel_mat(pts, test_kernel_length_scale)
        pts_mmd = (K_mat @ wts) @ wts
    return (ref_mmd - 2*cross_mmd + pts_mmd).item()


# def get_self_mmd(
#     test_kernel_length_scale: float,
#     ref_samples: BatchPtType, chunk_size: int,
#     kernel_mat: MatSelfKernelFunction, N_ref: int,
#     N_chunks: int
# ) -> Float:
#     @torch.compile
#     def self_mmd_chunk(i: int, j: int):
#         min_idx_i, max_idx_i = i*chunk_size, min((i + 1)*chunk_size, N_ref)
#         min_idx_j, max_idx_j = j*chunk_size, min((j + 1)*chunk_size, N_ref)
#         samples_chunk_i = ref_samples[min_idx_i:max_idx_i]
#         samples_chunk_j = ref_samples[min_idx_j:max_idx_j]
#         K_mat =
#         mul = 1 if i == j else 2
#         return mul * K_mat.mean()

#     # prog = tqdm(range( (N_chunks * (N_chunks + 1)) // 2 ))
#     self_mmd = torch.zeros((), device=ref_samples.device, dtype=ref_samples.dtype)
#     for i in range(N_chunks):
#         for j in range(i, N_chunks):
#             self_mmd += self_mmd_chunk(i, j)
#             # prog.update()
#     return self_mmd


def process_output(
    log_dens: BatchLogDensity,
    pts: BatchPtType,
    wts: Optional[BatchType],
    kernel_elem: KernelFunction,
    test_kernel_length_scale: float,
    ref_samples: Optional[BatchPtType]
):
    ksd = get_ksd(log_dens, pts, wts, kernel_elem, test_kernel_length_scale)
    mmd = get_mmd(pts, wts, kernel_elem, test_kernel_length_scale, ref_samples)
    mean, cov, rmse, foerstner, spectral = get_moments(pts, wts, ref_samples)
    pts_np = pts.cpu().numpy()
    if wts is not None:
        wts_np = wts.cpu().numpy()
    else:
        wts_np = None
    return TestOutput(pts_np, wts_np, ksd, mean, cov, rmse, foerstner, spectral, mmd)


all_kernels = nak_torch.tools.kernel.kernel_elem_dict


def get_kernel_elem(test_kernel: str) -> KernelFunction:
    return all_kernels[test_kernel]


def run_single_test(configuration: dict, data_dir: str, verbose: bool):
    config = configuration_factory(configuration)
    if config.test_length_scale is None:
        raise ValueError("Expected test_length_scale to be set")
    if config.test_kernel is None:
        raise ValueError("Expected test_kernel to be set")
    if config.problem == "test":
        print(config)
        return
    prob, log_dens, pts, wts = run_config(config, verbose)
    kernel_elem = get_kernel_elem(config.test_kernel)

    out = process_output(log_dens, pts, wts, kernel_elem,
                         config.test_length_scale, prob.reference_samples)
    experiment_run = {"config": asdict(config), "output": asdict(out)}
    fname = f"{config.algorithm}_{config.problem}_{datetime.datetime.now()}"
    fname = '-'.join(fname.split(' '))
    save_name = os.path.join(data_dir, f"{fname}.pkl")
    with open(save_name, "wb") as f:
        pickle.dump(experiment_run, f)
