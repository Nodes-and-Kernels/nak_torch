# Various tools for the package
# Ayoub Belhadji
# 05/12/2025

import importlib.util
from . import kernel, types, quadrature, adaptive_step, metrics
from .average import recursive_weighted_average_alpha_v
from .torchify import differentiable_density_factory
from .types import GaussianModel, LogisticRegressionModel

__all__ = [
    "kernel",
    "types",
    "recursive_weighted_average_alpha_v",
    "differentiable_density_factory",
    "GaussianModel",
    "LogisticRegressionModel",
    "quadrature",
    "adaptive_step",
    "metrics",
]

if importlib.util.find_spec("pyro") is not None:
    from . import pyro_tools  # noqa: F401

    __all__.append("pyro_tools")

if importlib.util.find_spec("stanpy") is not None:
    from . import stan_tools  # noqa: F401

    __all__.append("stan_tools")
