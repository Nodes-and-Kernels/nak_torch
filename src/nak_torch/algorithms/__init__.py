# Optimization, sampling and quantization algorithms
# msip: mean shift interacting particles
# wfr-ips: interacting particles that follow the Wasserstein-Fisher-Rao gradient flow
# cbo: consensus-based optimization


# Ayoub Belhadji
# 05/12/2025

from .eks import eks
from .msip import MSIP, MSIPGS
from .svgd import SVGD
from .deepensembles import deepensembles
from .grad_aldi import GradALDI
from .gradfree_aldi import GradFreeALDI
from .cbs import CBS
from .kfrflow import kfrflow
from .loop import nak


__all__ = [
    "nak",
    "MSIP",
    "MSIPGS",
    "SVGD",
    "deepensembles",
    "GradALDI",
    # "gradfree_aldi",
    "GradFreeALDI",
    "eks",
    "CBS",
    "kfrflow",
]
