# Optimization, sampling and quantization algorithms
# msip: mean shift interacting particles
# wfr-ips: interacting particles that follow the Wasserstein-Fisher-Rao gradient flow
# cbo: consensus-based optimization


# Ayoub Belhadji
# 05/12/2025

from .eks import eks
from .msip import msip, msip_ni, msip_greedy, msip_geom_greedy, msip_adapt
from .svgd import svgd
from .deepensembles import deepensembles
from .grad_aldi import grad_aldi
from .gradfree_aldi import gradfree_aldi
from .cbs import cbs
from .kfrflow import kfrflow


__all__ = [
    "msip",
    "msip_ni",
    "msip_greedy",
    "msip_geom_greedy",
    "msip_adapt",
    "svgd",
    "deepensembles",
    "grad_aldi",
    "gradfree_aldi",
    "eks",
    "cbs",
    "kfrflow",
]
