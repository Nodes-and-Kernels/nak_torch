# Various functions for illustrations
# twodims: 2D functions
# nns: neural networks
# bayes: posterior distributions for 'realistic' problems
# kde: kernel density estimator, useful to go from discrete to continuous

# Ayoub Belhadji
# 05/12/2025


from . import banana, himmelblau
from .aristoff_bangerth import build_aristoff_bangerth

__all__ = [
    "banana",
    "himmelblau",
    "build_aristoff_bangerth"
]
