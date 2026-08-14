# Kernel-based quantization algorithms

## Installation
We recommend installing with `uv`. Currently, the way to install this locally would be
```bash
$ uv pip install -e git+https://github.com/Nodes-and-Kernels/nak_torch
```

If you plan on using the examples, make sure that `[examples]` option is installed. Also, make sure that there is no other installation of `pystan`, which is a dependency---we use a fork of the original package to reduce latency for our algorithms when using a stan posterior.

## List of Algorithms
### MSIP
We largely focus on _mean-shift interacting particle_ (MSIP) algorithms, and we are working to implement several of these. Currently, we have:

- MSIP
- MSIPGS

For these algorithms, we have multiple estimators---each of these produces a certain set of dynamics. In particular, we have:

- MSIPFredholm
- MSIPGradientFree
- MSIPGradientInformed
- MSIPGMMGaussianKernel

### Other algorithms
We also include several other typical interacting-particle sampling algorithms.

- Consensus-based sampler (`CBS`)
- Deep ensembles (`DeepEnsembles`)
- Ensemble Kalman Sampler (`EKS`)
- Gradient-informed affine-invariant Langevin dynamics (`GradALDI`)
- Gradient-free ALDI (`GradFreeALDI`)
- Stein variational gradient descent (`SVGD`)

