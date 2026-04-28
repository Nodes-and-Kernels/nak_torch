import pytest
import numpy as np
import numpy.matlib as nm
import torch
import nak_torch

# Copied under MIT License from https://github.com/DartML/Stein-Variational-Gradient-Descent/blob/8d8f94974e1b91384dc44991ed5ad9a26212f136/python/bayesian_logistic_regression.py


def grad_logistic_regression_posterior(
    theta, data, labels, a0=1.0, b0=0.01, total_N=None
):
    if total_N is None:
        total_N = data.shape[0]
    Xs = data
    Ys = labels

    w = theta[:, :-1]  # logistic weights
    alpha = np.exp(theta[:, -1])  # the last column is logalpha
    d = w.shape[1]

    wt = np.multiply((alpha / 2), np.sum(w**2, axis=1))
    coff = np.matmul(Xs, w.T)
    y_hat = 1.0 / (1.0 + np.exp(-1 * coff))

    dw_data = np.matmul(
        ((nm.repmat(np.vstack(Ys), 1, theta.shape[0]) + 1) / 2.0 - y_hat).T, Xs
    )  # Y \in {-1,1}
    dw_prior = -np.multiply(nm.repmat(np.vstack(alpha), 1, d), w)
    dw = dw_data * float(total_N / Xs.shape[0]) + dw_prior  # re-scale

    dalpha = (
        d / 2.0 - wt + (a0 - 1) - b0 * alpha + 1
    )  # the last term is the jacobian term

    return np.hstack([dw, np.vstack(dalpha)])  # % first order derivative

def test_logistic_regression():
    N_DATA, N_PARTICLE, DIM = 100, 5, 2
    data = np.random.randn(N_DATA, DIM)
    labels = np.random.rand(N_DATA) > 0.5
    data_t, labels_t = torch.as_tensor(data), torch.as_tensor(labels)
    model = nak_torch.LogisticRegressionModel(data_t, labels_t, hyperprior_b=0.01)
    PROP_SUBSET = 0.2
    n_subset = int(N_DATA * PROP_SUBSET)
    data_subset, labels_subset = model.train_data[:n_subset], model.train_labels[:n_subset]
    theta = np.random.randn(N_PARTICLE, DIM + 1 + 1)
    labels_subset_np = labels_subset.numpy() * 2 - 1
    ref_grad_log_dens = grad_logistic_regression_posterior(theta, data_subset.numpy(), labels_subset_np, total_N = N_DATA)
    log_dens = model.to_log_dens(False)
    grad_log_dens_fcn = torch.func.grad(lambda x,a: log_dens(x,a).sum())
    theta_t = torch.as_tensor(theta)
    grad_log_dens = grad_log_dens_fcn(theta_t, (data_subset, labels_subset))
    assert grad_log_dens == pytest.approx(ref_grad_log_dens)
