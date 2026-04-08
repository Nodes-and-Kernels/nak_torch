import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.nn.utils import vector_to_parameters
from torch.func import functional_call
from collections import OrderedDict

# ══════════════════════════════════════════════════════════════════════════════
# BNN model
# ══════════════════════════════════════════════════════════════════════════════



class bnn(nn.Module):
    """
    Standard MLP for binary classification.
    Output is a raw logit; apply sigmoid to get probabilities.

    Parameters
    ----------
    d_in       : int   input dimension
    hidden_dim : int   width of each hidden layer
    n_layers   : int   number of hidden layers
    """
    def __init__(self, d_in: int, hidden_dim: int, n_layers: int = 1):
        super().__init__()
        layers = []
        in_dim = d_in
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        return self.net(xb).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset loading with train-test splitting
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(dataset_name, train_ratio=0.8, seed=0):
    """
    Loads  datasets/{dataset_name}.npz  (keys X: (d,N), Y: (1,N))
    and splits into train / test.

    Returns X_train, Y_train, X_test, Y_test as torch.double tensors
    with shapes (N, d) and (N,)
    """
    data  = np.load(f"datasets/{dataset_name}.npz")
    X = torch.from_numpy(data["X"].T).double()
    Y = torch.from_numpy(data["Y"].T).double().squeeze()

    N_total = X.shape[0]
    rng     = np.random.RandomState(seed)
    idx     = rng.permutation(N_total)
    n_train = int(N_total * train_ratio)

    i_train, i_test = idx[:n_train], idx[n_train:]

    return X[i_train], Y[i_train], X[i_test], Y[i_test]


# ══════════════════════════════════════════════════════════════════════════════
# Objective functions
# ══════════════════════════════════════════════════════════════════════════════

def make_objective(X, Y, model_class='bnn',
                   hidden_dim=10, n_layers=1,
                   beta=1.0, lambda2=0.01):
    """
    Returns a log-posterior callable  theta -> scalar

    Parameters
    ----------
    X, Y         : torch.double tensors  (N, d) and (N,)
    model_class  : 'bnn' for now, but we can think about something else
    hidden_dim   : int   hidden width
    n_layers     : int   depth
    beta         : float temperature
    lambda2      : float prior weight
    """
    d = X.shape[1]

    if model_class == 'bnn':
        model = bnn(d_in=d, hidden_dim=hidden_dim, n_layers=n_layers).double()
    else:
        raise ValueError(f"Unknown model_class: {model_class!r}")

    param_info  = [(name, p.shape, p.numel())
                   for name, p in model.named_parameters()]
    buffer_dict = OrderedDict(model.named_buffers())
    total_numel = sum(nu for _, _, nu in param_info)

    print(f"  Model   : {model_class}  |  "
          f"hidden_dim = {hidden_dim}  n_layers = {n_layers}  |  "
          f"dim(theta) = {total_numel}")

    def theta_to_param_dict(theta_1d):
        out, i = OrderedDict(), 0
        for name, shape, numel in param_info:
            out[name] = theta_1d[i:i + numel].view(shape)
            i += numel
        return out

    def loss_for_single_theta(theta_1d):
        param_dict = theta_to_param_dict(theta_1d)
        pred       = functional_call(model, (param_dict, buffer_dict), (X,))
        pred       = pred.squeeze()
        data_loss  = soft_margin_loss(pred, Y)
        reg        = lambda2 * (theta_1d ** 2).sum()
        return -(data_loss + reg) / beta

    def objective_function(theta):
        if theta.ndim == 1:
            return loss_for_single_theta(theta)
        elif theta.ndim == 2:
            return torch.stack([loss_for_single_theta(theta[i])
                                for i in range(theta.shape[0])])
        else:
            raise ValueError(f"theta must be 1D or 2D, got {theta.shape}")


    objective_function.total_numel = total_numel
    objective_function.model_class = model_class
    objective_function.model       = model
    objective_function.param_info  = param_info
    objective_function.buffer_dict = buffer_dict

    return objective_function



# ══════════════════════════════════════════════════════════════════════════════
# Tools for handling grids
# ══════════════════════════════════════════════════════════════════════════════

def _make_grid(bounds, M_res):
    a, b   = bounds
    xs, ys = np.linspace(a, b, M_res), np.linspace(a, b, M_res)
    Xg, Yg = np.meshgrid(xs, ys)
    grid   = torch.tensor(np.stack([Xg.ravel(), Yg.ravel()], 1),
                          dtype=torch.double)
    return Xg, Yg, grid


def _get_ensemble_probs(traj, obj_fn, grid):
    particles = traj[-1]
    model     = obj_fn.model
    probs     = []
    for i in range(len(particles)):
        vector_to_parameters(particles[i].double(), model.parameters())
        with torch.no_grad():
            probs.append(torch.sigmoid(model(grid).squeeze()).numpy())
    return np.stack(probs, axis=0)


def _scatter(ax, X, Y):
    x = X.numpy()
    y = Y.numpy()
    ax.scatter(x[y > 0, 0], x[y > 0, 1], c='gold',   s=25, zorder=5)
    ax.scatter(x[y < 0, 0], x[y < 0, 1], c='tomato', s=25, zorder=5)



# ══════════════════════════════════════════════════════════════════════════════
# Visualization tools
# ══════════════════════════════════════════════════════════════════════════════

def plot_boundaries(trajectories_dict, objective_fns_dict,
                      X_tr, Y_tr, bounds=[-0.1, 1.1], M_res=60):
    Xg, Yg, grid = _make_grid(bounds, M_res)
    M = len(trajectories_dict)
    fig, axes = plt.subplots(1, M, figsize=(6*M, 5))
    if M == 1: axes = [axes]

    for ax, (name, traj) in zip(axes, trajectories_dict.items()):
        probs = _get_ensemble_probs(traj, objective_fns_dict[name], grid)
        for pi in probs:
            ax.contour(Xg, Yg, np.sign(pi - 0.5).reshape(M_res, M_res),
                       levels=[0], colors=['steelblue'], linewidths=0.6, alpha=0.3)
        p_bar = probs.mean(0).reshape(M_res, M_res)
        ax.contour(Xg, Yg, p_bar, levels=[0.5], colors=['black'], linewidths=2.0)
        _scatter(ax, X_tr, Y_tr)
        ax.set_title(name, fontsize=13, fontweight='bold')
        ax.set_xlim(*bounds)
        ax.set_ylim(*bounds)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')

    plt.suptitle("Decision boundary (final iteration)", fontsize=14)
    plt.savefig("de_vs_msip_boundaries.pdf")
    plt.tight_layout()
    plt.show()



def plot_mean_prediction(trajectories_dict, objective_fns_dict,
                          X_tr, Y_tr, bounds=[-0.1, 1.1], M_res=60):
    Xg, Yg, grid = _make_grid(bounds, M_res)
    M = len(trajectories_dict)
    fig, axes = plt.subplots(1, M, figsize=(6*M, 5))
    if M == 1: axes = [axes]

    for ax, (name, traj) in zip(axes, trajectories_dict.items()):
        probs = _get_ensemble_probs(traj, objective_fns_dict[name], grid)
        p_bar = probs.mean(0).reshape(M_res, M_res)
        im = ax.imshow(p_bar, extent=[*bounds, *bounds], origin='lower',
                       cmap='RdBu_r', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.contour(Xg, Yg, p_bar, levels=[0.5], colors=['white'], linewidths=2.0)
        _scatter(ax, X_tr, Y_tr)
        ax.set_title(f"{name}", fontsize=12,
                     fontweight='bold')
        ax.set_xlabel('x1'); ax.set_ylabel('x2')

    plt.suptitle("Mean predictive probability", fontsize=14)
    plt.savefig("de_vs_msip_mean_predictive_p.pdf")
    plt.tight_layout()
    plt.show()


def plot_diversity_curve(trajectories_dict, subsample=10):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, traj in trajectories_dict.items():
        T, N, D = traj.shape
        steps, divs = [], []
        for t in range(0, T, subsample):
            P   = traj[t]
            sq  = ((P.unsqueeze(0) - P.unsqueeze(1)) ** 2).sum(-1)
            idx = torch.triu_indices(N, N, offset=1)
            divs.append(sq[idx[0], idx[1]].sqrt().min().item())
            steps.append(t)
        ax.plot(steps, divs, lw=2, label=name)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Smallest pairwise distance")
    ax.set_title("Particle diversity over training")
    plt.savefig("de_vs_msip_div.pdf")
    ax.legend()
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Evaluations on the dataset
# ══════════════════════════════════════════════════════════════════════════════


def evaluate(trajectories_dict, objective_fns_dict, X, Y, split_name="test"):
    y_true = (Y.numpy() > 0)
    grid   = X.double()

    print(f"\n{'─'*75}")
    print(f"  Evaluation on {split_name} set  ({len(X)} points)")
    print(f"{'─'*75}")
    print(f"  {'Method':<12} {'Accuracy':>6} {'DAMV':>10}")
    print(f"{'─'*75}")

    results = {}
    for name, traj in trajectories_dict.items():
        probs      = _get_ensemble_probs(traj, objective_fns_dict[name], grid)
        p_bar      = probs.mean(0)
        acc        = ((p_bar > 0.5) == y_true).mean()
        damv       = traj[-1].var(dim=0).mean().item()

        print(f"  {name:<12} {acc:>6.3f} {damv:>10.3f}")
        results[name] = dict(p_bar=p_bar, accuracy=acc, damv=damv)

    print(f"{'─'*75}\n")
    return results


def plot_diversity_curve(trajectories_dict):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, traj in trajectories_dict.items():
        T, N, D = traj.shape
        steps, divs = [], []
        for t in range(0, T, 1):
            P   = traj[t]
            sq  = ((P.unsqueeze(0) - P.unsqueeze(1)) ** 2).sum(-1)
            idx = torch.triu_indices(N, N, offset=1)
            divs.append(sq[idx[0], idx[1]].sqrt().min().item())
            steps.append(t)
        ax.plot(steps, divs, lw=2, label=name)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Smallest pairwise distance")
    ax.set_title("Particle diversity over training")
    plt.savefig("de_vs_msip_div.pdf")
    ax.legend()
    plt.tight_layout()
    plt.show()



def eval_function_trajectories(obj_fn, trajectories, algo_name):
    T, M, d = trajectories.shape
    eval_tensor = torch.zeros(T, M)
    for t in range(T):
        for m in range(M):
            eval_tensor[t, m] = -obj_fn(trajectories[t, m, :])


    plt.plot(eval_tensor.detach().numpy().min(1), label=algo_name)
    plt.xlabel("Iteration")
    plt.ylabel("Objective function")
    plt.title("Evaluation of the best particle for "+algo_name)
    plt.savefig("best_particle_"+algo_name+".pdf")
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.show()


def soft_margin_loss(x,y):
    return torch.mean(F.softplus(-x * y))