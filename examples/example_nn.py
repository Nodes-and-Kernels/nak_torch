 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.nn.utils import vector_to_parameters
import matplotlib.pyplot as plt
from viz_tools import animate_trajectories_box
from functions import loss_nn_dataset
from nak_torch.algorithms import msip_greedy

from datetime import datetime

# Define the density
def eval_function_trajectories(the_objective_function,the_trajectories):
    T,M,d = the_trajectories.shape
    eval_tensor = torch.zeros((T,M))
    for t in range(T):
        for m in range(M):
            eval_tensor[t,m] = -the_objective_function(the_trajectories[t,m,:])
    return eval_tensor


def plot_eval_tensor(the_eval_tensor):
    _fig = plt.figure()
    T,M = the_eval_tensor.shape
    np_eval_tensor = the_eval_tensor.detach().numpy()
    for m in range(M):
        plt.plot(list(range(T)), np_eval_tensor[:,m])

    plt.show()

def plot_eval_best_tensor(the_eval_tensor):
    _fig = plt.figure()
    T,M = the_eval_tensor.shape
    np_eval_tensor = the_eval_tensor.detach().numpy()
    np_eval_min_tensor = np.min(np_eval_tensor,1)
    plt.plot(list(range(T)), np_eval_min_tensor)

    plt.show()

def plot_2D_classification_tensor(the_trajectories,dataset_name,m,t):
    plot_2D_classification_with_dataset_from_theta(the_trajectories[t,m,:],dataset_name, [-2,2], 50, device="cpu")


def plot_2D_classification_with_dataset_from_theta(theta,dataset_name, bounds, M_res, device="cpu"):
    a = bounds[0]
    b = bounds[1]
    M = M_res
    data = np.load(f"datasets/{dataset_name}.npz")
    x_train = torch.from_numpy(data["X"])
    y_train = torch.from_numpy(data["Y"]).view(-1)

    model = sigma_pi(2, 1, 10, 1, 'ReLU').to(device) # noqa: F821 # type: ignore
    vector_to_parameters(torch.from_numpy(theta), model.parameters())



    #funct = function_list[0]
    # Generate x and y values
    x = np.linspace(a, b, M)
    y = np.linspace(a, b, M)
    X, Y = np.meshgrid(x, y)  # Create a grid of x and y values

    # Calculate corresponding z values using the function
    Z = np.zeros((M,M))
    for m_1 in range(M):
        #print(m_1)
        for m_2 in range(M):
            z = torch.from_numpy(np.array((X[m_1,m_2],Y[m_1,m_2]))).float()
            z = torch.sign(model.forward(z)[0])

            Z[m_1,m_2] = z

    plt.figure(figsize=(10, 8))

    plt.imshow(Z, extent=[a, b, a, b], origin='lower')
    #plt.show()
    plt.colorbar(label='Z')
    plt.plot( x_train[0,y_train.flatten()>0], x_train[1,y_train.flatten()>0], 'b.' )
    plt.plot( x_train[0,y_train.flatten()<0], x_train[1,y_train.flatten()<0], 'r.' )

    plt.xlabel('X')
    plt.ylabel('Y')
    #plt.title('3D Heatmap Plot of the Three-Dimensional Function')
    plt.show()



if __name__ == "__main__":
    save_gif = False
    now_stamp = datetime.now()
    #algorithm_name = "msip_ni"
    algorithm_name = "msip"
    #function_name = "mixture_of_gaussians"
    #objective_function = mixture_of_gaussians

    function_name = "loss_nn_dataset"
    objective_function = loss_nn_dataset('two_bananas', beta = 2.0, lambda2=0.01, device="cpu")

    M = 5
    bounds = (-5., 5.)
    trajectories = msip_greedy(
        objective_function,
        n_particles=M,
        n_steps=300,          # now interpreted as "epochs" (passes over all particles)
        dim=60,
        bounds=bounds,
        projection = True,
        lr=0.5,
        noise=0.05,          # currently unused, kept for compatibility
        kernel_bandwidth=5.0,
        bandwidth_factor = 0.001,#0.001,
        inner_tol=1e-4,      # equilibrium tolerance for a particle
        max_inner_steps=15,  # max inner iterations per particle
        seed=None,
        device="cpu"
    )

    eval_tensor = eval_function_trajectories(objective_function,trajectories)
    plot_eval_tensor(eval_tensor)
    plot_eval_best_tensor(eval_tensor)
    T,_,_ = trajectories.shape

    for m in range(M):
        t_m = np.argmin(eval_tensor[:,m].detach().numpy())

        plot_2D_classification_tensor(trajectories,'two_bananas',m,t_m)


    if save_gif:
        fpath = f"results/gif/{algorithm_name}_{function_name}_particles_{now_stamp}.gif"
        animate_trajectories_box(
            objective_function,
            trajectories, 50, bounds,
            save_path=fpath
        )