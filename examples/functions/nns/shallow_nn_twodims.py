#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# This file contains a typical 2D mixture of Gaussians with isotropic
# covariance matrices
# Ayoub Belhadji
# 05/12/2025


import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import math
from torch.nn.utils import vector_to_parameters




class sigma_pi(nn.Module):
    def __init__(self,d,M,N,C,activation):
        super().__init__()
        self.vs = nn.Parameter(torch.randn(d,M,N,C) / math.sqrt(d))
        self.alphas = nn.Parameter(torch.randn(M,N,C))
        self.betas = nn.Parameter(torch.randn(M,N,C))
        self.bias = nn.Parameter(torch.zeros(M,N,C))
        self.ws = nn.Parameter(torch.randn(N,C))
        self.activation = activation
        self.d = d
        self.C = C
        self.M = M
        self.N = N
    def forward(self, xb):
        if self.activation == 'ReLU':
            nn_eval = torch.zeros(xb.shape[0],self.C)
            #print(np.linalg.norm(self.ws.detach().numpy()))
            for c in list(range(self.C)):
                for n in list(range(self.N)):
                    tmp_var = 1
                    for m in list(range(self.M)):
                        #print("xb.shape         =", xb.shape)
                        #print("vs_slice.shape   =", self.vs[:, m, n, c].shape)
                        #print("bias.shape   =", self.bias.shape)
                        tmp_var = tmp_var*(self.alphas[m,n,c]+ self.betas[m,n,c]*nn.ReLU()(xb @ self.vs[:,m,n,c] + self.bias[m,n,c]))
                    nn_eval[:,c] += self.ws[n,c]*tmp_var
        elif self.activation == 'Tanh':
            nn_eval = torch.zeros(xb.shape[0],self.C)
            #print(np.linalg.norm(self.ws.detach().numpy()))
            for c in list(range(self.C)):
                for n in list(range(self.N)):
                    tmp_var = 1
                    for m in list(range(self.M)):
                        #print("xb.shape         =", xb.shape)
                        #print("vs_slice.shape   =", self.vs[:, m, n, c].shape)
                        #print("bias.shape   =", self.bias.shape)
                        tmp_var = tmp_var*(self.alphas[m,n,c]+ self.betas[m,n,c]*nn.Tanh()(xb @ self.vs[:,m,n,c] + self.bias[m,n,c]))
                    nn_eval[:,c] += self.ws[n,c]*tmp_var
        return nn_eval






def loss_nn_dataset(dataset_name, beta = 1.0, lambda2=0.01, device="cpu"):
    # --- Load dataset once ---
    data = np.load(f"datasets/{dataset_name}.npz")
    X = torch.from_numpy(data["X"].T).float().to(device)
    Y = torch.from_numpy(data["Y"].T).float().to(device)

    loss_func = F.soft_margin_loss

    # --- Build the model once ---
    model = sigma_pi(2, 1, 10, 1, 'ReLU').to(device)

    def loss_for_single_theta(theta_1d: torch.Tensor) -> torch.Tensor:
        """theta_1d: (d,) -> scalar loss"""
        # assign parameters (no grad tracking for the assignment)
        #print(theta_1d.shape)
        with torch.no_grad():

            vector_to_parameters(theta_1d, model.parameters())

        # forward on the dataset
        pred = model(X)
        data_loss = loss_func(pred, Y)
        reg = lambda2 * (theta_1d ** 2).sum()
        return -(data_loss +reg)/beta

    def objective_function(theta):
        """
        theta can be:
          - shape (d,)       -> returns scalar tensor
          - shape (N, d)     -> returns tensor of shape (N,)
        """
        theta = torch.as_tensor(theta, device=device, dtype=torch.float32)

        if theta.ndim == 1:
            # single parameter vector
            return loss_for_single_theta(theta)

        elif theta.ndim == 2:
            # batch of parameter vectors
            N, d = theta.shape
            #print(N)
            #print(d)
            losses = []
            for i in range(N):
                loss_i = loss_for_single_theta(theta[i])
                losses.append(loss_i)
            return torch.stack(losses)  # (N,)

        else:
            raise ValueError(f"theta must be 1D or 2D, got shape {theta.shape}")

    return objective_function




