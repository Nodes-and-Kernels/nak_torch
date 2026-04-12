import numpy as np
import os

def sigmoid(x):
    out = np.empty_like(x)
    x_neg, x_pos = x[x < 0], x[x >= 0]
    out[x < 0] = np.exp(x_neg) / (1 + np.exp(x_neg))
    out[x >= 0] = 1 / (1 + np.exp(-x_pos))
    return out

if __name__ == '__main__':
    np.random.seed(0)
    N_SAMPLES = 1000
    BETA_X, BETA_Y, BETA_0 = -3., -3., 2.
    x = np.random.randn(N_SAMPLES)
    y = np.random.randn(N_SAMPLES)
    logits = BETA_X * x + BETA_Y * y + BETA_0
    probs = sigmoid(logits)
    aux_z = np.random.rand(N_SAMPLES)
    label_pts = aux_z < probs
    data = np.column_stack((x, y, label_pts))
    fname = os.path.join(os.path.dirname(__file__), "simple_linear.npy")
    np.save(fname, data)