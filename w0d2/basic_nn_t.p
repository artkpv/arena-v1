# Artyom K, www.artkpv.net

import torch as t
import plotly.express as px
import plotly.graph_objs as go
from typing import Optional, Callable
import ipywidgets as wg
from fancy_einsum import einsum
from math import pi

import utils

NUM_FREQUENCIES = 2
TARGET_FUNC = lambda x: 1 * (x > 1)
TOTAL_STEPS = 4000
LEARNING_RATE = 1e-6

x = t.linspace(-pi, pi, 2000)
y = TARGET_FUNC(x)

x_cos = t.tensor([t.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)])
x_sin = t.tensor([t.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)])

a_0 = t.randn()
A_n = t.randn(NUM_FREQUENCIES)
B_n = t.randn(NUM_FREQUENCIES)

y_pred_list = []
coeffs_list = []

for step in range(TOTAL_STEPS):

    # TODO: compute `y_pred` using your coeffs, and the terms `x_cos`, `x_sin`
    y_pred = a_0 / 2 + t.einsum('i,j->', A_n, x_cos) + t.einsum('i,j->', B_n, x_sin)

    # TODO: compute `loss`, which is the sum of squared error between `y` and `y_pred`
    loss = (y - y_pred)**2

    if step % 100 == 0:
        print(f"{loss = :.2f}")
        coeffs_list.append([a_0, A_n.copy(), B_n.copy()])
        y_pred_list.append(y_pred)

    # TODO: compute gradients of coeffs with respect to `loss`
    dLda_0 = (y - y_pred)
    dLdA_n = 2 * (y - y_pred) * x_cos
    dLdB_n = 2 * (y - y_pred) * x_sin

    # TODO update weights using gradient descent (using the parameter `LEARNING_RATE`)
    a_0 -= LEARNING_RATE * dLda_0
    A_n -= LEARNING_RATE * dLdA_n
    B_n -= LEARNING_RATE * dLdB_n

utils.visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list)
