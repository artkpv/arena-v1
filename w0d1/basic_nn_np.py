# Artyom K, www.artkpv.net

import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from typing import Optional, Callable
import ipywidgets as wg
from fancy_einsum import einsum
import streamlit as st

import utils

NUM_FREQUENCIES = 2
TARGET_FUNC = lambda x: 1 * (x > 1)
TOTAL_STEPS = 4000
LEARNING_RATE = 1e-6
EXAMPLES = 2000

x = np.linspace(-np.pi, np.pi, EXAMPLES)
y = TARGET_FUNC(x)

x_cos = np.array([np.cos(n*x) for n in range(1, NUM_FREQUENCIES+1)])
x_sin = np.array([np.sin(n*x) for n in range(1, NUM_FREQUENCIES+1)])

a_0 = np.random.randn()
A_n = np.random.randn(NUM_FREQUENCIES)
B_n = np.random.randn(NUM_FREQUENCIES)

y_pred_list = []
coeffs_list = []

for step in range(TOTAL_STEPS):

    # TODO: compute `y_pred` using your coeffs, and the terms `x_cos`, `x_sin`
    y_pred = a_0 / 2 + np.einsum('i,jk->k', A_n, x_cos) + np.einsum('i,jk->k', B_n, x_sin)

    # TODO: compute `loss`, which is the sum of squared error between `y` and `y_pred`
    loss = sum((y - y_pred)**2) # / EXAMPLES
    
    if step % 100 == 0:
        print(f"{loss = :.2f}")
        coeffs_list.append([a_0, A_n.copy(), B_n.copy()])
        y_pred_list.append(y_pred)

    # TODO: compute gradients of coeffs with respect to `loss`
    dLda_0 = sum(2*(y_pred - y))
    dLdA_n = (2 * (y_pred - y) * x_cos).sum(axis=1)
    dLdB_n = (2 * (y_pred - y) * x_sin).sum(axis=1)

    # TODO update weights using gradient descent (using the parameter `LEARNING_RATE`)
    a_0 -= LEARNING_RATE * dLda_0
    A_n -= LEARNING_RATE * dLdA_n
    B_n -= LEARNING_RATE * dLdB_n

print('Before visualise_fourier_coeff_convergence')
fig = utils.visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list)
st.plotly_chart(fig)
