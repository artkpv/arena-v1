# Artyom K, www.artkpv.net

import torch as t
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from typing import Optional, Callable
import ipywidgets as wg
from fancy_einsum import einsum
import streamlit as st
import math 

import utils

NUM_FREQUENCIES = 2
TARGET_FUNC = lambda x: 1 * (x > 1)
TOTAL_STEPS = 4000
LEARNING_RATE = 1e-6
EXAMPLES = 2000

x = t.linspace(-math.pi, math.pi, EXAMPLES)
y = TARGET_FUNC(x)
x_cos = t.stack([(n*x).cos() for n in range(1, NUM_FREQUENCIES+1)])
x_sin = t.stack([(n*x).sin() for n in range(1, NUM_FREQUENCIES+1)])

a_0 = t.randn(1, requires_grad=True)
A_n = t.randn(NUM_FREQUENCIES, requires_grad=True)
B_n = t.randn(NUM_FREQUENCIES, requires_grad=True)

y_pred_list = []
coeffs_list = []

for step in range(TOTAL_STEPS):

    # TODO: compute `y_pred` using your coeffs, and the terms `x_cos`, `x_sin`
    y_pred = a_0 / 2 + t.einsum('i,jk->k', A_n, x_cos) + t.einsum('i,jk->k', B_n, x_sin)

    # TODO: compute `loss`, which is the sum of squared error between `y` and `y_pred`
    loss = ((y - y_pred)**2).sum()
    loss.backward()
    
    if step % 100 == 0:
        print(f"{loss = :.2f}")
        coeffs_list.append([a_0.detach().item(), A_n.detach().numpy(), B_n.detach().numpy()])
        y_pred_list.append(y_pred.detach().numpy())

    # TODO update weights using gradient descent (using the parameter `LEARNING_RATE`)
    with t.no_grad():
        a_0 = a_0 - LEARNING_RATE * a_0.grad
        A_n = A_n - LEARNING_RATE * A_n.grad
        B_n = B_n - LEARNING_RATE * B_n.grad
    a_0.grad = None
    A_n.grad = None
    B_n.grad = None
    a_0.requires_grad = True
    A_n.requires_grad = True
    B_n.requires_grad = True

print('Before visualise_fourier_coeff_convergence')
fig = utils.visualise_fourier_coeff_convergence(x.numpy(), y.numpy(), y_pred_list, coeffs_list)
st.plotly_chart(fig)

