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
x_cos_sin = t.vstack(
    (t.stack([(n*x).cos() for n in range(1, NUM_FREQUENCIES+1)]),
     t.stack([(n*x).sin() for n in range(1, NUM_FREQUENCIES+1)]))).T

print('x_cos_sin.shape', x_cos_sin.shape)

model = t.nn.Sequential(
    t.nn.Linear(NUM_FREQUENCIES * 2, 1),
    t.nn.Flatten(0,1)
)

y_pred_list = []
coeffs_list = []

for step in range(TOTAL_STEPS):

    # Compute `y_pred` using your coeffs, and the terms `x_cos`, `x_sin`
    y_pred = model(x_cos_sin)

    # Compute `loss`, which is the sum of squared error between `y` and `y_pred`
    loss = ((y - y_pred)**2).sum()
    loss.backward()

    if step % 100 == 0:
        print(f"{loss = :.2f}")
        A_n = list(model.parameters())[0].detach().numpy().squeeze()[:NUM_FREQUENCIES]
        B_n = list(model.parameters())[0].detach().numpy().squeeze()[NUM_FREQUENCIES:]
        a_0 = list(model.parameters())[1].item()
        y_pred_list.append(y_pred.cpu().detach().numpy())
        coeffs_list.append([a_0, A_n.copy(), B_n.copy()])

    # Update weights using gradient descent (using the parameter `LEARNING_RATE`)
    with t.inference_mode():
        for p in model.parameters():
            p -= LEARNING_RATE * p.grad
    model.zero_grad()

print('Before visualise_fourier_coeff_convergence')
fig = utils.visualise_fourier_coeff_convergence(x.numpy(), y.numpy(), y_pred_list, coeffs_list)
st.plotly_chart(fig)

