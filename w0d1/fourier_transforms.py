# Artyom K, www.artkpv.net
# Run it with ` streamlit run w0d1/fourier_transforms.py `

import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from typing import Optional, Callable
import ipywidgets as wg
from fancy_einsum import einsum

import utils

def DFT_1d(arr: np.ndarray, inverse: bool = False) -> np.ndarray:
    """
    Returns the DFT of the array `arr`, with an optional `inverse` argument.
    """
    N = arr.shape[0]
    powers = np.array([list(range(N))])
    powers = powers.T @ powers
    W = np.exp(np.cdouble(complex(0, (1 if inverse else -1) * 2 * np.pi / N))) ** powers
    return W @ arr * (1/N if inverse else 1)

def mytest():
    x = np.array([
        complex(1),
        complex(2, -1),
        complex(0, -1),
        complex(-1, 2)
    ])
    res = DFT_1d(x)
    res_inv = DFT_1d(res, inverse=True)
    expected = np.array([
        complex(2),
        complex(-2, -2),
        complex(0, -2),
        complex(4, 4)
    ])
    np.testing.assert_allclose(res, expected)
    np.testing.assert_allclose(res_inv, x, atol=1e-10, err_msg="Inverse DFT failed")

utils.test_DFT_func(DFT_1d)
mytest()
print('DFT_1d - all tests pass')


def integrate_function(func: Callable, x0: float, x1: float, n_samples: int = 1000):
    """
    Calculates the approximation of the Riemann integral of the function `func`, 
    between the limits x0 and x1.

    You should use the Left Rectangular Approximation Method (LRAM).
    """
    if n_samples <= 1:
        return 0

    h = (x1 - x0) / (n_samples - 1)
    x = np.linspace(x0, x1, n_samples)
    return h * sum(func(e) for e in x)

utils.test_integrate_function(integrate_function)
print('PASS utils.test_integrate_function(integrate_function)')

def integrate_product(func1: Callable, func2: Callable, x0: float, x1: float, n_samples: int = 1000):
    """
    Computes the integral of the function x -> func1(x) * func2(x).
    """
    return integrate_function(lambda x: func1(x) * func2(x), x0, x1, n_samples)

utils.test_integrate_product(integrate_product)
print('PASS utils.test_integrate_product(integrate_product)')


def calculate_fourier_series(func: Callable, max_freq: int = 50):
    """
    Calculates the fourier coefficients of a function, 
    assumed periodic between [-pi, pi].

    Your function should return ((a_0, A_n, B_n), func_approx), where:
        a_0 is a float
        A_n, B_n are lists of floats, with n going up to `max_freq`
        func_approx is the fourier approximation, as described above
    """
    a_0 = integrate_function(func, -np.pi, np.pi) / np.pi
    A_n = [integrate_product(func, lambda x: np.cos(x*n), -np.pi, np.pi) / np.pi for n in range(1, max_freq+1)]
    B_n = [integrate_product(func, lambda x: np.sin(x*n), -np.pi, np.pi) / np.pi for n in range(1, max_freq+1)]
    def f(xarr):
        return [
            a_0/2 + sum(
                A_n[n-1] * np.cos(n * x) + B_n[n-1] * np.sin(n * x) for n in range(1, max_freq+1)
            ) for x in xarr]
    return ((a_0, A_n, B_n), f)
            

step_func = lambda x: 1 * (x > 0)
fig = utils.create_interactive_fourier_graph(calculate_fourier_series, func = step_func)
fig 

