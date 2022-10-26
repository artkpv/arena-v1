import numpy as np
#import plotly.express as px
#import plotly.graph_objs as go
from typing import Optional, Callable
import ipywidgets as wg
#from fancy_einsum import einsum

import utils

def DFT_1d(arr: np.ndarray, inverse: bool = False) -> np.ndarray:
    """
    Returns the DFT of the array `arr`, with an optional `inverse` argument.
    """
    N = arr.shape[0]
    powers = np.ndarray([list(range(N))])
    powers = powers.T @ powers
    W = np.exp(-powers * 2 * np.pi / N)
    return W @ arr

utils.test_DFT_func(DFT_1d)
