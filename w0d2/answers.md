

> Question - what is the derivative of the quadratic loss for a single sample, wrt the weights?

L(x, a_0, a_1, a_2, b_1, b_2) = (y_pred - y)^2
L(x, w) = (y_pred(x,w) - y)^2
dLdw = 2(y_pred - y) * [
    1/2 ,
    cos(x),
    cos(2x),
    sin(x),
    sin(2x)
    ]
