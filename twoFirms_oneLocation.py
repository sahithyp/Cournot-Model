import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from mpl_toolkits.mplot3d import axes3d
import scipy.optimize as opt


# price function
def price_func(a, b, Ak, Ai):
    return (a - b * Ak) * Ai


# rent cost function
def cost_func(r, Ak, Ai):
    return (r * Ak) * Ai


# production cost function
def prod_cost_func(c, Ai):
    return c * Ai ** 2


# utility function

# def fun1(x1, x2):
#     f1 = price_func(10, 2, (x1+x2))*x1 - cost_func(1, (x1+x2))*x1 - prod_cost_func(3, x1)
#     return f1
#
# def fun2(x1, x2):
#     f2 = price_func(10, 2, (x1+x2))*x2 - cost_func(1, (x1+x2))*x2 - prod_cost_func(4, x2)
#     return f2
# f = price_func(coefs[0], coefs[1], (x1 + x2), x1) - cost_func(coefs[2], (x1 + x2), x1) - prod_cost_func(coefs[-1], x1)

def grad_func1(variables, coefs):
    x1, x2 = variables
    a, b, r, c1, c2 = coefs
    df = a - 2 * (b + r + c1) * x1 - (b + r) * x2
    return df


def grad_func2(variables, coefs):
    x1, x2 = variables
    a, b, r, c1, c2 = coefs
    df = a - 2 * (b + r + c2) * x2 - (b + r) * x1
    return df


def error(variables, coefs):
    e = grad_func1(variables, coefs) ** 2 + grad_func2(variables, coefs) ** 2
    return e


# coefs = [a,b,r,c1, c2]
coefs = np.array([10, 2, 4, 3, 4])

initial = np.array([0, 0])

result = opt.minimize(error, initial, args=coefs)

if result.success:
    print(result.x)
    print(error(result.x, coefs))

else:
    raise ValueError(result.message)
