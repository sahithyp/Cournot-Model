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
# list_of_func = []

def grad_func1_x(variables, coefs):
    x1, x2, y1, y2 = variables
    ax, bx, ay, by, rx, ry, c1, c2 = coefs
    df = ax - 2 * (bx + rx + c1) * x1 - (bx + rx) * x2 - 2*c1*y1
    # list_of_func.append(grad_func1_x(variables, coefs))
    return df

def grad_func1_y(variables, coefs):
    x1, x2, y1, y2 = variables
    ax, bx, ay, by, rx, ry, c1, c2 = coefs
    df = ay - 2 * (by + ry + c1) * y1 - (by + ry) * y2 - 2*c1*x1
    # list_of_func.append(grad_func1_y(variables, coefs))
    return df

def grad_func2_x(variables, coefs):
    x1, x2, y1, y2 = variables
    ax, bx, ay, by, rx, ry, c1, c2 = coefs
    df = ax - 2 * (bx + rx + c2) * x2 - (bx + rx) * x1 - 2*c2*y2
    # list_of_func.append(grad_func2_x(variables, coefs))
    return df

def grad_func2_y(variables, coefs):
    x1, x2, y1, y2 = variables
    ax, bx, ay, by, rx, ry, c1, c2 = coefs
    df = ay - 2 * (by + ry + c2) * y2 - (by + ry) * y1 - 2*c2*x2
    # list_of_func.append(grad_func2_y(variables, coefs))
    return df

def error(variables, coefs):
    e = 0
    # for i in list_of_func:
    #     e += i ** 2
    e += grad_func1_x(variables, coefs) ** 2 + grad_func1_y(variables, coefs) ** 2 + grad_func2_x(variables, coefs) ** 2 + grad_func2_y(variables, coefs) ** 2
    return e


# coefs = [a,b,r,c1, c2]
coefs = np.array([10, 2, 20, 1, 4, 5, 1, 2])

initial = np.array([0, 0, 0, 0])

result = opt.minimize(error, initial, args=coefs)

if result.success:
    variables = result.x
    print("Firm 1: x1 = ", variables[0], "\ty1 = ", variables[2], "\nFirm 2: x2 = ", variables[1], "\ty2 = ", variables[-1])
    print(error(result.x, coefs))

else:
    raise ValueError(result.message)
