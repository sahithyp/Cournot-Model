import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from mpl_toolkits.mplot3d import axes3d


# price function
def price_func(alpha, beta, Ak):
    return (alpha - beta * Ak) * Ak


# rent cost function
def cost_func(gamma, Ak):
    return gamma * Ak


# production cost function
def prod_cost_func(rho, Ai):
    return rho * Ai ** 2


# utility function
x, y = symbols("variables y", real=True)


def fun(x, y):
    f = price_func(10, 2, x) + price_func(20, 1, y) - cost_func(1, x) - cost_func(2, y) - prod_cost_func(3, x + y)
    return f


# solve partial derivatives & z value
f_x = diff(fun(x, y), x)
f_y = diff(fun(x, y), y)

sol = solve((f_x, f_y), (x, y))
val_list = list(sol.values())

# check constraints, if values are positive
# f_z = 0
#
# for i in val_list:
#     if i < 0:
#         z1 = solve(f_y, (0,y))
#         z2 = solve(f_x, (variables, 0))
#         z3 = fun(0, 0)
#
#         z_list = (z1, z2, z3)
#         print(z_list)
#         z = max(z_list)
#         f_z = z
#         break
# else:
#     f_z = fun(val_list[0], val_list[1])

f_z = fun(val_list[0], val_list[1])

sol['z'] = f_z
val_list.append(f_z)

print("Optimal Quantity for Function: ", sol)

# plot function
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-5.0, 5.0, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array([fun(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z)

ax.scatter(val_list[0], val_list[1], val_list[2], color="red", s=100, alpha=1)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
