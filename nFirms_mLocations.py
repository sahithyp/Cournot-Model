import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import random

# a12 = firm 1 at location 2 = a[0][1]
# action = [[1,2],[1,2]]
# coefs = [n rows, m columns, list(a) for all locations, list(b) for all locations, list(r) for all locations, list(c) for all firms

# grad function
def grad_func(i, k, action, coefs):
    total_firms, total_locations, a, b, r, c = coefs

    aik = action[i * total_locations + k]
    ai = sum(action[i * total_locations + k] for k in range(total_locations))            # output of firm i at all locations
    ak = sum(action[i * total_locations + k] for i in range(total_firms))       # output at all n firms at location k

    df = a[k] - (b[k] + r[k]) * (aik + ak) - 2 * c[i] * ai
    return df

def error(action, coefs):
    total_firms = coefs[0]
    total_locations = coefs[1]
    e = 0

    for i in range(total_firms):
        for k in range(total_locations):
            e += grad_func(i, k, action, coefs) ** 2

    return e


total_firms = 5             # firms
total_locations = 5         # locations

# coefs = [[a], [b], [r], [c]]
# coefs = [total_firms, total_locations, [10,20], [2,1], [4,5], [1,2]]          # coefs for 2x2
# coefs = [total_firms, total_locations, [10], [2], [4], [3, 4]]                   # coefs for 2x1
# coefs = [total_firms, total_locations, [10,20], [2,1], [4,2], [3]]            # coefs for 1x2
# coefs = [total_firms, total_locations, [60],[2], [4], [1]]                    # coefs for 1x1

coefs = [total_firms, total_locations, [random.randint(50,150) for a in range(total_locations)],[random.randint(1,7) for b in range(total_locations)],[random.randint(5,10) for r in range(total_locations)],[random.randint(3,8) for c in range(total_firms)]]

initial = np.ones(total_firms * total_locations)

bnds = ((0, None), (0, None), (0,None), (0,None))

result = opt.minimize(error, initial, args=coefs)

if result.success:
    final_error = error(result.x, coefs)
    values = list(result.x)
    dict_of_vals = {}

    for i in range(total_firms):
        for k in range(total_locations):
            dict_of_vals["firm %d, location %d" % (i+1, k+1)] = values[i * total_locations + k]

    print(dict_of_vals)
    print("\nerror: ", final_error)

else:
    print("fail")
    print(result.x)
    raise ValueError(result.message)
