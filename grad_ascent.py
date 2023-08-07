import random
import numpy as np

from nFirms_mLocations import grad_func

def grad_ascent(learning_rate, tolerance, max_iter, action, coefs):
    total_firms, total_locations, a, b, r, c = coefs
    max_error = np.infty
    indices = [(i, k) for i in range(total_firms) for k in range(total_locations)]
    print(indices)
    iter = 0
    while max_error > tolerance and iter < max_iter:
        max_error = 0
        random.shuffle(indices)
        for (i,k) in indices:
            grad = grad_func(i,k, action, coefs)
            aik = action[i * total_locations + k]

            aik = max(aik + learning_rate * grad, 0)
            eik = grad**2 * aik

            if eik > max_error:
                max_error = eik

            action[i * total_locations + k] = aik

        iter += 1

    return action, max_error

total_firms = 4
total_locations = 5

coefs = [total_firms, total_locations, [random.randint(2500,3500) for a in range(total_locations)],[random.randint(1,7) for b in range(total_locations)],[random.randint(5,10) for r in range(total_locations)],[random.randint(3,8) for c in range(total_firms)]]
action = np.ones(total_firms * total_locations)

learning_rate = 0.01
tolerance = 0.000001
max_iter = 10000
max_error = 1


print(grad_ascent(learning_rate, tolerance, max_iter, action, coefs))
