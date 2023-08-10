import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import random

# from grad_ascent import grad_ascent
def grad_func(i, k, action, coefs):
    total_firms, total_locations, a, b, r, c = coefs

    aik = action[i * total_locations + k]
    ai = sum(action[i * total_locations + k] for k in range(total_locations))            # output of firm i at all locations
    ak = sum(action[i * total_locations + k] for i in range(total_firms))       # output at all n firms at location k

    df = a[k] - (b[k] + r[k]) * (aik + ak) - 2 * c[i] * ai
    return df

def grad_ascent(learning_rate, tolerance, max_iter, action, coefs):
    total_firms, total_locations, a, b, r, c = coefs
    max_error = np.infty
    indices = [(i, k) for i in range(total_firms) for k in range(total_locations)]
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

    print("error: ", max_error)
    return action

total_firms = 4
total_locations = 5

# coefs = [total_firms, total_locations, [random.randint(150,650) for a in range(total_locations)],[random.randint(1,7) for b in range(total_locations)],[random.randint(5,10) for r in range(total_locations)],[random.randint(3,8) for c in range(total_firms)]]
coefs = [4, 5, [257, 252, 331, 263, 256], [1, 2, 7, 5, 5], [10, 5, 7, 9, 7], [8, 3, 6, 5]]
action = grad_ascent(learning_rate=0.01, tolerance=0.000001, max_iter=10000, action=np.ones(total_firms * total_locations), coefs=coefs)

def demand_plot():
    for i in range(total_firms):
        perc_ai_ts_dict = {}

        start_index = i * total_locations
        end_index = start_index + total_locations

        firm_i = action[start_index:end_index]  # firm i's output at all k locations
        ai = sum(firm_i)

        for k in range(total_locations):
            firms,locations,a,b,r,c = coefs
            surplus = (a[k]**2) / (2*b[k])
            perc_ai = firm_i[k] / ai

            perc_ai_ts_dict[surplus] = perc_ai

        dict_keys = list(perc_ai_ts_dict.keys())
        dict_keys.sort()
        sorted_perc_ai_ts_dict = {i: perc_ai_ts_dict[i] for i in dict_keys}

        lists = sorted(sorted_perc_ai_ts_dict.items())
        x,y = zip(*lists)

        plt.scatter(x, y)
        plt.plot(x, y, label="Firm %d" % i)

    plt.xlabel("Total Surplus")
    plt.ylabel("% ai")
    plt.legend()
    plt.show()

demand_plot()