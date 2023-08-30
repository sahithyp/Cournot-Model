import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from scipy.interpolate import interp1d
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

total_firms = 2
total_locations = 50

coefs = [total_firms, total_locations, [random.randint(150,650) for a in range(total_locations)],[random.uniform(1.0,5.0) for b in range(total_locations)],[random.uniform(0.5,2.0) for r in range(total_locations)],[random.uniform(0.001,0.05) for c in range(total_firms)]]
# coefs = [total_firms, total_locations, [10,20], [2,1], [4,5], [1,2]]          # coefs for 2x2
action = grad_ascent(learning_rate=0.01, tolerance=0.000001, max_iter=10000, action=np.ones(total_firms * total_locations), coefs=coefs)
print(action)

def demand_plot_v1():
    for i in range(total_firms):
        ts_perc_ai_dict = {}

        start_index = i * total_locations
        end_index = start_index + total_locations

        firm_i = action[start_index:end_index]  # firm i's output at all k locations
        ai = sum(firm_i)

        for k in range(total_locations):
            firms,locations,a,b,r,c = coefs
            surplus = (a[k]**2) / (2*b[k])
            perc_ai = firm_i[k] / ai

            ts_perc_ai_dict[surplus] = perc_ai

        # lists = sorted(ts_perc_ai_dict.items(), key=lambda kv: kv[1])       # sort by values
        lists = sorted(ts_perc_ai_dict.items())                             # sort by keys
        x,y = zip(*lists)
        # print(ts_perc_ai_dict)
        plt.scatter(x, y)
        plt.plot(x, y, label="Firm %d, Cost: %f" % (i, coefs[-1][i]))

    plt.xlabel("TS")
    plt.ylabel("% ai")
    plt.legend()
    plt.show()

def demand_plot_sorta():
    for i in range(total_firms):
        a_ts_perc_ai_dict = {}

        start_index = i * total_locations
        end_index = start_index + total_locations

        firm_i = action[start_index:end_index]  # firm i's output at all k locations
        ai = sum(firm_i)

        for k in range(total_locations):
            firms,locations,a,b,r,c = coefs
            # vals_lst = []
            # surplus = (a[k]**2) / (2*b[k])
            perc_ai = firm_i[k] / ai
            # vals_lst.append(surplus)
            # vals_lst.append(perc_ai)

            a_ts_perc_ai_dict[a[k]] = perc_ai

        # lists = sorted(ts_perc_ai_dict.items(), key=lambda kv: kv[1])       # sort by values
        lists = sorted(a_ts_perc_ai_dict.items())                             # sort by keys
        x,y = zip(*lists)
        plt.scatter(x, y)
        plt.plot(x, y, label="Firm %d, Cost: %f" % (i, coefs[-1][i]))

    plt.xlabel("a[k]")
    plt.ylabel("% ai")
    plt.title("a[k] vs % ai")
    plt.legend()
    plt.show()

def demand_plot_sortb():
    for i in range(total_firms):
        a_ts_perc_ai_dict = {}

        start_index = i * total_locations
        end_index = start_index + total_locations

        firm_i = action[start_index:end_index]  # firm i's output at all k locations
        ai = sum(firm_i)

        for k in range(total_locations):
            firms, locations, a, b, r, c = coefs
            # vals_lst = []
            # surplus = (a[k] ** 2) / (2 * b[k])
            perc_ai = firm_i[k] / ai
            # vals_lst.append(surplus)
            # vals_lst.append(perc_ai)
            a_ts_perc_ai_dict[b[k]] = perc_ai

        # lists = sorted(ts_perc_ai_dict.items(), key=lambda kv: kv[1])       # sort by values
        lists = sorted(a_ts_perc_ai_dict.items())  # sort by keys
        x, y = zip(*lists)
        plt.scatter(x, y)
        plt.plot(x, y, label="Firm %d, Cost: %f" % (i, coefs[-1][i]))

    plt.xlabel("b[k]")
    plt.ylabel("% ai")
    plt.title("b[k] vs % ai")
    plt.legend()
    plt.show()

demand_plot_v1()
demand_plot_sorta()
demand_plot_sortb()