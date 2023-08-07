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

coefs = [total_firms, total_locations, [random.randint(150,650) for a in range(total_locations)],[random.randint(1,7) for b in range(total_locations)],[random.randint(5,10) for r in range(total_locations)],[random.randint(3,8) for c in range(total_firms)]]
# coefs = [4, 5, [257, 252, 331, 263, 256], [1, 2, 7, 5, 5], [10, 5, 7, 9, 7], [8, 3, 6, 5]]
action = grad_ascent(learning_rate=0.01, tolerance=0.000001, max_iter=10000, action=np.ones(total_firms * total_locations), coefs=coefs)

def demand_plot_v1():
    for i in range(total_firms):
        start_index = i*total_locations
        end_index = start_index + total_locations

        output = action[start_index:end_index]      # x-direction
        output.sort()

        tot_prod = sum(output)
        perc_prod = [output[i]/tot_prod for i in range(total_locations)]        # y-direction

        plt.scatter(output, perc_prod)
        plt.plot(output, perc_prod, label="Firm %d" % i)

        print(output)
        print(perc_prod)
        print()

    plt.xlabel("Total Surplus")
    plt.ylabel("% of firm production")
    plt.legend()
    plt.show()

def demand_plot_v2():
    ts = []
    locations = []
    for k in range(total_locations):
        output_vals = action[k::total_locations]
        output = sum(output_vals)
        ts.append(output)
        locations.append(k)

    for i in range(total_firms):
        perc_ts = []

        start_index = i * total_locations
        end_index = start_index + total_locations

        firm_i = action[start_index:end_index]  # x-direction
        for k in range(total_locations):
            perc_ts.append((firm_i[k]/ts[k]))

        plt.scatter(locations, perc_ts)
        plt.plot(locations, perc_ts, label="Firm %d" % i)

        print(perc_ts)

    print(ts)
    print()

    plt.xlabel("Location ID")
    plt.ylabel("% of output at location")
    plt.legend()
    plt.show()

def demand_plot_v3():
    ts = []
    locations = []
    for k in range(total_locations):
        output_vals = action[k::total_locations]
        output = sum(output_vals)
        ts.append(output)
        locations.append(k)

    for i in range(total_firms):
        perc_ts = []
        perc_firm_prod = []

        start_index = i * total_locations
        end_index = start_index + total_locations

        firm_i = action[start_index:end_index]  # x-direction
        tot_firm_i = sum(firm_i)

        for k in range(total_locations):
            perc_firm_prod_k = firm_i[k] / tot_firm_i
            perc_firm_prod.append(perc_firm_prod_k)

            perc_ts_k = firm_i[k] / ts[k]
            perc_ts.append(perc_ts_k)

            plt.annotate("location: %d" % k, (perc_firm_prod_k, perc_ts_k))

        # fig, ax = plt.subplots()
        plt.scatter(x=perc_firm_prod, y=perc_ts)
        plt.plot(perc_firm_prod, perc_ts, label="Firm %d" % i)

        # mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(lst_of_locations[sel.index]))
        print("\nfirm %d" % i)
        print("% of firm production", perc_firm_prod)
        print("% of total surplus", perc_ts)

    plt.xlabel("% of firm production")
    plt.ylabel("% of total surplus")
    plt.legend()
    plt.show()

def demand_plot_v4():
    ts = []

    for k in range(total_locations):
        output_vals = action[k::total_locations]
        output = sum(output_vals)
        ts.append(output)

    for i in range(total_firms):
        perc_ts = []
        ts_by_firm = {}

        start_index = i * total_locations
        end_index = start_index + total_locations

        firm_i = action[start_index:end_index]  # x-direction
        for k in range(total_locations):
            perc_ts_k = firm_i[k] / ts[k]
            # perc_ts.append(perc_ts_k)
            # plt.annotate("location: %d" % k, (ts[k], perc_ts_k))
            ts_by_firm[ts[k]] = perc_ts_k

        dict_keys = list(ts_by_firm.keys())
        dict_keys.sort()
        sorted_ts_by_firm = {i:ts_by_firm[i] for i in dict_keys}

        lists = sorted(sorted_ts_by_firm.items())
        x,y = zip(*lists)

        plt.scatter(x, y)
        plt.plot(x, y, label="Firm %d" % i)

        print("\nfirm %d" % i)
        print(perc_ts)

    print(ts)
    print()

    plt.xlabel("Total Surplus")
    plt.ylabel("% of output at location")
    plt.legend()
    plt.show()


demand_plot_v4()