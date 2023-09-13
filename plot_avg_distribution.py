import numpy as np
import matplotlib.pyplot as plt
import random
import cournot_game
from datetime import datetime
from statistics import mean
import examples

def initialize_game(costs, n, m):
    def grad_factory(c):
        def grad(game, i, k):
            a, b, r = game.location_params[k]
            aik = game.actions[i * game.number_locations + k]
            ai = sum(game.actions[i * game.number_locations + k] for k in range(game.number_locations))  # output of firm i at all locations
            ak = sum(game.actions[i * game.number_locations + k] for i in range(game.number_firms))  # output at all n firms at location k
            df = a - (b + r) * (aik + ak) - 2 * c * ai
            return df
        return grad

    game = cournot_game.Game()
    locations = [(random.randint(50,650), random.uniform(1,5), 1) for k in range(m)]

    for location in locations:
        game.add_location(location)
    for c in costs:
        game.add_firm(grad_factory(c), [1] * game.number_locations)

    return game

def plot_avg_dist(iterations):
    costs = [0.0001, 0.05]
    number_firms = len(costs)
    total_distr = {}
    max_surp = 0
    min_surp = np.infty
    number_locations = 10

    for i in range(number_firms):
        total_distr[i] = []

    for iter in range(iterations):
        game = initialize_game(costs, len(costs), number_locations)
        game.solve_grad_ascent(0.01, 0.00000001, 100000)
        sing_distr = game.demand_plot(costs)
        # sing_dict = {}
        # print(sing_distr)
        # for pair in sing_distr[0]:
        #     sing_dict[pair[0]] = pair[1]
        # lists = sorted(sing_dict.items())
        # x, y = zip(*lists)
        # plt.scatter(x, y)
        # plt.plot(x, y, label="iter: %d" % iter)

        for i in range(number_firms):
            max_in_sing_distr = max(sing_distr[i])[0]
            min_in_sing_distr = min(sing_distr[i])[0]
            if max_in_sing_distr > max_surp:
                max_surp = max_in_sing_distr
            if min_in_sing_distr < min_surp:
                min_surp = min_in_sing_distr

            total_distr[i].extend(sing_distr[i])
            # total_distr[i].append({("iter %d" % iter): sing_distr[i]})

        # for i in range(number_firms):
        #     for firm in sing_distr:
        #         if firm == i:
        #             total_distr[i].extend(sing_distr[firm])
    print(total_distr)

    tot_ai = {}
    for i in total_distr:
        surp_range = max_surp - min_surp
        number_bins = int(number_locations / 2)
        increment = surp_range / (number_bins)
        increments = [min_surp]
        tot_ai[min_surp] = []
        print("max val", max_surp)
        for num in range(1,number_bins+1):
            increments.append(increments[num-1] + increment)
            tot_ai[increments[num]] = []

        for p in increments:
            for (surplus, ai) in total_distr[i]:
                if surplus >= p and surplus < (p + increment):
                    tot_ai[p].append(ai)
        print("total perc ai", tot_ai)
        tot_perc_ai = {}
        for (surplus,lst_ai) in tot_ai.items():
            if len(lst_ai) > 0:
                tot_perc_ai[surplus] = mean(lst_ai)

        lists = sorted(tot_perc_ai.items())
        x,y = zip(*lists)
        plt.scatter(x,y)
        plt.plot(x,y,label="Firm %d, Cost: %f" % (i,costs[i]))

    plt.xlabel("TS")
    plt.ylabel("% ai")
    plt.title("total surplus vs percent ai graph")
    plt.legend()
    time = datetime.now().strftime("%H:%M:%S")
    plt.savefig(f"plots/avg_distribution_{time}")
    plt.show()

    # return total_distr