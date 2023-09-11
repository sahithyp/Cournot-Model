import numpy as np
import matplotlib.pyplot as plt
import random
import cournot_game
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
    locations = [(random.randint(50,650), random.uniform(1,5), random.uniform(.5,2)) for k in range(m)]

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
    number_locations = 3

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

            total_distr[i].append({("iter %d" % iter): sing_distr[i]})

        # for i in range(number_firms):
        #     for firm in sing_distr:
        #         if firm == i:
        #             total_distr[i].extend(sing_distr[firm])
    print(total_distr)

    for i in total_distr:
        surp_range = max_surp - min_surp
        increment = surp_range / (number_locations * 2)
        first = min_surp + increment
        increments = [first]
        for num in range(1,number_locations*2):
            increments.append(increments[num-1] + increment)

        for increment in increments:
            surp_near_increment = []
            for iter in iterations:
                surp_vals = iter
                surp = min(iter, key=lambda x:abs(x-increment))

        tot_surp_perc_ai = {}
        for firm_key in total_distr:
            if firm_key == i:
                for pair in total_distr[firm_key]:
                    tot_surp_perc_ai[pair[0]] = pair[1]
        # print("surp", tot_surp_perc_ai)
        lists = sorted(tot_surp_perc_ai.items())
        x,y = zip(*lists)
        plt.scatter(x,y)
        plt.plot(x,y,label="Firm %d, Cost: %f" % (i,costs[i]))

    plt.xlabel("TS")
    plt.ylabel("% ai")
    plt.title("total surplus vs percent ai graph")
    plt.legend()
    plt.show()

    # return total_distr


'''
from iter1: surp to perc_ai
from iter2: surp to perc_ai
from iter3: surp to perc_ai

create increments: divide range(surp) by 2 x number locations
iterate through each list:
    get perc_ai value linked to surplus closest to each increment
    add to new dictionary to plot final graph
'''
