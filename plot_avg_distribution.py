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
    costs = [0.0001]
    total_distr = {}

    for i in range(len(costs)):
        total_distr[i] = []

    for i in range(iterations):
        game = initialize_game(costs, len(costs), 5)
        game.solve_grad_ascent(0.01, 0.00000001, 100000)
        sing_distr = game.demand_plot(costs)
        sing_dict = {}
        for pair in sing_distr[0]:
            sing_dict[pair[0]] = pair[1]
        lists = sorted(sing_dict.items())
        x, y = zip(*lists)
        plt.scatter(x, y)
        plt.plot(x, y, label="iter: %d" % i)

        for c in range(len(costs)):
            for firm in sing_distr:
                if firm == c:
                    total_distr[c].extend(sing_distr[firm])

    for i in range(len(costs)):
        tot_surp_perc_ai = {}
        for firm_key in total_distr:
            if firm_key == i:
                for pair in total_distr[firm_key]:
                    tot_surp_perc_ai[pair[0]] = pair[1]

        lists = sorted(tot_surp_perc_ai.items())
        x,y = zip(*lists)
        plt.scatter(x,y)
        plt.plot(x,y,label="Firm %d, Cost: %f" % (i,costs[i]))

    plt.xlabel("TS")
    plt.ylabel("% ai")
    plt.title("total surplus vs percent ai graph")
    plt.legend()
    plt.show()

    return total_distr




