import numpy as np
import matplotlib.pyplot as plt
import random
import cournot_game

def initialize_game(costs, n, m):
    def grad_factory(c):
        def grad(game, i, k):
            a, b, r = game.location_params[k]
            aik = game.actions[i * game.number_locations + k]
            ai = sum(game.actions[i * game.number_locations + k] for k in range(game.number_locations))  # output of firm i at all locations
            ak = sum(game.actions[i * game.number_locations + k] for i in range(game.number_firms))  # output at all n firms at location k
            df = a - (b + r) * (aik + ak) - 2 * c * ai
            return df
        return [grad]

    game = cournot_game.Game()
    locations = [(random.randint(50,650), random.uniform(1,5), random.uniform(.5,2)) for k in range(m)]

    for location in locations:
        game.add_location(location)
    for c in costs:
        game.add_firm(grad_factory(c), [1] * game.number_locations)

    return game

def plot_avg_dist(iterations):
    costs = [0.001, 0.05]
    tot_surp_perc_ai = {}


    for i in range(iterations):
        game = initialize_game(costs, len(costs), 5)
        game.solve_grad_ascent(0.01, 0.00000001, 100000)

        surplus, perc_ai = game.demand_plot(costs)

        for i in range(len(surplus)):
            tot_surp_perc_ai[surplus[i]] = perc_ai[i]

    lists = sorted(tot_surp_perc_ai.items())
    x, y = zip(*lists)
    plt.scatter(x,y)
    plt.plot(x, y, label="Firm %d, Cost: %f" % (i, costs[i]))      # add cost

    plt.xlabel("TS")
    plt.ylabel("% ai")
    plt.title("total surplus vs percent ai graph")
    plt.legend()
    plt.show()

    return(tot_surp)




