import matplotlib.pyplot as plt
import numpy as np
import random

class Game():
    """Cournot Game with solver. Locations must be added first, then firms"""
    def __init__(self):
        self.number_firms = 0
        self.number_locations = 0
        self.location_params = []
        self.firm_grads = []
        self.actions = []
        # self.costs = []

    def add_location(self,params):
        self.number_locations += 1
        self.location_params.append(params)

    def add_firm(self, grad_func, action):
        self.number_firms += 1
        self.firm_grads.append(grad_func[0])
        # self.costs.append(grad_func[-1])
        self.actions.extend(action)

    def solve_grad_ascent(self, learning_rate, tolerance, max_iter):
        max_error = np.infty
        indices = [(i, k) for i in range(self.number_firms) for k in range(self.number_locations)]
        iter = 0
        while max_error > tolerance and iter < max_iter:
            max_error = 0
            random.shuffle(indices)
            for (i, k) in indices:
                grad = self.firm_grads[i](self, i, k)
                aik = self.actions[i * self.number_locations + k]

                aik = max(aik + learning_rate * grad, 0)
                eik = grad ** 2 * aik

                if eik > max_error:
                    max_error = eik

                self.actions[i * self.number_locations + k] = aik

            iter += 1

        return self.actions, max_error

    def demand_plot(self,costs):
        total_distr = {}
        for i in range(self.number_firms):
            ts_perc_ai_dict = {}
            prices = []
            start_index = i * self.number_locations
            end_index = start_index + self.number_locations
            firm_i = self.actions[start_index:end_index]
            ai = sum(firm_i)

            for k in range(self.number_locations):
                a,b,r = self.location_params[k]
                prices.append(a)
                surplus = (a**2) / (2 * b)
                perc_ai = firm_i[k] / ai
                ts_perc_ai_dict[surplus] = perc_ai

            lists = sorted(ts_perc_ai_dict.items())
            # x, y = zip(*lists)
            total_distr[i] = lists
        return total_distr
        #     plt.scatter(x,y)
        #     plt.plot(x, y, label="Firm %d, Cost: %f" % (i, costs[i]))      # add cost
        #
        # plt.xlabel("TS")
        # plt.ylabel("% ai")
        # plt.title("total surplus vs percent ai graph")
        # plt.legend()
        # plt.show()