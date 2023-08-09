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

    def add_location(self,params):
        self.number_locations += 1
        self.location_params.append(params)

    def add_firm(self,grad_func, action):
        self.number_firms += 1
        self.firm_grads.append(grad_func)
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