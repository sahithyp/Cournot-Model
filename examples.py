import random

import cournot_game
def one_firm_two_locations():
    game = cournot_game.Game()

    game.add_location((10,2,4))
    game.add_location((20,1,2))

    def grad(game, i, k):
        a, b, r = game.location_params[k]

        aik = game.actions[i * game.number_locations + k]
        ai = sum(game.actions[i * game.number_locations + k] for k in range(game.number_locations))  # output of firm i at all locations
        ak = sum(game.actions[i * game.number_locations + k] for i in range(game.number_firms))  # output at all n firms at location k

        df = a - (b + r) * (aik + ak) - 2 * 3 * ai
        return df

    game.add_firm(grad, [1]*game.number_locations)

    return game

def two_firms_two_locations():
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
    locations = [(10, 2, 4), (20, 1, 5)]
    costs = [1,2]

    for location in locations:
        game.add_location(location)

    for c in costs:
        game.add_firm(grad_factory(c), [1] * game.number_locations)

    return game

def n_firms_m_locations(n,m):
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
    locations = [(random.randint(50,150), random.randint(1,7), random.randint(5,10)) for k in range(m)]
    costs = [(random.randint(1,10)) for i in range(n)]

    for location in locations:
        game.add_location(location)

    for c in costs:
        game.add_firm(grad_factory(c), [1] * game.number_locations)

    return game