import examples

def main():
    # game = examples.one_firm_two_locations()
    # print(game.solve_grad_ascent(0.01, 0.00000001, 100000))
    #
    # game = examples.two_firms_two_locations()
    # print(game.solve_grad_ascent(0.01, 0.00000001, 100000))

    game = examples.n_firms_m_locations(2,5)
    game.solve_grad_ascent(0.01, 0.00000001, 100000)
    game.demand_plot()


if __name__ == '__main__':
    main()

