import examples
import plot_avg_distribution

def main():
    # game = examples.one_firm_two_locations()
    # print(game.solve_grad_ascent(0.01, 0.00000001, 100000))
    #
    # game = examples.two_firms_two_locations()
    # print(game.solve_grad_ascent(0.01, 0.00000001, 100000))

    # game = examples.n_firms_m_locations(2,5)
    # game.solve_grad_ascent(0.01, 0.00000001, 100000)
    # costs = [0.001, 0.05]
    # print(game.demand_plot(costs))

    print(plot_avg_distribution.plot_avg_dist(3))


if __name__ == '__main__':
    main()

