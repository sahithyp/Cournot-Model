import examples

def main():
    game = examples.one_firm_two_locations()
    print(game.solve_grad_ascent(0.01, 0.00000001, 100000))

    game = examples.two_firms_two_locations()
    print(game.solve_grad_ascent(0.01, 0.00000001, 100000))

    game = examples.n_firms_m_locations(4,500)
    print(game.solve_grad_ascent(0.01, 0.00000001, 100000))


if __name__ == '__main__':
    main()

