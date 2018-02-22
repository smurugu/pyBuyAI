import logging
import agents
import auction as env
import datetime as dt


def main():
    """
    Get params
    Instantiate env
    Instantiate players
    Run auction
    """
    # Game parameters
    trials = 10000

    # Environment parameters
    bid_periods = 5
    price_levels = 10
    num_players = 1

    # Player parameters
    alpha = 0.8
    gamma = 0.5
    epsilon = 0.9
    epsilon_decay_1 = 0.999
    epsilon_decay_2 = 0.99
    epsilon_threshold = 0.6
    agent_valuation = price_levels * 0.7

    S = env.get_possible_states(price_levels,num_players)

    # Instantiate the players
    player_list = []
    for p in range(num_players):
        player_list[p] = agents.Player(p, alpha, gamma, epsilon, epsilon_decay_1, epsilon_decay_2, epsilon_threshold, agent_valuation)
        player_list[p].generate_r()
        player_list[p].generate_q()

    for t in range(trials):
        for period in range(bid_periods):
            for p in player_list:
                p.select_action()

            for p in player_list:
                p.update_Q()

        for p in player_list:
            p.update_epsilon()



if __name__ == '__main__':
    logging.basicConfig(filename='bidding_{0}.log'.format(dt.datetime.strftime(dt.datetime.now(), '%Y%m%d-%H%M%S')),
                        format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logging.info('Process start')
    main()

    print('Process end')