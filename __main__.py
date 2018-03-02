import logging
import agent
import environment as env
import sys
import graphics as grap

def main():
    """
    Get params
    Instantiate env
    Instantiate players
    Run auction
    """

    S = env.get_possible_states(config_dict['price_levels'],config_dict['num_players'])

    # Initialise the players
    player_list = []
    for p in range(config_dict['num_players']):
        new_player = agent.Player(
            p,
            config_dict['alpha'],
            config_dict['gamma'],
            config_dict['epsilon'],
            config_dict['epsilon_decay_1'],
            config_dict['epsilon_decay_2'],
            config_dict['epsilon_threshold'],
            config_dict['agent_valuation'][p],
            S,
            config_dict['output_folder']
        )
        new_player.set_r(S,config_dict['bid_periods'])
        new_player.set_q()
        player_list = player_list + [new_player]

    for i in range(config_dict['episodes']):
        logging.info('Begin episode {0} of {1}'.format(i, config_dict['episodes'] - 1))
        s = env.get_initial_state(S, config_dict['initial_state_random'])
        for t in range(config_dict['bid_periods']):
            is_final_period = False if t < config_dict['bid_periods'] - 1 else True
            logging.info('Begin bidding period {0}, final period: {1}, state: {2}'.format(t, is_final_period, S[s]))
            for p in player_list:
                a = p.select_action(t,s)
                p.write_path_log_entry(log_args=(i, t, s, a))
                p.update_q(t, s, a, is_final_period)
                s = a

    logging.info('All episodes complete, printing path history and auction results for all agents...')

    path_dataframes = []
    for i,player in enumerate(player_list):
        player.path_df = player.get_path_log_from_hdf(player.get_serialised_file_name()+'.hdf')
        path_dataframes.append(player.path_df)
        player.serialise_agent()

        fig,axs = grap.path_graphics(player.path_df,alpha=0.03,sub_plots=5)
        fig.savefig(player.get_serialised_file_name()+'.png')
        fig.show()

        #print results per agent: temporary
        results_df = env.get_results_summary(path_dataframes, 1000)
        results_path = player.get_serialised_file_name()+'.csv'
        results_df.to_csv(results_path,index=False)

    return

if __name__ == '__main__':
    logging.basicConfig(filename='bidding.log'.format(dt.datetime.strftime(dt.datetime.now(), '%Y%m%d-%H%M%S')),
                        format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logging.info('Process start')

    config_dict = env.interpret_args(sys.argv)
    if len(config_dict) == 0:
        config_dict = {
            # Auction parameters
            'episodes': 10,
            'initial_state_random': False,

            # Environment parameters
            'bid_periods': 1,
            'price_levels': 10,
            'num_players': 1,

            # Script run parameters
            'output_folder':r'./results',

            # Player parameters
            'alpha': 0.8,
            'gamma': 0.5,
            'epsilon': 1,
            'epsilon_decay_1': 0.9999,
            'epsilon_decay_2': 0.99,
            'epsilon_threshold': 0.4,
            'agent_valuation': [7]
        }

    main()
    logging.info('Process end')