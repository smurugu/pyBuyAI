import logging
import agent
import environment as env
import datetime as dt
import sys
import ast

def interpret_args(sys_args):
    if len(sys_args) == 1:
        return {}

    arg_list = sys.argv[1].split(',')
    arg_dict = {x.split(':')[0]: x.split(':')[1] for x in arg_list}

    for k in arg_dict:
        try:
            arg_dict[k] = ast.literal_eval(arg_dict[k])
        except Exception as ex:
            print('interpret_args: unable to do literal eval for argument: {0} \n Error: {1}'.format(arg_dict[k], ex))

    return arg_dict

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
        path = []
        for t in range(config_dict['bid_periods']):
            is_final_period = False if t < config_dict['bid_periods'] - 1 else True
            logging.info('Begin bidding period {0}, final period: {1}, state: {2}'.format(t, is_final_period, S[s]))
            for p in player_list:
                a = p.select_action(t,s)
                path = path + [a]
                p.write_path_log_entry(log_args=(i, t, s, a))
                p.update_q(t, s, a, is_final_period)
                s = a

        p.update_epsilon()
        path = [(ac, S[ac]) for ac in path]
        logging.info('Auction complete, path taken: {0}'.format(path))
    logging.info('All episodes complete, printing path history for all agents...')

    for i,player in enumerate(player_list):
        player.path_df = player.get_path_log_from_hdf(player.get_serialised_file_name()+'.hdf')
        player.serialise_agent()
        fig,axs = player.get_path_graphics()
        fig.savefig(player.get_serialised_file_name()+'.png')
        fig.show()

    return

if __name__ == '__main__':
    logging.basicConfig(filename='bidding.log'.format(dt.datetime.strftime(dt.datetime.now(), '%Y%m%d-%H%M%S')),
                        format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logging.info('Process start')

    config_dict = interpret_args(sys.argv)
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