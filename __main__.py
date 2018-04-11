import logging
import agent
import environment as env
import os
import sys
import graphics as grap
import datetime as dt
from math import ceil
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#np.set_printoptions(linewidth=300,edgeitems=16)

def main():
    """
    Get params
    Instantiate env
    Instantiate players
    Run auction
    """

    S = env.get_possible_states(config_dict['price_levels'],config_dict['num_players'])
    game_id = env.get_game_id()

    # Initialise the players
    player_list = []
    for p in range(config_dict['num_players']):
        new_player = agent.Player(
            player_id=p,
            alpha=config_dict['alpha'],
            gamma=config_dict['gamma'],
            epsilon=config_dict['epsilon'],
            epsilon_decay_1=config_dict['epsilon_decay_1'],
            epsilon_decay_2=config_dict['epsilon_decay_2'],
            epsilon_threshold=config_dict['epsilon_threshold'],
            agent_valuation=config_dict['agent_valuation'],
            S=S,
            q_convergence_threshold=config_dict['q_convergence_threshold'],
            print_directory=config_dict['output_folder'],
            q_update_mode=config_dict['q_update_mode'],
            file_name_base=str(game_id)
        )
        new_player.set_r(S,config_dict['bid_periods'])
        new_player.set_q()
        player_list = player_list + [new_player]

    i = 0
    while i<config_dict['episodes']+1:
        i = i + 1
        logging.info('Begin episode {0} of {1}'.format(i, config_dict['episodes'] - 1))
        s = env.get_initial_state(S, config_dict['initial_state_random'])

        for t in range(config_dict['bid_periods']):
            is_final_period = False if t < config_dict['bid_periods'] - 1 else True
            logging.info('Begin bidding period {0}, final period: {1}, state: {2}'.format(t, is_final_period, S[s]))
            for p in player_list[::-1]:
                a = p.select_action(t,s)
                p.write_path_log_entry(log_args=(i, t, s, a))
                p.update_q(t, s, a, is_final_period)
                p.update_epsilon()

                if is_final_period:
                    logging.debug('Player {} penultimate Q pane: \n {}'.format(p.player_id, p.Q[t-1]))
                    logging.debug('Player {} final Q pane: \n {}'.format(p.player_id, p.Q[t]))
                    logging.debug(
                        'Player {} payoff matrix for state {}: {}: \n {}'.format(p.player_id, s, p.S[s], p.get_current_qmatrix(t-1,s)))
                s = a

    logging.info('All episodes complete, printing path history and auction results for all agents...')

    path_dataframes = []
    results_dataframes = []
    for i,player in enumerate(player_list[::-1]):
        player.path_df = player.get_path_log_from_hdf(player.get_serialised_file_name()+'.hdf')
        path_dataframes.append(player.path_df)
        player.serialise_agent()

        fig,axs = grap.plot_rewards_per_episode(player.path_df)
        fig.suptitle('Rewards per Episode')
        fig.savefig(player.get_serialised_file_name() + '_rewards_per_episode.png')

        fig,axs = grap.path_graphics(player.path_df,alpha=0.03,sub_plots=5)
        fig.suptitle('Bids placed')
        fig.savefig(player.get_serialised_file_name()+'.png')
        #fig.show()

        results_df = env.get_results_summary(path_dataframes, 100)
        results_path = player.get_serialised_file_name()+'_results.csv'
        results_df.to_csv(results_path,index=False)
        results_dataframes.append(results_df)

        config_dict['path_df'] = player.path_df.to_dict()
        config_dict['results_df'] = results_df.to_dict()
        config_dict['player_id'] = player.player_id

        pickle_file = 'Player'+str(p.player_id)+'_'+config_dict['output_file']+'.pck'
        pickle_path = os.path.join(config_dict['output_folder'],pickle_file)
        with open(pickle_path, 'wb') as fp:
            pickle.dump(config_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    final_result_df = pd.DataFrame()
    for result in results_dataframes:
        final_result_df = pd.concat([final_result_df,result],axis=1)
    final_result_path = os.path.join(config_dict['output_folder'],str(game_id)+'_results.csv')
    final_result_df.to_csv(final_result_path)

    return

if __name__ == '__main__':
    logging.basicConfig(filename='bidding.log'.format(dt.datetime.strftime(dt.datetime.now(), '%Y%m%d-%H%M%S')),
                        format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logging.info('Process start')

    config_dict = env.interpret_args(sys.argv)
    if len(config_dict) == 0:
        config_dict = {
            # Auction parameters
            'episodes': 1000,
            'initial_state_random': False,

            # Environment parameters
            'bid_periods': 4,
            'price_levels': 5,
            'num_players': 1,
            'q_convergence_threshold':100,

            # Script run parameters
            'output_folder':r'./results',
            'output_file':'results.bat',

            # Player parameters
            'alpha': 0.8,
            'gamma': 0.5,
            'epsilon': 1,
            'epsilon_decay_1': 0.9999,
            'epsilon_decay_2': 0.99,
            'epsilon_threshold': 0.3,
            'agent_valuation': 4.1,
            'q_update_mode':'qlearn'
        }

    main()
    logging.info('Process end')