import logging
import agent as agent
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
import random

np.set_printoptions(linewidth=300,edgeitems=20)

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
            share_rewards_on_tie=config_dict['share_rewards_on_tie'],
            file_name_base=str(game_id)
        )

        new_player.set_q(S,config_dict['bid_periods'])

        player_list = player_list + [new_player]

    #begin episodes
    i = 0
    while i<config_dict['episodes']+1:
        i = i+1

        if config_dict['randomise_turn_order']:
            random.shuffle(player_list)
            logging.info('Shuffled player list. Turn order: {}'.format([p.player_id for p in player_list]))

        logging.info('Begin episode {0} of {1}'.format(i, config_dict['episodes']-1))
        s = env.get_initial_state(S, config_dict['initial_state_random'])

        #begin bidding
        for t in range(config_dict['bid_periods']):
            is_final_period = False if t < config_dict['bid_periods'] - 1 else True
            logging.info('Begin bidding period {0}, final period: {1}, state: {2}'.format(t, is_final_period, S[s]))

            #collect action history for auction in dict at
            at = {}
            state_before_actions=s
            for o,p in enumerate(player_list):
                a = p.select_action(t,s)
                at[p.player_id] = {}
                at[p.player_id]['order'] = o
                at[p.player_id]['action'] = a
                at[p.player_id]['state'] = s
                p.write_path_log_entry(log_args=(i, t, s, a, is_final_period))
                s = a
                logging.info('Update state to {0}'.format({s: S[s]}))

            for p,player in enumerate(player_list):
                player.update_q(t, state_before_actions, at, is_final_period)
                player.update_epsilon()

    logging.info('All episodes complete, printing path history and auction results for all agents...')

    path_dataframes = []
    results_dataframes = []

    for j,player in enumerate(player_list):
        player.path_df = player.get_path_log_from_hdf(player.get_serialised_file_name()+'.hdf')
        path_dataframes.append(player.path_df)
        player.serialise_agent()

        results_df = env.get_results_summary(path_dataframes, 100)
        results_path = player.get_serialised_file_name()+'_results.csv'
        results_df.to_csv(results_path,index=False)
        results_dataframes.append(results_df)

        #add results into the config dict to be saved together: grid search result aggregation needs results + params
        config_dict['path_df'] = player.path_df.to_dict()
        config_dict['results_df'] = results_df.to_dict()
        config_dict['player_id'] = player.player_id

        pickle_file = 'Player'+str(player.player_id)+'_'+config_dict['output_file']+'.pck'
        pickle_path = os.path.join(config_dict['output_folder'],pickle_file)
        with open(pickle_path, 'wb') as fp:
            pickle.dump(config_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

    final_result_df = pd.DataFrame()
    for result in results_dataframes:
        final_result_df = pd.concat([final_result_df,result],axis=1)
    final_result_path = os.path.join(config_dict['output_folder'],str(game_id)+'_results.csv')
    final_result_df.to_csv(final_result_path)

    #Produce final bids heatmap if multiple agent
    if config_dict['num_players'] > 1:
        n_bids = int(100)
        final_bids_df = env.get_last_x_bids_array(path_dataframes,n_bids)
        title = '{}, last {} games'.format(config_dict['q_update_mode'],n_bids)
        fig,axs = grap.plot_final_bids_heatmap(final_bids_df,config_dict['price_levels'],title)
        if config_dict['randomise_turn_order']:
            suptitle='Randomised Turn Order, share rewards on tie = {}'.format(config_dict['share_rewards_on_tie'])
            fig.suptitle(suptitle)
        else:
            suptitle='Turn Order: {}, share rewards on tie = {}'.format(', '.join([str(pl.player_id) for pl in player_list]),config_dict['share_rewards_on_tie'])
        fig.suptitle(suptitle)
        fig.savefig(player.get_serialised_file_name() + '_final_{}bids_heatmap.png'.format(str(n_bids)))

    return

if __name__ == '__main__':
    logging.basicConfig(filename='bidding.log'.format(dt.datetime.strftime(dt.datetime.now(), '%Y%m%d-%H%M%S')),
                        format='%(asctime)s - %(levelname)s - %(message)s', level=logging.DEBUG)
    logging.info('Process start')

    #if input args supplied via stdin, use these. Otherwise use config defaults below
    config_dict = env.interpret_args(sys.argv)
    if len(config_dict) == 0:
        config_dict = {
            # Auction parameters
            'episodes': 2000,
            'initial_state_random': False,

            # Environment parameters
            'bid_periods': 2,
            'price_levels': 3,
            'num_players': 2,
            'q_convergence_threshold':100,

            # Script run parameters
            'output_folder':r'./results',
            'output_file':'results.bat',

            # Player parameters
            'alpha': 0.8,
            'gamma': 0.5,
            'epsilon': 1,
            'epsilon_decay_1': 0.999,
            'epsilon_decay_2': 0.99,
            'epsilon_threshold': 0.4,
            'agent_valuation': 3.3,
            'q_update_mode':'foe',
            'randomise_turn_order':False,
            'share_rewards_on_tie':True
        }

    main()
    logging.info('Process end')