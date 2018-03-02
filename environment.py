import ast
import os
import sys
import pandas as pd
import collections
import itertools
import logging
import numpy as np
from math import ceil

def get_possible_states(num_price_levels, num_players):
    """
    Function generates a list of all possible states S, where each state is a named tuple of state attributes
    :param num_price_levels: number of discrete prices
    :param num_players: number of players in auction
    :return: list of possible states S
    """
    prices = (np.nan,) + tuple(range(num_price_levels))
    bid_combinations = list(itertools.product(*[prices for p in range(num_players)]))

    State = define_state()
    S = [State(current_winner=np.nan, current_bids=(b)) for b in bid_combinations]
    S = [set_winner(s) for s in S]
    logging.info('Declared {0} possible states, examples: \n{1}'.format(len(S), S[0:3]))
    return S

def get_initial_state(S,initial_state_random):
    if initial_state_random:
        s = np.random.choice(len(S))
        logging.info('Randomly initialize state to {0}'.format({s:S[s]}))
    else:
        s = 0
        logging.info('Set initial state to {0}'.format({s:S[s]}))

    return s


def set_winner(state):
    """
    Function reads in a State named tuple and returns a new State tuple with the field 'current_winner' overwritten
    using the values in the current_bids attribute
    :param state:
    :return:
    """
    state_values = state._asdict()
    state_values['current_winner'] = get_winner(state)
    State = define_state()
    return State(**state_values)

def get_winner(state):
    """
    Function returns a value for the current winner, given the current_bids in the input state
    :param state: input State
    :return: output State with current_winner attribute overwritten
    """
    if all([np.isnan(bid) for bid in state.current_bids]):
        return np.nan
    elif len(state.current_bids) == 1:
        return 0
    elif len(set(state.current_bids)) == 1:
        return np.nan
    else:
        for i, b in enumerate(state.current_bids):
            if b == np.nanmax(state.current_bids):
                return i

def define_state():
    return collections.namedtuple('State', 'current_winner current_bids')

def check_and_create_directory(dir):
    if not os.path.isdir(dir):
        try:
            os.mkdir(dir)
            logging.info('Created directory {0}'.format(dir))
        except Exception as ex:
            logging.error('Unable to create directory {0} \n Error:{1}'.format(dir,ex))
            return False

def get_environment_level_file_name(episodes, bid_periods, price_levels, num_players):
    file_name = 'Auction_E{0}_BP{1}_PL{2}_NP{3}'.format(episodes, bid_periods, price_levels, num_players)
    return file_name

def calc_rewards_vector(path_df, reward_vector_interval:int):
    rewards_vector = []
    for k in range(0,int(ceil(path_df['episode'].max()/reward_vector_interval))):
        start_idx = int(k*reward_vector_interval)
        end_idx = int((k+1)*reward_vector_interval)
        rewards_vector.append(sum(path_df.iloc[start_idx:end_idx]['reward'])/reward_vector_interval)
    return rewards_vector

def get_results_summary(path_dataframes:list,reward_vector_interval=1000):
    """
    Function reads several path_df dataframes (intended to be 1 per player for a single game)
    and returns a results summary in results_df format
    :param path_dataframes: path dataframes of players
    :return: results dataframe
    """
    results_df = pd.DataFrame(columns=['Player ID', 'Total Episodes', 'Player Converged', 'Period Converged','Avg Reward','Avg Reward Vector'])

    for path_df in path_dataframes:

        players = path_df['player_id'].drop_duplicates()
        if len(players) > 1:
            logging.warning('get_results_summary: multiple players in the same path_df! Results will be wrong!')
        player_id = players[0]

        total_episodes = path_df['episode'].max()
        convergence_status = path_df['q_converged'].tail(1).values[0]
        if convergence_status:
            period_converged = path_df[path_df['q_converged']]['episode'].max()
        else:
            period_converged = np.nan

        reward_per_episode = path_df.groupby(['episode'])['reward'].sum()
        avg_reward = round(sum(reward_per_episode)/total_episodes,2)

        reward_vector = calc_rewards_vector(path_df,reward_vector_interval)

        row_df = pd.DataFrame(columns=results_df.columns,index=[0])
        row_df['Player ID'] = player_id
        row_df['Total Episodes'] = total_episodes
        row_df['Player Converged'] = convergence_status
        row_df['Period Converged'] = period_converged
        row_df['Avg Reward'] = avg_reward
        row_df['Avg Reward Vector'] = '_'.join([str(round(x,4)) for x in reward_vector])

        results_df = results_df.append(row_df)

        return results_df

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